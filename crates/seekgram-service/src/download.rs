//! Shared Telegram file transport: pipelined parts, sticky DC routing and bounded memory.

use std::{collections::HashSet, fmt, future::Future, io::{self, SeekFrom}, sync::atomic::{AtomicI32, AtomicU64, Ordering}, time::{Duration, Instant}};

use futures_util::{stream::FuturesUnordered, StreamExt};
use grammers_client::{Client, InvocationError, grammers_tl_types as tl, types::{Downloadable, Media, photo_sizes::PhotoSize}};
use tokio::{io::{AsyncSeek, AsyncSeekExt, AsyncWrite, AsyncWriteExt}, sync::{Mutex, Semaphore}, time::timeout};

const PART: u64 = 512 * 1024;
const PIPELINE: usize = 4;
const REQUEST_TIMEOUT: Duration = Duration::from_secs(180);
static NEXT_DOWNLOAD: AtomicU64 = AtomicU64::new(1);

#[derive(Debug)]
pub(crate) enum Error {
    Telegram(InvocationError),
    Io(io::Error),
    Timeout,
    Deferred,
    TooLarge,
    Empty,
    SizeMismatch,
    Protocol(&'static str),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Telegram(error) => write!(f, "Telegram download failed: {error}"),
            Self::Io(error) => write!(f, "Cannot save downloaded media: {error}"),
            Self::Timeout => f.write_str("Telegram download request timed out"),
            Self::Deferred => f.write_str("Telegram repeatedly deferred this file part; retry this media later"),
            Self::TooLarge => f.write_str("Media exceeds the download byte limit"),
            Self::Empty => f.write_str("Empty or unavailable media"),
            Self::SizeMismatch => f.write_str("Media download size differs from Telegram metadata"),
            Self::Protocol(reason) => f.write_str(reason),
        }
    }
}

impl From<io::Error> for Error {
    fn from(error: io::Error) -> Self { Self::Io(error) }
}

pub(crate) struct Downloader {
    client: Client,
    // Shared by indexing and interactive exports, not multiplied by file concurrency.
    requests: Semaphore,
    // grammers 0.7 races cold invoke_in_dc calls into duplicate auth exports. Serialize
    // only the first successful call to a DC; subsequent calls share its connection.
    connected: Mutex<HashSet<i32>>,
    cooldown: Mutex<Instant>,
}

impl Downloader {
    pub(crate) fn new(client: Client) -> Self {
        Self { client, requests: Semaphore::new(8), connected: Mutex::new(HashSet::new()), cooldown: Mutex::new(Instant::now()) }
    }

    pub(crate) async fn save<F, Fut>(
        &self,
        media: &Downloadable,
        file: &mut tokio::fs::File,
        max_bytes: u64,
        mut progress: impl FnMut(u64),
        on_flood: F,
    ) -> Result<u64, Error>
    where
        F: Fn(u64) -> Fut + Sync,
        Fut: Future<Output = ()> + Send,
    {
        if let Downloadable::PhotoSize(PhotoSize::Cached(photo)) = media {
            if photo.bytes.is_empty() { return Err(Error::Empty); }
            if photo.bytes.len() as u64 > max_bytes { return Err(Error::TooLarge); }
            file.write_all(&photo.bytes).await?;
            file.flush().await?;
            progress(photo.bytes.len() as u64);
            return Ok(photo.bytes.len() as u64);
        }
        let size = match media {
            Downloadable::Media(Media::Document(document)) => Some(document.size()),
            Downloadable::Media(Media::Sticker(sticker)) => Some(sticker.document.size()),
            Downloadable::PhotoSize(photo) => Some(photo.size() as i64),
            _ => None,
        };
        if size.is_some_and(|size| size <= 0) { return Err(Error::Empty); }
        let location = media.to_raw_input_location().ok_or(Error::Empty)?;
        // File traffic belongs on the DC carried by the media, not necessarily the
        // account's home DC. PhotoSize omits that metadata, so it needs discovery.
        let home_dc = self.client.session().get_user().map(|user| user.dc);
        let source_dc = media_dc(media);
        let dc = AtomicI32::new(route_dc(home_dc, source_dc)?);
        let id = NEXT_DOWNLOAD.fetch_add(1, Ordering::Relaxed);
        let started = Instant::now();
        let mut published = started;
        let mut received = 0u64;
        eprintln!("download: {}", serde_json::json!({"id":id,"state":"started","total_bytes":size,"source_dc":source_dc,"home_dc":home_dc}));
        let fetch = |offset| {
            let (location, dc, on_flood) = (&location, &dc, &on_flood);
            async move {
                retry_flood(
                    || async { self.ready().await; self.part(location, dc, offset).await },
                    |seconds| async move {
                        let until = Instant::now() + Duration::from_secs(seconds.max(1));
                        let mut cooldown = self.cooldown.lock().await;
                        *cooldown = (*cooldown).max(until);
                        drop(cooldown);
                        eprintln!("download: {}", serde_json::json!({"id":id,"state":"flood_wait","offset":offset,"wait_seconds":seconds}));
                        on_flood(seconds).await;
                        self.ready().await;
                    },
                ).await
            }
        };
        let result = transfer(file, size.map(|size| size as u64), max_bytes, fetch, |bytes| {
            received += bytes;
            progress(bytes);
            if published.elapsed() >= Duration::from_secs(1) {
                eprintln!("download: {}", serde_json::json!({"id":id,"state":"downloading","bytes":received,"total_bytes":size,"elapsed_ms":started.elapsed().as_millis(),"dc":dc.load(Ordering::Relaxed)}));
                published = Instant::now();
            }
        }).await;
        eprintln!("download: {}", serde_json::json!({"id":id,"state":if result.is_ok() {"completed"} else {"failed"},"bytes":received,"total_bytes":size,"elapsed_ms":started.elapsed().as_millis(),"dc":dc.load(Ordering::Relaxed)}));
        result
    }

    async fn ready(&self) {
        loop {
            let remaining = self.cooldown.lock().await.saturating_duration_since(Instant::now());
            if remaining.is_zero() { return; }
            tokio::time::sleep(remaining).await;
        }
    }

    async fn part(&self, location: &tl::enums::InputFileLocation, dc: &AtomicI32, offset: u64) -> Result<Vec<u8>, Error> {
        let _permit = self.requests.acquire().await.expect("download semaphore is never closed");
        self.ready().await;
        timeout(REQUEST_TIMEOUT, async {
            let request = tl::functions::upload::GetFile {
                precise: false, cdn_supported: false, location: location.clone(),
                offset: offset as i64, limit: PART as i32,
            };
            // Bound bad migration loops instead of letting malformed RPC data index
            // grammers' fixed DC address table or keep a request alive forever.
            for _ in 0..5 {
                let route = dc.load(Ordering::Relaxed);
                let result = if route == 0 {
                    self.client.invoke(&request).await
                } else {
                    let mut connected = self.connected.lock().await;
                    if connected.contains(&route) {
                        drop(connected);
                        self.client.invoke_in_dc(&request, route).await
                    } else {
                        let result = self.client.invoke_in_dc(&request, route).await;
                        if result.is_ok() { connected.insert(route); }
                        result
                    }
                };
                match result {
                    Ok(tl::enums::upload::File::File(file)) => return Ok(file.bytes),
                    Ok(tl::enums::upload::File::CdnRedirect(_)) => return Err(Error::Protocol("Unexpected Telegram CDN redirect (CDN was not requested)")),
                    Err(InvocationError::Rpc(error)) if error.code == 303 && error.name == "FILE_MIGRATE" => {
                        let Some(next @ 1..=5) = error.value else { return Err(Error::Protocol("Invalid Telegram file datacenter")); };
                        let next = route_dc(self.client.session().get_user().map(|user| user.dc), Some(next as i32))?;
                        let _ = dc.compare_exchange(route, next, Ordering::Relaxed, Ordering::Relaxed);
                    }
                    Err(error) => return Err(Error::Telegram(error)),
                }
            }
            Err(Error::Protocol("Telegram file datacenter migration loop"))
        }).await.map_err(|_| Error::Timeout)?
    }
}

fn media_dc(media: &Downloadable) -> Option<i32> {
    let document = match media {
        Downloadable::Media(Media::Document(document)) => Some(document),
        Downloadable::Media(Media::Sticker(sticker)) => Some(&sticker.document),
        Downloadable::Media(Media::Photo(photo)) => return match &photo.raw.photo {
            Some(tl::enums::Photo::Photo(photo)) => Some(photo.dc_id),
            _ => None,
        },
        Downloadable::UserProfilePhoto(photo) => return Some(photo.raw.dc_id),
        Downloadable::ChatPhoto(photo) => return Some(photo.raw.dc_id),
        _ => None,
    }?;
    match &document.raw.document {
        Some(tl::enums::Document::Document(document)) => Some(document.dc_id),
        _ => None,
    }
}

fn route_dc(home: Option<i32>, source: Option<i32>) -> Result<i32, Error> {
    match source {
        Some(dc) if !(1..=5).contains(&dc) => Err(Error::Protocol("Invalid Telegram file datacenter")),
        Some(dc) if Some(dc) != home => Ok(dc),
        _ => Ok(0),
    }
}

/// A server wait retries this request, not the enclosing file. Completed parts and
/// other pending responses stay owned by the transfer while it waits.
async fn retry_flood<F, Fut, W, Wait>(call: F, wait: W) -> Result<Vec<u8>, Error>
where
    F: Fn() -> Fut,
    Fut: Future<Output = Result<Vec<u8>, Error>>,
    W: Fn(u64) -> Wait,
    Wait: Future<Output = ()>,
{
    let mut waits = 0;
    loop {
        match call().await {
            Err(Error::Telegram(InvocationError::Rpc(error)))
                if error.code == 420 && matches!(error.name.as_str(), "FLOOD_WAIT" | "FLOOD_PREMIUM_WAIT") && error.value.is_some() => {
                eprintln!("download throttle: {}", serde_json::json!({"name":error.name,"seconds":error.value}));
                wait(u64::from(error.value.unwrap())).await;
                waits += 1;
                if waits == 3 { return Err(Error::Deferred); }
            }
            result => return result,
        }
    }
}

/// One writer places completed parts at their offsets. No spawned tasks or unbounded
/// channels: dropping this future drops every pending request and buffered response.
async fn transfer<W, F, Fut>(
    file: &mut W,
    size: Option<u64>,
    max_bytes: u64,
    fetch: F,
    mut progress: impl FnMut(u64),
) -> Result<u64, Error>
where
    W: AsyncWrite + AsyncSeek + Unpin,
    F: Fn(u64) -> Fut,
    Fut: Future<Output = Result<Vec<u8>, Error>>,
{
    if size == Some(0) { return Err(Error::Empty); }
    if size.is_some_and(|size| size > max_bytes) { return Err(Error::TooLarge); }
    let first = fetch(0).await?;
    check_part(0, first.len() as u64, size, max_bytes)?;
    file.write_all(&first).await?;
    let mut received = first.len() as u64;
    progress(received);
    let last = first.len() < PART as usize;
    drop(first);
    if let Some(size) = size {
        let mut pending = FuturesUnordered::new();
        let mut next = PART;
        let mut position = received;
        while next < size || !pending.is_empty() {
            while next < size && pending.len() < PIPELINE {
                let offset = next;
                let future = fetch(offset);
                pending.push(async move { (offset, future.await) });
                next += PART;
            }
            let (offset, bytes) = pending.next().await.expect("nonempty download pipeline");
            let bytes = bytes?;
            check_part(offset, bytes.len() as u64, Some(size), max_bytes)?;
            if position != offset { file.seek(SeekFrom::Start(offset)).await?; }
            file.write_all(&bytes).await?;
            position = offset + bytes.len() as u64;
            received += bytes.len() as u64;
            progress(bytes.len() as u64);
        }
    } else if !last {
        // Unknown-size photos cannot safely schedule beyond EOF. Keep the same DC
        // and stop on a short part, enforcing the cap before writing any excess.
        loop {
            let bytes = fetch(received).await?;
            check_part(received, bytes.len() as u64, None, max_bytes)?;
            file.write_all(&bytes).await?;
            received += bytes.len() as u64;
            progress(bytes.len() as u64);
            if bytes.len() < PART as usize { break; }
        }
    }
    file.flush().await?;
    if received == 0 { return Err(Error::Empty); }
    Ok(received)
}

fn check_part(offset: u64, length: u64, size: Option<u64>, max_bytes: u64) -> Result<(), Error> {
    if length > PART { return Err(Error::SizeMismatch); }
    if offset.checked_add(length).is_none_or(|end| end > max_bytes) { return Err(Error::TooLarge); }
    if size.is_some_and(|size| length != PART.min(size.saturating_sub(offset))) { return Err(Error::SizeMismatch); }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{cell::Cell, io::Cursor, sync::{Arc, atomic::AtomicUsize}};
    use tokio::sync::Barrier;

    #[tokio::test]
    async fn pipelines_parts_and_writes_out_of_order_responses_at_their_offsets() {
        let size = PART * 4 + 17;
        let barrier = Barrier::new(PIPELINE);
        let active = Cell::new(0usize);
        let peak = Cell::new(0usize);
        let completed = std::cell::RefCell::new(Vec::new());
        let mut file = Cursor::new(Vec::new());
        let mut reported = 0;
        let run = transfer(&mut file, Some(size), size, |offset| {
            let (barrier, active, peak, completed) = (&barrier, &active, &peak, &completed);
            async move {
                if offset != 0 {
                    active.set(active.get() + 1);
                    peak.set(peak.get().max(active.get()));
                    barrier.wait().await;
                    // The first pending part only completes after all the later ones.
                    if offset == PART { while completed.borrow().len() < 3 { tokio::task::yield_now().await; } }
                    completed.borrow_mut().push(offset);
                    active.set(active.get() - 1);
                }
                Ok(vec![(offset / PART) as u8; PART.min(size - offset) as usize])
            }
        }, |bytes| reported += bytes);
        assert_eq!(timeout(Duration::from_secs(5), run).await.unwrap().unwrap(), size);
        assert_eq!(reported, size);
        assert_eq!(peak.get(), PIPELINE);
        assert_eq!(completed.borrow().last(), Some(&PART));
        for (part, bytes) in file.get_ref().chunks(PART as usize).enumerate() {
            assert!(bytes.iter().all(|byte| *byte == part as u8));
        }
        assert_eq!(file.get_ref().len() as u64, size);
    }

    #[tokio::test]
    async fn rejects_truncation_and_oversized_metadata_without_accepting_partial_files() {
        for size in [1, PART, PART + 17] {
            let mut file = Cursor::new(Vec::new());
            let result = transfer(&mut file, Some(size), size, |offset| async move {
                let length = PART.min(size - offset);
                Ok(vec![0; (if offset + length == size { length - 1 } else { length }) as usize])
            }, |_| {}).await;
            assert!(matches!(result, Err(Error::SizeMismatch)));
        }
        let result = transfer(&mut Cursor::new(Vec::new()), Some(PART + 1), PART,
            |_| async { panic!("must reject size before requesting data") }, |_| {}).await;
        assert!(matches!(result, Err(Error::TooLarge)));
    }

    #[tokio::test]
    async fn unknown_size_stops_at_eof_and_never_writes_past_limit() {
        for size in [0, 17, PART, PART + 17] {
            let mut file = Cursor::new(Vec::new());
            let calls = Cell::new(0);
            let result = transfer(&mut file, None, PART, |offset| {
                calls.set(calls.get() + 1);
                async move { Ok(vec![7; PART.min(size - offset) as usize]) }
            }, |_| {}).await;
            match size {
                0 => assert!(matches!(result, Err(Error::Empty))),
                size if size > PART => assert!(matches!(result, Err(Error::TooLarge))),
                _ => assert_eq!(result.unwrap(), size),
            }
            assert!(file.get_ref().len() as u64 <= PART);
            assert_eq!(calls.get(), if size < PART { 1 } else { 2 });
        }
    }

    fn rpc(name: &str, code: i32, value: Option<u32>) -> Error {
        Error::Telegram(InvocationError::Rpc(grammers_mtsender::RpcError {
            name: name.to_owned(), code, value, caused_by: None,
        }))
    }

    #[tokio::test]
    async fn repeated_flood_waits_retry_only_the_failed_part() {
        for name in ["FLOOD_WAIT", "FLOOD_PREMIUM_WAIT"] {
            let calls: [Cell<usize>; 3] = std::array::from_fn(|_| Cell::new(0));
            let waits = std::cell::RefCell::new(Vec::new());
            let size = PART * 2 + 7;
            let mut file = Cursor::new(Vec::new());
            let received = transfer(&mut file, Some(size), size, |offset| {
                let (calls, waits) = (&calls, &waits);
                async move {
                    let index = (offset / PART) as usize;
                    retry_flood(|| {
                        calls[index].set(calls[index].get() + 1);
                        let attempt = calls[index].get();
                        async move {
                            if index == 1 && attempt <= 2 { return Err(rpc(name, 420, Some(3))); }
                            Ok(vec![index as u8; PART.min(size - offset) as usize])
                        }
                    }, |seconds| { waits.borrow_mut().push(seconds); async {} }).await
                }
            }, |_| {}).await.unwrap();
            assert_eq!(received, size);
            assert_eq!(calls.map(|count| count.get()), [1, 3, 1]);
            assert_eq!(*waits.borrow(), [3, 3]);
            for (part, bytes) in file.get_ref().chunks(PART as usize).enumerate() {
                assert!(bytes.iter().all(|byte| *byte == part as u8));
            }
        }
    }

    #[tokio::test]
    async fn a_part_that_never_progresses_is_deferred_without_an_endless_retry_loop() {
        let calls = Cell::new(0);
        let waited = Cell::new(0);
        let result = retry_flood(|| {
            calls.set(calls.get() + 1);
            async { Err(rpc("FLOOD_WAIT", 420, Some(7))) }
        }, |seconds| { waited.set(waited.get() + seconds); async {} }).await;
        assert!(matches!(result, Err(Error::Deferred)));
        assert_eq!(calls.get(), 3);
        assert_eq!(waited.get(), 21);
    }

    #[tokio::test]
    async fn non_wait_errors_are_not_retried() {
        let calls = Cell::new(0);
        let result = retry_flood(|| {
            calls.set(calls.get() + 1);
            async { Err(rpc("FILE_REFERENCE_EXPIRED", 400, None)) }
        }, |_| async { panic!("not a flood wait") }).await;
        assert!(matches!(result, Err(Error::Telegram(_))));
        assert_eq!(calls.get(), 1);
    }

    struct InFlight(Arc<AtomicUsize>);
    impl Drop for InFlight {
        fn drop(&mut self) { self.0.fetch_sub(1, Ordering::SeqCst); }
    }

    #[tokio::test]
    async fn cancellation_drops_every_pending_part() {
        let active = Arc::new(AtomicUsize::new(0));
        let ready = tokio::sync::Notify::new();
        let mut file = Cursor::new(Vec::new());
        let run = transfer(&mut file, Some(PART * 10), PART * 10, |offset| {
            let (active, ready) = (active.clone(), &ready);
            async move {
                if offset == 0 { return Ok(vec![0; PART as usize]); }
                let _guard = InFlight(active.clone());
                if active.fetch_add(1, Ordering::SeqCst) + 1 == PIPELINE { ready.notify_one(); }
                std::future::pending().await
            }
        }, |_| {});
        {
            let run = std::pin::pin!(run);
            tokio::select! {
                _ = run => panic!("pending download completed"),
                _ = ready.notified() => assert_eq!(active.load(Ordering::SeqCst), PIPELINE),
                _ = tokio::time::sleep(Duration::from_secs(5)) => panic!("pipeline did not fill"),
            }
        }
        assert_eq!(active.load(Ordering::SeqCst), 0);
    }
}
