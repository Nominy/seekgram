//! The logged-in Telegram client and the read, draft and media-export actions built on it.

use std::{collections::HashMap, fs, num::NonZeroU32, path::Path, sync::{Arc, atomic::{AtomicU64, Ordering}}, time::{Duration, Instant, SystemTime, UNIX_EPOCH}};
use grammers_client::{Client, media::{Document, Downloadable as _, Media, Photo}, message::Message, peer::Peer, tl};
use grammers_mtsender::{AutoSleep, ConnectionParams, SenderPool, SenderPoolHandle};
use grammers_session::{types::{PeerId, PeerRef}, updates::UpdatesLike};
use rusqlite::{params, Connection as Database, OptionalExtension};
use serde_json::{json, Value};
use tokio::{sync::{Mutex, mpsc}, task::JoinHandle};
use uuid::Uuid;

use crate::paths;

use super::{download::Source, session_store::SessionStore};

/// Telegram answered with objects of another API layer than grammers'. Telegram serves an
/// authorization in the layer its most recent connection announced, so another program
/// using this login (an older Telethon script, say) switches Seekgram's replies to that
/// layer. A new connection announces ours again; see `TelegramService::reconnect`.
pub(crate) const UNREADABLE: &str = "Seekgram cannot read Telegram's replies. This happens when another program uses Seekgram's Telegram login; Seekgram reconnects and tries again in a minute.";

/// A client and the connection pool it sends through. The pool runs until `close`.
pub struct Connection {
    pub client: Client,
    pub session: Arc<SessionStore>,
    /// Updates Telegram pushes; taken by whoever listens to them.
    pub updates: Option<mpsc::Receiver<UpdatesLike>>,
    pool: SenderPoolHandle,
    runner: JoinHandle<()>,
}

impl Connection {
    /// Ends every connection to Telegram. Clients still held elsewhere fail from then on.
    pub async fn close(self) {
        self.pool.quit();
        let _ = self.runner.await;
    }
}

/// Opens the stored session, or a new empty one when none exists yet (first login).
/// Nothing is sent yet; the first request connects.
pub fn connect(api_id: i32) -> Result<Connection, String> {
    fs::create_dir_all(paths::data_dir()).map_err(|e| format!("Cannot create the data folder: {e}"))?;
    let session = Arc::new(SessionStore::open(&paths::session_db(), &paths::legacy_session_file())
        .map_err(|e| format!("The saved Telegram login cannot be read: {e}"))?);
    let params = ConnectionParams {
        // Telegram throttles file downloads with waits of a second or two whenever the account
        // downloads heavily, which indexing does constantly. The client sleeps through those and
        // repeats the request; longer waits surface as errors for the caller to handle. A dropped
        // connection is retried the same way, a second apart.
        retry_policy: Box::new(AutoSleep { tries: NonZeroU32::new(3).expect("nonzero"), threshold: Duration::from_secs(30), io_errors_as_flood_of: Some(Duration::from_secs(1)) }),
        ..Default::default()
    };
    let SenderPool { runner, handle, updates } = SenderPool::with_configuration(session.clone(), api_id, params);
    let pool = handle.thin.clone();
    let client = Client::new(handle);
    Ok(Connection { client, session, updates: Some(updates), pool, runner: tokio::spawn(runner.run()) })
}
fn db() -> Result<Database, String> {
    let c = Database::open(paths::data_dir().join("drafts.sqlite3")).map_err(|e| e.to_string())?;
    c.busy_timeout(Duration::from_secs(5)).map_err(|e| e.to_string())?;
    c.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; CREATE TABLE IF NOT EXISTS drafts (id TEXT PRIMARY KEY, chat TEXT NOT NULL, text TEXT NOT NULL, state TEXT NOT NULL, created_at INTEGER NOT NULL, result TEXT)").map_err(|e| e.to_string())?;
    Ok(c)
}
fn timestamp() -> u64 { SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs() }
fn string_arg<'a>(args: &'a Value, key: &str) -> Result<&'a str, String> { args.get(key).and_then(Value::as_str).filter(|s| !s.trim().is_empty()).ok_or_else(|| format!("{key} must be a nonempty string")) }
fn integer(args: &Value, key: &str, default: i64, min: i64, max: i64) -> Result<i64, String> {
    let value = match args.get(key) { None | Some(Value::Null) => default, Some(v) => v.as_i64().ok_or_else(|| format!("{key} must be an integer"))? };
    if !(min..=max).contains(&value) { return Err(format!("{key} must be between {min} and {max}")); } Ok(value)
}
/// The Bot API form of a chat ID, as the index and settings store it: users positive,
/// basic groups negative, channels and supergroups below -10^12. grammers' placeholder for
/// "the logged-in user", which only message senders carry, maps to 0, which no chat has.
pub(crate) fn marked_id(peer: PeerId) -> i64 {
    peer.bot_api_dialog_id().unwrap_or(0)
}
fn message(m: &Message) -> Value {
    let sender = m.sender();
    json!({"message_id":m.id(), "chat_id":marked_id(m.peer_id()), "date":m.date().to_string(), "text":m.text(), "outgoing":m.outgoing(),
        "sender_id":sender.map(|peer| marked_id(peer.id())), "sender_name":sender.and_then(Peer::name), "media":m.media().as_ref().map(media_summary)})
}
/// What a message carries, so a reader can tell an attachment from an empty message and
/// pick `file_download` or `media_view` without fetching it.
fn media_summary(media: &Media) -> Value {
    match media {
        Media::Photo(_) => json!({"kind":"photo"}),
        Media::Document(document) if document.raw.document.is_some() => json!({"kind":"document", "file_name":document.name().filter(|n| !n.is_empty()),
            "mime_type":document.mime_type(), "size_bytes":document.size()}),
        Media::Sticker(_) => json!({"kind":"sticker"}),
        Media::WebPage(_) => json!({"kind":"web_page"}),
        Media::Poll(_) => json!({"kind":"poll"}),
        Media::Contact(_) => json!({"kind":"contact"}),
        Media::Geo(_) | Media::GeoLive(_) | Media::Venue(_) => json!({"kind":"location"}),
        _ => json!({"kind":"other"}),
    }
}
/// Largest file `media_view` or `file_download` exports.
const MAX_EXPORT_BYTES: usize = 256 * 1024 * 1024;
/// `name` as a single file name Windows and Unix accept: no directories, reserved
/// characters or trailing dots, at most 150 characters. Telegram file names are
/// sender-controlled, so they must never reach the file system unchanged.
fn safe_file_name(name: &str) -> Option<String> {
    let base = name.rsplit(['/', '\\']).next().unwrap_or_default();
    let cleaned: String = base.chars().map(|c| if c.is_control() || matches!(c, '<' | '>' | ':' | '"' | '|' | '?' | '*') { '_' } else { c }).take(150).collect();
    let cleaned = cleaned.trim().trim_end_matches(['.', ' ']).to_string();
    let stem = cleaned.split('.').next().unwrap_or_default().to_ascii_uppercase();
    let reserved = matches!(stem.as_str(), "CON" | "PRN" | "AUX" | "NUL") || (stem.len() == 4 && (stem.starts_with("COM") || stem.starts_with("LPT")) && stem.as_bytes()[3].is_ascii_digit());
    (!cleaned.is_empty() && !reserved).then_some(cleaned)
}
/// The downloadable media of `m`: a sticker as its document, a web preview as its photo or document.
fn downloadable(m: &Message) -> Result<Media, String> {
    if matches!(&m.raw, tl::enums::Message::Message(raw) if raw.ttl_period.is_some()) { return Err("Self-destructing messages cannot be exported".into()); }
    let media = match m.media().ok_or("Message has no downloadable media")? {
        Media::Sticker(sticker) => Media::Document(sticker.document),
        Media::WebPage(page) => match page.raw.webpage {
            tl::enums::WebPage::Page(web) => {
                if let Some(photo) = web.photo {
                    Media::Photo(Photo::from_raw(photo))
                } else if let Some(document) = web.document {
                    Media::Document(document_media(document))
                } else {
                    return Err("Web preview has no downloadable image or document".into());
                }
            }
            _ => return Err("Web preview is unavailable".into()),
        },
        media => media,
    };
    match &media {
        Media::Photo(photo) if photo.ttl_seconds().is_some() => return Err("Self-destructing photos cannot be exported".into()),
        Media::Document(document) => {
            if document.raw.ttl_seconds.is_some() { return Err("Self-destructing documents cannot be exported".into()); }
            let size = document.size().unwrap_or(0);
            if size == 0 { return Err("Document size is unavailable or empty".into()); }
            if size > MAX_EXPORT_BYTES { return Err("Media exceeds the 256 MiB export limit".into()); }
        }
        Media::Photo(_) => {}
        _ => return Err("Message media is not a photo or a file".into()),
    }
    if media.to_raw_input_location().is_none() { return Err("Media cannot be downloaded".into()); }
    Ok(media)
}

/// A chat and the reference that lets this account act on it.
#[derive(Clone)]
pub(crate) struct Chat {
    pub(crate) peer: Peer,
    pub(crate) reference: PeerRef,
}

impl Chat {
    pub(crate) async fn new(peer: Peer) -> Result<Self, String> {
        let reference = peer.to_ref().await.map_err(|e| format!("The saved Telegram login cannot be read: {e}"))?
            .ok_or("Telegram did not say how to reach this chat")?;
        Ok(Self { peer, reference })
    }
    /// The chat a message is in.
    pub(crate) async fn of(message: &Message) -> Result<Self, String> {
        let peer = message.peer().cloned().ok_or("Telegram did not describe the message's chat")?;
        let reference = message.peer_ref().await.map_err(|e| format!("The saved Telegram login cannot be read: {e}"))?
            .ok_or("Telegram did not say how to reach the message's chat")?;
        Ok(Self { peer, reference })
    }
    pub(crate) fn id(&self) -> i64 { marked_id(self.reference.id) }
    pub(crate) fn name(&self) -> &str { self.peer.name().unwrap_or_default() }
}

impl From<&Chat> for PeerRef {
    fn from(chat: &Chat) -> Self { chat.reference }
}

/// `document` as message media, the form grammers downloads and sends.
pub(crate) fn document_media(document: tl::enums::Document) -> Document {
    Document::from_raw_media(tl::types::MessageMediaDocument {
        nopremium: false, spoiler: false, video: false, round: false, voice: false,
        document: Some(document), alt_documents: None, video_cover: None, video_timestamp: None, ttl_seconds: None,
    })
}

pub struct TelegramService {
    pub(crate) client: Client,
    pub(crate) downloads: Arc<super::download::Downloader>,
    pool: SenderPoolHandle,
    cache: Mutex<HashMap<String, (Instant, Value)>>,
    /// Chats by marked ID from the last dialog listing. Listing dialogs for every
    /// lookup drew Telegram flood waits; a miss relists once, which also finds new chats.
    chats: Mutex<HashMap<i64, Chat>>,
    generation: AtomicU64,
}
impl TelegramService {
    /// Wraps the client of a connection that is already authorized.
    pub fn new(connection: &Connection) -> Result<Self, String> {
        db()?;
        let client = connection.client.clone();
        let downloads = Arc::new(super::download::Downloader::new(client.clone(), connection.session.clone()));
        Ok(Self { client, downloads, pool: connection.pool.clone(), cache: Mutex::new(HashMap::new()), chats: Mutex::new(HashMap::new()), generation: AtomicU64::new(0) })
    }
    /// Drops every connection; the next request opens a new one, whose `initConnection`
    /// makes Telegram answer in grammers' API layer again (`UNREADABLE`).
    pub(crate) fn reconnect(&self) {
        eprintln!("Telegram: reconnecting to get replies in Seekgram's API layer again");
        for dc in 1..=5 { self.pool.disconnect_from_dc(dc); }
    }
    pub async fn account_id(&self) -> Result<i64, String> {
        self.client.get_me().await.map(|user| marked_id(user.id())).map_err(|e| format!("Fetching Telegram account failed: {e}"))
    }
    pub(crate) async fn resolve(&self, recipient: &str) -> Result<Chat, String> {
        let r = recipient.trim();
        if let Ok(id) = r.parse::<i64>() {
            // Held across the listing so concurrent misses share one listing.
            let mut chats = self.chats.lock().await;
            if let Some(chat) = chats.get(&id) { return Ok(chat.clone()); }
            let mut dialogs = self.client.iter_dialogs();
            while let Some(d) = dialogs.next().await.map_err(|e| format!("Resolving chat failed: {e}"))? {
                chats.insert(marked_id(d.peer_id()), Chat { peer: d.peer().clone(), reference: d.peer_ref() });
            }
            return chats.get(&id).cloned().ok_or_else(|| "Numeric chat ID not found in accessible dialogs; use a username for a public chat".to_string());
        }
        let peer = self.client.resolve_username(r.trim_start_matches('@')).await.map_err(|e| format!("Resolving username failed: {e}"))?.ok_or_else(|| "Username not found".to_string())?;
        Chat::new(peer).await
    }
    pub async fn dispatch(&self, action: &str, args: Value) -> Result<Value, String> {
        match action {
            "draft_create" => self.draft_create(&args),
            "drafts" => self.drafts(),
            "draft_send" => self.draft_send(&args).await,
            "media_view" => self.media_view(&args).await,
            "file_download" => self.file_download(&args).await,
            "media_counts" => self.media_counts(&args).await,
            "dialogs" | "history" | "search" => {
                let key = format!("{action}:{}", args);
                let generation = self.generation.load(Ordering::SeqCst);
                if let Some((when, value)) = self.cache.lock().await.get(&key) {
                    if when.elapsed() < Duration::from_secs(5) && self.generation.load(Ordering::SeqCst) == generation { let mut v = value.clone(); v["cache_age_ms"] = json!(when.elapsed().as_millis() as u64); return Ok(v); }
                }
                let mut value = self.read_action(action, &args).await?;
                value["fetched_at_unix"] = json!(timestamp()); value["cache_age_ms"] = json!(0);
                let mut cache = self.cache.lock().await;
                if self.generation.load(Ordering::SeqCst) == generation { if cache.len() >= 128 { cache.clear(); } cache.insert(key, (Instant::now(), value.clone())); }
                Ok(value)
            }
            _ => Err(format!("Unknown Telegram action: {action}")),
        }
    }
    /// Telegram's own per-chat counts of messages by media type: one cheap request, no history scan.
    async fn media_counts(&self, args: &Value) -> Result<Value, String> {
        use tl::enums::MessagesFilter as Filter;
        let chat = self.resolve(string_arg(args, "chat")?).await?;
        let kinds = [
            ("photos", Filter::InputMessagesFilterPhotos), ("videos", Filter::InputMessagesFilterVideo),
            ("gifs", Filter::InputMessagesFilterGif), ("voice", Filter::InputMessagesFilterVoice),
            ("music", Filter::InputMessagesFilterMusic), ("round_videos", Filter::InputMessagesFilterRoundVideo),
            ("documents", Filter::InputMessagesFilterDocument),
        ];
        let counters = self.client.invoke(&tl::functions::messages::GetSearchCounters {
            peer: (&chat.reference).into(), saved_peer_id: None, top_msg_id: None,
            filters: kinds.iter().map(|(_, filter)| filter.clone()).collect(),
        }).await.map_err(|e| format!("Counting chat media failed: {e}"))?;
        let mut result = json!({"chat_id": chat.id(), "inexact": false});
        for counter in counters {
            let tl::enums::messages::SearchCounter::Counter(counter) = counter;
            if let Some((name, _)) = kinds.iter().find(|(_, filter)| *filter == counter.filter) { result[*name] = json!(counter.count); }
            if counter.inexact { result["inexact"] = json!(true); }
        }
        Ok(result)
    }
    async fn read_action(&self, action: &str, args: &Value) -> Result<Value, String> {
        let limit = integer(args, "limit", 30, 1, 100)? as usize;
        if action == "dialogs" {
            let offset = integer(args, "offset", 0, 0, 10000)? as usize;
            let mut iter = self.client.iter_dialogs(); let mut items = Vec::new(); let mut skipped = 0;
            while skipped < offset { if iter.next().await.map_err(|e| e.to_string())?.is_none() { break; } skipped += 1; }
            while items.len() < limit { match iter.next().await.map_err(|e| e.to_string())? { Some(d) => { let c = d.peer(); items.push(json!({"chat_id":marked_id(d.peer_id()),"name":c.name(),"username":c.username()})); }, None => break } }
            let next_offset = if items.len() == limit { Some(offset + items.len()) } else { None };
            return Ok(json!({"dialogs":items,"next_offset":next_offset}));
        }
        let offset = integer(args, "offset_id", 0, 0, i32::MAX as i64)? as i32;
        let mut items = Vec::new();
        if action == "history" {
            let chat = self.resolve(string_arg(args, "chat")?).await?;
            let mut iter = self.client.iter_messages(&chat).offset_id(offset).limit(limit);
            while let Some(m) = iter.next().await.map_err(|e| format!("History failed: {e}"))? { items.push(message(&m)); }
        } else {
            let query = string_arg(args, "query")?;
            if args.get("chat").is_some_and(|v| !v.is_null()) {
                let chat = self.resolve(string_arg(args, "chat")?).await?;
                let mut iter = self.client.search_messages(&chat).query(query).offset_id(offset).limit(limit);
                while let Some(m) = iter.next().await.map_err(|e| format!("Search failed: {e}"))? { items.push(message(&m)); }
            } else {
                if offset != 0 { return Err("Use offset for global search; offset_id alone is ambiguous".into()); }
                let skip = integer(args, "offset", 0, 0, 10000)? as usize;
                let mut iter = self.client.search_all_messages().query(query).limit(skip + limit);
                for _ in 0..skip { if iter.next().await.map_err(|e| format!("Global search failed: {e}"))?.is_none() { break; } }
                while let Some(m) = iter.next().await.map_err(|e| format!("Global search failed: {e}"))? { items.push(message(&m)); }
                let next = if items.len() == limit { Some(skip + items.len()) } else { None };
                return Ok(json!({"messages":items, "next_offset":next}));
            }
        }
        let next_offset = if items.len() == limit && (action == "history" || args.get("chat").is_some_and(|v| !v.is_null())) { items.last().and_then(|v| v["message_id"].as_i64()) } else { None };
        Ok(json!({"messages":items,"next_offset_id":next_offset, "global_pagination_supported":action != "search" || args.get("chat").is_some_and(|v| !v.is_null())}))
    }
    fn draft_create(&self, args: &Value) -> Result<Value, String> {
        let chat = string_arg(args,"chat")?; let text = string_arg(args,"text")?;
        if text.chars().count() > 4096 { return Err("Draft text exceeds 4096 characters".into()); }
        let id = Uuid::new_v4().to_string();
        db()?.execute("INSERT INTO drafts (id,chat,text,state,created_at) VALUES (?1,?2,?3,'draft',?4)", params![id,chat,text,timestamp() as i64]).map_err(|e| e.to_string())?;
        Ok(json!({"draft_id":id,"state":"draft"}))
    }
    fn drafts(&self) -> Result<Value, String> {
        let conn = db()?; let mut stmt = conn.prepare("SELECT id,chat,text,state,created_at,result FROM drafts ORDER BY created_at DESC, id DESC LIMIT 1000").map_err(|e| e.to_string())?;
        let rows = stmt.query_map([], |r| { let result: Option<String> = r.get(5)?; Ok(json!({"draft_id":r.get::<_,String>(0)?,"chat":r.get::<_,String>(1)?,"text":r.get::<_,String>(2)?,"state":r.get::<_,String>(3)?,"created_at_unix":r.get::<_,i64>(4)?,"result":result.and_then(|s| serde_json::from_str::<Value>(&s).ok())})) }).map_err(|e| e.to_string())?;
        let values: Result<Vec<_>,_> = rows.collect(); Ok(json!({"drafts":values.map_err(|e| e.to_string())?, "limit":1000}))
    }
    async fn draft_send(&self, args: &Value) -> Result<Value, String> {
        let id = string_arg(args,"draft_id")?;
        let row: Option<(String,String,String)> = db()?.query_row("SELECT chat,text,state FROM drafts WHERE id=?1", [id], |r| Ok((r.get(0)?,r.get(1)?,r.get(2)?))).optional().map_err(|e| e.to_string())?;
        let (recipient,text,state) = row.ok_or("Draft not found")?;
        if state != "draft" { return Err(format!("Draft state is {state}; automatic resend is forbidden. Reconcile with Telegram before creating a new draft")); }
        let chat = self.resolve(&recipient).await?;
        let changed = db()?.execute("UPDATE drafts SET state='unknown' WHERE id=?1 AND state='draft'", [id]).map_err(|e| e.to_string())?;
        if changed != 1 { return Err("Draft was already claimed; automatic resend is forbidden".into()); }
        self.generation.fetch_add(1, Ordering::SeqCst); self.cache.lock().await.clear();
        let outcome = self.client.send_message(&chat, text.as_str()).await;
        self.generation.fetch_add(1, Ordering::SeqCst); self.cache.lock().await.clear();
        let sent = outcome.map_err(|e| format!("Send outcome unknown; do not retry this draft; reconcile with Telegram: {e}"))?;
        let result = message(&sent);
        db()?.execute("UPDATE drafts SET state='sent', result=?2 WHERE id=?1 AND state='unknown'", params![id,result.to_string()]).map_err(|e| format!("Message sent but receipt could not be persisted; draft remains unknown: {e}"))?;
        Ok(json!({"draft_id":id,"state":"sent","message":result}))
    }
    /// The message `message_id` in `chat`, checked to belong to it.
    async fn fetch_message(&self, args: &Value) -> Result<(Chat, i32, Message), String> {
        let chat = self.resolve(string_arg(args,"chat")?).await?;
        let id = integer(args,"message_id",0,1,i32::MAX as i64)? as i32;
        let found = self.client.get_messages_by_id(&chat, &[id]).await.map_err(|e| format!("Fetching media message failed: {e}"))?;
        let m = found.into_iter().next().flatten().ok_or("Message not found")?;
        if m.peer_id() != chat.reference.id { return Err("Message does not belong to requested chat".into()); }
        Ok((chat, id, m))
    }
    /// Downloads `media` into the new file `path`, which is removed again on failure.
    async fn save_media(&self, media: Media, path: &Path) -> Result<u64, String> {
        let mut options = tokio::fs::OpenOptions::new(); options.write(true).create_new(true);
        #[cfg(unix)] { options.mode(0o600); }
        let mut file = options.open(path).await.map_err(|e| e.to_string())?;
        let result: Result<u64, String> = async {
            let total = self.downloads.save(&Source::Media(media), &mut file, MAX_EXPORT_BYTES as u64, |_| {}, |_| async {})
                .await.map_err(|error| error.to_string())?;
            file.sync_all().await.map_err(|e| e.to_string())?;
            Ok(total)
        }.await;
        drop(file);
        match result {
            Ok(size) => { tokio::task::spawn_blocking(move || super::cache::grew(size)); Ok(size) }
            Err(e) => { let _ = tokio::fs::remove_file(path).await; Err(e) }
        }
    }
    async fn media_view(&self, args: &Value) -> Result<Value, String> {
        let context_limit = integer(args,"context_limit",10,0,30)? as usize;
        let (chat, id, m) = self.fetch_message(args).await?;
        let media = downloadable(&m)?;
        let (mime, extension) = match &media {
            Media::Document(document) => {
                let mime = document.mime_type().ok_or("Document MIME type is unavailable")?.to_ascii_lowercase();
                let parts: Vec<&str> = mime.split('/').collect();
                if parts.len() != 2 || !matches!(parts[0], "image" | "audio" | "video")
                    || parts[1].is_empty() || !parts[1].bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'+' | b'-')) {
                    return Err("Unsupported document MIME type; expected image, audio, or video. Use file_download for other files".into());
                }
                let ext = mime_guess::get_mime_extensions_str(&mime).and_then(|exts| exts.first()).ok_or("No safe file extension is known for this media MIME type")?;
                (mime, (*ext).to_string())
            }
            _ => ("image/jpeg".to_string(), "jpg".to_string()),
        };
        let mut context = Vec::new();
        if context_limit > 0 { let mut iter = self.client.iter_messages(&chat).offset_id(id).limit(context_limit); while let Some(item) = iter.next().await.map_err(|e| format!("Fetching media context failed: {e}"))? { context.push(message(&item)); } }
        let dir = paths::cache_dir().join("views"); fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let path = dir.join(format!("{}.{}",Uuid::new_v4(), extension));
        let size = self.save_media(media, &path).await?;
        Ok(json!({"path":path.to_string_lossy(), "mime_type":mime, "size_bytes":size, "message":message(&m), "context_before":context, "fetched_at_unix":timestamp()}))
    }
    /// Saves any attached file (documents of every type, photos) under its sanitized original
    /// name, in a fresh folder of the media cache so names never collide.
    async fn file_download(&self, args: &Value) -> Result<Value, String> {
        let (_, id, m) = self.fetch_message(args).await?;
        let media = downloadable(&m)?;
        let (mime, name) = match &media {
            Media::Document(document) => {
                let mime = document.mime_type().map(str::to_ascii_lowercase);
                let ext = mime.as_deref().and_then(mime_guess::get_mime_extensions_str).and_then(|exts| exts.first()).copied().unwrap_or("bin");
                (mime.clone(), document.name().and_then(safe_file_name).unwrap_or_else(|| format!("{id}.{ext}")))
            }
            _ => (Some("image/jpeg".to_string()), format!("{id}.jpg")),
        };
        let dir = paths::cache_dir().join("files").join(Uuid::new_v4().to_string());
        fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let path = dir.join(&name);
        let size = self.save_media(media, &path).await?;
        Ok(json!({"path":path.to_string_lossy(), "file_name":name, "mime_type":mime, "size_bytes":size, "message":message(&m), "fetched_at_unix":timestamp()}))
    }
}

#[cfg(test)]
mod tests {
    use super::safe_file_name;

    #[test]
    fn file_names_cannot_escape_or_be_reserved() {
        assert_eq!(safe_file_name("..\\..\\Windows\\evil.dll").as_deref(), Some("evil.dll"));
        assert_eq!(safe_file_name("a/b/МВ_Лаб.doc").as_deref(), Some("МВ_Лаб.doc"));
        assert_eq!(safe_file_name("what?:*.pdf").as_deref(), Some("what___.pdf"));
        assert_eq!(safe_file_name("report. . ").as_deref(), Some("report"));
        assert_eq!(safe_file_name("CON.txt"), None);
        assert_eq!(safe_file_name("com1"), None);
        assert_eq!(safe_file_name(".."), None);
        assert_eq!(safe_file_name("   "), None);
    }
}
