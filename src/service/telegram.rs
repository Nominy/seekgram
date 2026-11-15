//! The logged-in Telegram client and the read, draft and media-export actions built on it.

use std::{collections::HashMap, fs, sync::atomic::{AtomicU64, Ordering}, time::{Duration, Instant, SystemTime, UNIX_EPOCH}};
use grammers_client::{Client, Config, grammers_tl_types as tl, types::{Chat, Message, Downloadable, Media, Photo, media::Document}};
use grammers_session::{PackedType, Session};
use rusqlite::{params, Connection, OptionalExtension};
use serde_json::{json, Value};
use tokio::{io::AsyncWriteExt, sync::Mutex};
use uuid::Uuid;

use crate::paths;

static RECONNECT: grammers_mtsender::FixedReconnect = grammers_mtsender::FixedReconnect { attempts: 3, delay: Duration::from_secs(1) };

/// Connects with the stored session, or a new empty one when none exists yet (first login).
pub async fn connect(api_id: i32, api_hash: &str) -> Result<Client, String> {
    fs::create_dir_all(paths::data_dir()).map_err(|e| format!("Cannot create the data folder: {e}"))?;
    let session = Session::load_file_or_create(paths::session_file()).map_err(|e| format!("The saved Telegram login cannot be read: {e}"))?;
    let mut config = Config { session, api_id, api_hash: api_hash.to_owned(), params: Default::default() };
    // Telegram throttles file downloads with waits of a second or two whenever the account
    // downloads heavily, which indexing does constantly. The client sleeps through those and
    // repeats the request; longer waits surface as errors for the caller to handle.
    config.params.flood_sleep_threshold = 30;
    config.params.reconnection_policy = &RECONNECT;
    Client::connect(config).await.map_err(|e| format!("Could not reach Telegram: {e}"))
}
fn db() -> Result<Connection, String> {
    let c = Connection::open(paths::data_dir().join("drafts.sqlite3")).map_err(|e| e.to_string())?;
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
pub(crate) fn marked_id(chat: &Chat) -> i64 {
    match chat.pack().ty { PackedType::User | PackedType::Bot => chat.id(), PackedType::Chat => -chat.id(), _ => -1_000_000_000_000 - chat.id() }
}
fn message(m: &Message) -> Value { json!({"message_id":m.id(), "chat_id":marked_id(&m.chat()), "date":m.date().to_rfc3339(), "text":m.text(), "outgoing":m.outgoing()}) }

pub struct TelegramService {
    pub(crate) client: Client,
    cache: Mutex<HashMap<String, (Instant, Value)>>,
    /// Chats by marked ID from the last dialog listing. Listing dialogs for every
    /// lookup drew Telegram flood waits; a miss relists once, which also finds new chats.
    chats: Mutex<HashMap<i64, Chat>>,
    generation: AtomicU64,
}
impl TelegramService {
    /// Wraps a client that is already authorized.
    pub fn new(client: Client) -> Result<Self, String> {
        db()?;
        Ok(Self { client, cache: Mutex::new(HashMap::new()), chats: Mutex::new(HashMap::new()), generation: AtomicU64::new(0) })
    }
    pub async fn account_id(&self) -> Result<i64, String> {
        self.client.get_me().await.map(|user| user.id()).map_err(|e| format!("Fetching Telegram account failed: {e}"))
    }
    pub(crate) async fn resolve(&self, recipient: &str) -> Result<Chat, String> {
        let r = recipient.trim();
        if let Ok(id) = r.parse::<i64>() {
            // Held across the listing so concurrent misses share one listing.
            let mut chats = self.chats.lock().await;
            if let Some(chat) = chats.get(&id) { return Ok(chat.clone()); }
            let mut dialogs = self.client.iter_dialogs();
            while let Some(d) = dialogs.next().await.map_err(|e| format!("Resolving chat failed: {e}"))? {
                chats.insert(marked_id(d.chat()), d.chat().clone());
            }
            return chats.get(&id).cloned().ok_or_else(|| "Numeric chat ID not found in accessible dialogs; use a username for a public chat".to_string());
        }
        self.client.resolve_username(r.trim_start_matches('@')).await.map_err(|e| format!("Resolving username failed: {e}"))?.ok_or_else(|| "Username not found".to_string())
    }
    pub async fn dispatch(&self, action: &str, args: Value) -> Result<Value, String> {
        match action {
            "draft_create" => self.draft_create(&args),
            "drafts" => self.drafts(),
            "draft_send" => self.draft_send(&args).await,
            "media_view" => self.media_view(&args).await,
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
            peer: chat.pack().to_input_peer(), saved_peer_id: None, top_msg_id: None,
            filters: kinds.iter().map(|(_, filter)| filter.clone()).collect(),
        }).await.map_err(|e| format!("Counting chat media failed: {e}"))?;
        let mut result = json!({"chat_id": marked_id(&chat), "inexact": false});
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
            while items.len() < limit { match iter.next().await.map_err(|e| e.to_string())? { Some(d) => { let c = d.chat(); items.push(json!({"chat_id":marked_id(c),"name":c.name(),"username":c.username()})); }, None => break } }
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
    async fn media_view(&self, args: &Value) -> Result<Value, String> {
        const MAX_BYTES: usize = 256 * 1024 * 1024;
        let chat = self.resolve(string_arg(args,"chat")?).await?;
        let id = integer(args,"message_id",0,1,i32::MAX as i64)? as i32;
        let context_limit = integer(args,"context_limit",10,0,30)? as usize;
        let found = self.client.get_messages_by_id(&chat, &[id]).await.map_err(|e| format!("Fetching media message failed: {e}"))?;
        let m = found.into_iter().next().flatten().ok_or("Message not found")?;
        if marked_id(&m.chat()) != marked_id(&chat) { return Err("Message does not belong to requested chat".into()); }
        if m.raw.ttl_period.is_some() { return Err("Self-destructing messages cannot be exported".into()); }
        let media = match m.media().ok_or("Message has no downloadable media")? {
            Media::Sticker(sticker) => Media::Document(sticker.document),
            Media::WebPage(page) => match page.raw.webpage {
                tl::enums::WebPage::Page(web) => {
                    if let Some(photo) = web.photo {
                        Media::Photo(Photo::from_raw(photo))
                    } else if let Some(document) = web.document {
                        Media::Document(Document::from_raw_media(tl::types::MessageMediaDocument {
                            nopremium: false, spoiler: false, video: false, round: false,
                            voice: false, document: Some(document), alt_document: None, ttl_seconds: None,
                        }))
                    } else {
                        return Err("Web preview has no downloadable image or document".into());
                    }
                }
                _ => return Err("Web preview is unavailable".into()),
            },
            media => media,
        };
        let (mime, extension, expected_size) = match &media {
            Media::Photo(photo) => {
                if photo.ttl_seconds().is_some() { return Err("Self-destructing photos cannot be exported".into()); }
                ("image/jpeg".to_string(), "jpg".to_string(), None)
            }
            Media::Document(document) => {
                if document.raw.ttl_seconds.is_some() { return Err("Self-destructing documents cannot be exported".into()); }
                let size = document.size();
                if size <= 0 { return Err("Document size is unavailable or empty".into()); }
                if size > MAX_BYTES as i64 { return Err("Media exceeds the 256 MiB export limit".into()); }
                let mime = document.mime_type().ok_or("Document MIME type is unavailable")?.to_ascii_lowercase();
                let parts: Vec<&str> = mime.split('/').collect();
                if parts.len() != 2 || !matches!(parts[0], "image" | "audio" | "video")
                    || parts[1].is_empty() || !parts[1].bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'+' | b'-')) {
                    return Err("Unsupported document MIME type; expected image, audio, or video".into());
                }
                let ext = mime_guess::get_mime_extensions_str(&mime).and_then(|exts| exts.first()).ok_or("No safe file extension is known for this media MIME type")?;
                (mime, (*ext).to_string(), Some(size as usize))
            }
            _ => return Err("Message media is not an image, audio, or video document".into()),
        };
        if media.to_raw_input_location().is_none() { return Err("Media cannot be downloaded".into()); }
        let mut context = Vec::new();
        if context_limit > 0 { let mut iter = self.client.iter_messages(&chat).offset_id(id).limit(context_limit); while let Some(item) = iter.next().await.map_err(|e| format!("Fetching media context failed: {e}"))? { context.push(message(&item)); } }
        let dir = paths::cache_dir().join("views"); fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let path = dir.join(format!("{}.{}",Uuid::new_v4(), extension));
        let mut options = tokio::fs::OpenOptions::new(); options.write(true).create_new(true);
        #[cfg(unix)] { options.mode(0o600); }
        let mut file = options.open(&path).await.map_err(|e| e.to_string())?;
        let result: Result<usize, String> = async {
            let mut stream = self.client.iter_download(&Downloadable::Media(media));
            let mut total = 0usize;
            while let Some(chunk) = stream.next().await.map_err(|e| format!("Media download failed: {e}"))? {
                total = total.checked_add(chunk.len()).ok_or("Media size overflow")?;
                if total > MAX_BYTES { return Err("Media exceeds the 256 MiB export limit".into()); }
                file.write_all(&chunk).await.map_err(|e| e.to_string())?;
            }
            if total == 0 { return Err("Media download was empty".into()); }
            if expected_size.is_some_and(|size| size != total) { return Err("Media download size differs from Telegram metadata".into()); }
            file.sync_all().await.map_err(|e| e.to_string())?;
            Ok(total)
        }.await;
        drop(file);
        let size = match result { Ok(size) => size, Err(e) => { let _ = tokio::fs::remove_file(&path).await; return Err(e); } };
        tokio::task::spawn_blocking(move || super::cache::grew(size as u64));
        Ok(json!({"path":path.to_string_lossy(), "mime_type":mime, "size_bytes":size, "message":message(&m), "context_before":context, "fetched_at_unix":timestamp()}))
    }
}
