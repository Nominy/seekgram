//! Media search from a chat of the user's own.
//!
//! A message the user sends there starting with `?` is a search, and the matches are
//! posted back into that chat. It starts out as Saved Messages; `?here`, sent in a group
//! whose only member is the user, moves it there. This is the only place the server
//! sends without an explicit tool call, so it is deliberately narrow: it reacts only to
//! the account's own non-forwarded messages in that one chat, it only writes to that
//! chat, and that chat can never be one with other members.

use std::{path::Path, sync::Arc, time::Duration};

use grammers_client::{media::{Document, InputMedia, Media, Photo}, message::{InputMessage, Message}, peer::{Peer, User}, tl, update::Update};
use grammers_mtsender::UpdatesConfiguration;
use grammers_session::{types::PeerId, updates::UpdatesLike};
use serde_json::{json, Value};
use tokio::sync::mpsc;

use super::{download::Source, state::{SearchChat, Shared}, telegram::{marked_id, Chat, TelegramService}, workers::Workers};

const PREFIX: &str = "?";
const PAGE: usize = 10;
const HELP: &str = "Media search. Start a message here with ?\n\n\
?pink dragon  -  search images by words\n\
?  as a photo caption  -  find similar images (an album mixes its photos)\n\
?  in reply to a photo  -  more like that one\n\
?in snow  as caption or reply  -  the photo plus the words\n\
?video dragon  /  ?audio rain  -  search videos or audio\n\
?more  -  next 10 results\n\
?7  -  forward result number 7 itself\n\
?here  -  sent in a group only you are in: move search to that group\n\n\
Results come as one album. The numbers under it open each original message; if a number does \
nothing in your app, use ?N instead. Only indexed chats are searched.";

/// The previous search, kept so `?more` can continue it.
struct Last {
    label: String,
    hits: Vec<(i64, i32)>,
    next: usize,
}

/// Publishes where searches are answered, for the control panel.
fn publish(shared: &Shared, home: &Chat, me_id: i64) {
    let saved = home.id() == me_id;
    shared.update(|status| status.search_chat = SearchChat {
        kind: if saved { "saved" } else { "group" }.to_owned(),
        id: (!saved).then(|| home.id()),
        title: if saved { "Saved Messages".to_owned() } else { home.name().to_owned() },
    });
}

/// Opens the configured search chat. That chat is the only place results may go, so a
/// group that cannot be opened is waited for, never replaced by another chat.
async fn open_home(shared: &Shared, service: &TelegramService, me: &User) -> Option<Chat> {
    loop {
        let opened = match shared.settings().search_chat {
            None => Chat::new(Peer::User(me.clone())).await,
            Some(id) => service.resolve(&id.to_string()).await,
        };
        match opened {
            Ok(chat) => return Some(chat),
            Err(error) => {
                eprintln!("Chat search: cannot open the configured chat: {error}");
                tokio::select! {
                    _ = tokio::time::sleep(Duration::from_secs(30)) => {}
                    _ = shared.stopped() => return None,
                }
            }
        }
    }
}

/// Answers searches in a private group: the one used before, if it still exists and is
/// still only the user's, otherwise a newly created one.
pub async fn use_group(shared: &Shared, service: &TelegramService) -> Result<(), String> {
    if let Some(id) = shared.settings().search_group {
        let me = service.account_id().await?;
        if let Ok(chat) = service.resolve(&id.to_string()).await {
            if private_to(service, &chat, me).await.is_ok() {
                return shared.set_settings(|settings| settings.search_chat = Some(id));
            }
        }
    }
    create_group(shared, service).await
}

/// Creates a private group for searching and makes it the search chat.
async fn create_group(shared: &Shared, service: &TelegramService) -> Result<(), String> {
    let updates = service.client.invoke(&tl::functions::channels::CreateChannel {
        broadcast: false, megagroup: true, for_import: false, forum: false,
        title: "Search".to_owned(), about: "Seekgram media search. Send ? for help.".to_owned(),
        geo_point: None, address: None, ttl_period: None,
    }).await.map_err(|error| format!("Telegram did not create the group: {error}"))?;
    let chats = match updates {
        tl::enums::Updates::Updates(updates) => updates.chats,
        tl::enums::Updates::Combined(updates) => updates.chats,
        _ => Vec::new(),
    };
    let id = chats.iter().find_map(|chat| match chat { tl::enums::Chat::Channel(channel) => Some(channel.id), _ => None })
        .ok_or("Telegram created the group but did not say which one; send ?here in it to use it")?;
    let group = -1_000_000_000_000 - id;
    shared.set_settings(|settings| { settings.search_chat = Some(group); settings.search_group = Some(group); })
}

/// Whether `chat` may become the search chat: Saved Messages, or a group with no member
/// but the user. Results are private pictures, so they must never land in front of anyone else.
async fn private_to(service: &TelegramService, chat: &Chat, me_id: i64) -> Result<(), String> {
    match &chat.peer {
        Peer::User(user) if marked_id(user.id()) == me_id => Ok(()),
        Peer::User(_) => Err("that is a chat with someone else".into()),
        Peer::Channel(_) => Err("that is a channel; use a group".into()),
        Peer::Community(_) => Err("that is a community; use a group".into()),
        Peer::Group(_) => {
            let mut members = service.client.iter_participants(chat);
            while let Some(member) = members.next().await.map_err(|error| format!("cannot list its members: {error}"))? {
                if marked_id(member.user.id()) != me_id { return Err("that group has other members".into()); }
            }
            Ok(())
        }
    }
}

/// Answers searches until shutdown. `me` is the logged-in account; `updates` are the ones
/// Telegram pushes to its client.
pub async fn run(shared: Arc<Shared>, service: Arc<TelegramService>, workers: Arc<Workers>, me: User, updates: mpsc::Receiver<UpdatesLike>) {
    let me_id = marked_id(me.id());
    let mut stream = match service.client.stream_updates(updates, UpdatesConfiguration::default()).await {
        Ok(stream) => stream,
        Err(error) => return eprintln!("Chat search: cannot receive Telegram's updates: {error}"),
    };
    let Some(mut home) = open_home(&shared, &service, &me).await else { return };
    publish(&shared, &home, me_id);
    let mut configured = shared.settings().search_chat;
    let mut changed = shared.changed.subscribe();
    let mut last: Option<Last> = None;
    eprintln!("Chat search: listening");
    loop {
        let update = tokio::select! {
            update = stream.next() => update,
            _ = changed.changed() => {
                // The control panel moved the search chat.
                if shared.settings().search_chat != configured {
                    configured = shared.settings().search_chat;
                    let Some(chat) = open_home(&shared, &service, &me).await else { return };
                    home = chat;
                    last = None;
                    publish(&shared, &home, me_id);
                }
                continue;
            }
            _ = shared.stopped() => return,
        };
        let message = match update {
            Ok(Update::NewMessage(message)) => message.into_inner(),
            Ok(_) => continue,
            Err(error) => {
                eprintln!("Chat search: update stream error: {error}");
                tokio::time::sleep(Duration::from_secs(5)).await;
                continue;
            }
        };
        let chat_id = marked_id(message.peer_id());
        // Telegram does not flag Saved Messages posts as outgoing, so authorship is checked three ways.
        // Forwarded messages never count: a forwarded result must not be able to trigger anything.
        let mine = message.forward_header().is_none() && (message.outgoing()
            || chat_id == me_id
            || message.sender_id().is_some_and(|sender| sender == PeerId::self_user() || marked_id(sender) == me_id));
        let Some(rest) = message.text().trim().strip_prefix(PREFIX).filter(|_| mine) else { continue };
        let rest = rest.trim().to_owned();
        let here = chat_id == home.id();
        if rest.eq_ignore_ascii_case("here") {
            // Any refusal is said in the current search chat, never in the chat that was refused.
            let saved = chat_id == me_id;
            let target = (!saved).then_some(chat_id);
            let moved = match Chat::of(&message).await {
                Ok(chat) => match private_to(&service, &chat, me_id).await {
                    Ok(()) => shared.set_settings(|settings| {
                        if target.is_some() { settings.search_group = target; }
                        else if settings.search_chat.is_some() { settings.search_group = settings.search_chat; }
                        settings.search_chat = target;
                    }).map(|()| chat),
                    Err(error) => Err(error),
                },
                Err(error) => Err(error),
            };
            let reply = match moved {
                Ok(chat) => {
                    home = chat;
                    configured = target;
                    last = None;
                    publish(&shared, &home, me_id);
                    "Search now lives in this chat. Send ? for help.".to_owned()
                }
                Err(error) => format!("Search was not moved: {error}."),
            };
            if let Err(error) = service.client.send_message(&home, reply).await { eprintln!("Chat search: cannot reply: {error}"); }
            continue;
        }
        if !here { continue; }
        let mut progress = Progress { service: &service, home: &home, message: None };
        let reply = handle(&service, &workers, &home, me_id, &message, &rest, &mut last, &mut progress).await
            .unwrap_or_else(|error| format!("Search failed: {error}"));
        // The progress line is removed so the summary lands below the results it describes.
        if let Some(status) = progress.message.take() { let _ = status.delete().await; }
        // An empty reply means the album's own caption already said everything.
        if reply.is_empty() { continue; }
        if let Err(error) = service.client.send_message(&home, reply).await {
            eprintln!("Chat search: cannot reply: {error}");
        }
    }
}

/// One message, edited in place, that shows what a running search is doing.
struct Progress<'a> {
    service: &'a TelegramService,
    home: &'a Chat,
    message: Option<Message>,
}

impl Progress<'_> {
    /// Cosmetic only: a failed send or edit never fails the search.
    async fn show(&mut self, text: String) {
        match &self.message {
            Some(message) => { let _ = message.edit(text).await; }
            None => self.message = self.service.client.send_message(self.home, text).await.ok(),
        }
    }
}

async fn handle(service: &TelegramService, workers: &Arc<Workers>, home: &Chat, me: i64, message: &Message, rest: &str, last: &mut Option<Last>, progress: &mut Progress<'_>) -> Result<String, String> {
    if rest.eq_ignore_ascii_case("more") {
        let state = last.as_mut().ok_or("nothing to continue; send a search first")?;
        return page(service, home, state, progress).await;
    }
    if let Ok(rank) = rest.parse::<usize>() {
        let state = last.as_ref().ok_or("nothing to open; send a search first")?;
        let &(chat_id, message_id) = rank.checked_sub(1).and_then(|index| state.hits.get(index))
            .ok_or_else(|| format!("there are results 1 to {}", state.hits.len()))?;
        forward(service, home, chat_id, message_id).await?;
        return Ok(String::new());
    }
    let (first, tail) = rest.split_once(char::is_whitespace).unwrap_or((rest, ""));
    let (modality, text) = match first.to_ascii_lowercase().as_str() {
        kind @ ("video" | "audio") => (kind.to_owned(), tail.trim()),
        _ => ("image".to_owned(), rest),
    };
    if message.photo().is_some() || message.grouped_id().is_some() || message.reply_to_message_id().is_some() {
        progress.show("Reading the photos...".to_owned()).await;
    }
    let images = references(service, home, message).await?;
    if text.is_empty() && images.is_empty() { return Ok(HELP.to_owned()); }
    let mut inputs: Vec<Value> = images.iter().map(|path| json!({"image_path": path})).collect();
    if !text.is_empty() { inputs.push(json!({"query": text})); }
    let search = super::search::dispatch(workers, "media_search", json!({"inputs": inputs, "modality": modality, "limit": 100}));
    tokio::pin!(search);
    let mut tick = tokio::time::interval(Duration::from_millis(1200));
    let mut frame = 0;
    let found = loop {
        tokio::select! {
            found = &mut search => break found,
            _ = tick.tick() => {
                progress.show(format!("Searching{}", ".".repeat(frame % 3 + 1))).await;
                frame += 1;
            }
        }
    };
    for path in &images { let _ = tokio::fs::remove_file(path).await; }
    let found = found?;
    if found["account_id"].as_i64() != Some(me) { return Err("the index belongs to another account".into()); }
    let mut hits: Vec<(i64, i32)> = Vec::new();
    for result in found["results"].as_array().ok_or("invalid search response")? {
        let hit = (result["chat_id"].as_i64().ok_or("invalid result")?, result["message_id"].as_i64().ok_or("invalid result")? as i32);
        // A video or audio message matches once per segment; show the message once.
        if !hits.contains(&hit) { hits.push(hit); }
    }
    if hits.is_empty() { return Ok("No matches.".to_owned()); }
    let label = match (images.len(), text.is_empty()) {
        (0, _) => format!("\"{text}\""),
        (count, true) => format!("{count} image(s)"),
        (count, false) => format!("{count} image(s) + \"{text}\""),
    };
    let state = last.insert(Last { label: if modality == "image" { label } else { format!("{modality}: {label}") }, hits, next: 0 });
    page(service, home, state, progress).await
}

/// Forwards one original message into Saved Messages, where tapping it jumps to its chat.
async fn forward(service: &TelegramService, home: &Chat, chat_id: i64, message_id: i32) -> Result<(), String> {
    let source = service.resolve(&chat_id.to_string()).await?;
    let sent = service.client.forward_messages(home, &[message_id], &source).await.map_err(|error| error.to_string())?;
    if sent.iter().any(Option::is_some) { Ok(()) } else { Err("message no longer exists".to_owned()) }
}

/// A link Telegram opens at the original message. Private chats and basic groups have
/// no web link, only the app scheme, which not every Telegram client handles.
fn link(chat_id: i64, message_id: i32) -> String {
    if chat_id <= -1_000_000_000_000 { format!("https://t.me/c/{}/{message_id}", -chat_id - 1_000_000_000_000) }
    else if chat_id < 0 { format!("tg://openmessage?chat_id={}&message_id={message_id}", -chat_id) }
    else { format!("tg://openmessage?user_id={chat_id}&message_id={message_id}") }
}

/// Sends the next page of results as one album whose caption links each picture to its
/// original; media Telegram does not allow in albums is forwarded. Any step Telegram
/// refuses is an error, never a different presentation. Returns the text still worth
/// saying, empty when the album's caption covered it.
async fn page(service: &TelegramService, home: &Chat, state: &mut Last, progress: &mut Progress<'_>) -> Result<String, String> {
    let start = state.next;
    let end = (start + PAGE).min(state.hits.len());
    if start >= end { return Ok(format!("No more results for {}.", state.label)); }
    let hits = &state.hits[start..end];
    progress.show(format!("Collecting {} results...", hits.len())).await;
    // One request per source chat instead of one per result.
    let mut originals: Vec<Option<Message>> = vec![None; hits.len()];
    let mut chats: Vec<i64> = hits.iter().map(|hit| hit.0).collect();
    chats.sort_unstable();
    chats.dedup();
    for chat_id in chats {
        let Ok(source) = service.resolve(&chat_id.to_string()).await else { continue };
        let ids: Vec<i32> = hits.iter().filter(|hit| hit.0 == chat_id).map(|hit| hit.1).collect();
        let Ok(found) = service.client.get_messages_by_id(&source, &ids).await else { continue };
        for message in found.into_iter().flatten() {
            if let Some(position) = hits.iter().position(|hit| *hit == (chat_id, message.id())) { originals[position] = Some(message); }
        }
    }
    // Telegram albums hold photos and videos only. Stickers and pictures sent as files are
    // re-encoded as photos so an image search arrives as one album. GIFs, video stickers
    // and audio cannot be in one at all; forwarding is how those are shown.
    let mut album: Vec<(usize, Media)> = Vec::new();
    let mut separate: Vec<usize> = Vec::new();
    let mut failed = 0;
    for (position, original) in originals.iter().enumerate() {
        // A result whose message was deleted since indexing is reported, not replaced.
        let Some(original) = original else { failed += 1; continue };
        let image_file = |document: &Document| document.size().unwrap_or(0) <= 20 * 1024 * 1024 && document.mime_type().is_some_and(|mime| mime.starts_with("image/"));
        match original.media() {
            Some(media @ Media::Photo(_)) => album.push((position, media)),
            Some(Media::Document(document)) if !document.is_animated() && document.mime_type().is_some_and(|mime| mime.starts_with("video/")) => album.push((position, Media::Document(document))),
            Some(Media::Document(document)) if image_file(&document) => album.push((position, as_photo(service, home, original).await.map_err(|error| format!("result {} could not be converted to a photo: {error}", start + position + 1))?)),
            Some(Media::Sticker(sticker)) if image_file(&sticker.document) => album.push((position, as_photo(service, home, original).await.map_err(|error| format!("result {} could not be converted to a photo: {error}", start + position + 1))?)),
            // A link's preview picture is what was indexed; show that picture, not the text message around it.
            Some(Media::WebPage(page)) => match page.raw.webpage {
                tl::enums::WebPage::Page(tl::types::WebPage { photo: Some(photo), .. }) => album.push((position, Media::Photo(Photo::from_raw(photo)))),
                _ => separate.push(position),
            },
            Some(_) => separate.push(position),
            None => failed += 1,
        }
    }
    if !album.is_empty() {
        // The caption opens with one numbered link per picture, in album order. Those are
        // ASCII, so entity offsets (UTF-16 units) equal byte offsets.
        let mut caption = String::from("Open: ");
        let mut entities = Vec::with_capacity(album.len());
        for (position, _) in &album {
            let number = (start + position + 1).to_string();
            let (chat_id, message_id) = hits[*position];
            entities.push(tl::enums::MessageEntity::TextUrl(tl::types::MessageEntityTextUrl {
                offset: caption.len() as i32, length: number.len() as i32, url: link(chat_id, message_id),
            }));
            caption.push_str(&number);
            caption.push_str("  ");
        }
        caption.push_str(&format!("\n{}-{} of {} for {}", start + 1, end, state.hits.len(), state.label));
        if failed > 0 { caption.push_str(&format!("\n{failed} no longer exist")); }
        caption.push_str(if end < state.hits.len() { "\n?more for the next, ?N to forward number N" } else { "\n?N to forward number N" });
        progress.show("Sending...".to_owned()).await;
        if let [(_, media)] = album.as_slice() {
            // Telegram has no one-item album; a single result is an ordinary media message.
            service.client.send_message(home, InputMessage::new().text(&caption).fmt_entities(entities).copy_media(media)).await
                .map_err(|error| format!("Telegram rejected the result: {error}"))?;
        } else {
            let medias = album.iter().enumerate().map(|(index, (_, media))| {
                let item = if index == 0 { InputMedia::new().caption(&caption).fmt_entities(entities.clone()) } else { InputMedia::new().caption("") };
                item.copy_media(media)
            }).collect();
            service.client.send_album(home, medias).await.map_err(|error| format!("Telegram rejected the album: {error}"))?;
        }
    }
    for position in &separate {
        let (chat_id, message_id) = hits[*position];
        forward(service, home, chat_id, message_id).await.map_err(|error| format!("result {} could not be forwarded: {error}", start + position + 1))?;
        // Paced so a run of forwards does not draw a flood wait.
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    let total = state.hits.len();
    state.next = end;
    if !album.is_empty() { return Ok(String::new()); }
    let mut text = format!("{}-{} of {} for {}.", start + 1, end, total, state.label);
    if failed > 0 { text.push_str(&format!(" {failed} no longer exist.")); }
    if end < total { text.push_str(" Send ?more for the next ones."); }
    Ok(text)
}

/// Downloads the photos a search refers to: its own, its album's, and those of the message it replies to.
async fn references(service: &TelegramService, home: &Chat, message: &Message) -> Result<Vec<String>, String> {
    let mut sources = vec![message.clone()];
    if let Some(replied) = message.get_reply().await.map_err(|error| format!("cannot read the replied message: {error}"))? { sources.push(replied); }
    let mut photos: Vec<Message> = Vec::new();
    for source in sources {
        if let Some(group) = source.grouped_id() {
            // Album items arrive as separate messages with neighbouring IDs; give the rest a moment to land.
            tokio::time::sleep(Duration::from_millis(1500)).await;
            let ids: Vec<i32> = (source.id() - 9..=source.id() + 9).filter(|id| *id > 0).collect();
            let nearby = service.client.get_messages_by_id(home, &ids).await.map_err(|error| format!("cannot read the album: {error}"))?;
            photos.extend(nearby.into_iter().flatten().filter(|item| item.grouped_id() == Some(group) && item.photo().is_some()));
        } else if source.photo().is_some() {
            photos.push(source);
        }
    }
    photos.sort_by_key(Message::id);
    photos.dedup_by_key(|photo| photo.id());
    photos.truncate(8);
    let directory = crate::paths::cache_dir().join("views");
    tokio::fs::create_dir_all(&directory).await.map_err(|error| error.to_string())?;
    let mut paths = Vec::with_capacity(photos.len());
    for photo in photos {
        let path = directory.join(format!("query-{}.jpg", uuid::Uuid::new_v4()));
        let Some(media) = photo.media() else { continue };
        let mut file = tokio::fs::File::create(&path).await.map_err(|error| error.to_string())?;
        let result = service.downloads.save(&Source::Media(media), &mut file, u64::MAX, |_| {}, |_| async {}).await;
        drop(file);
        if let Err(error) = result {
            let _ = tokio::fs::remove_file(&path).await;
            return Err(format!("cannot download the photo: {error}"));
        }
        paths.push(path.to_string_lossy().into_owned());
    }
    Ok(paths)
}

/// Turns a sticker or image file into a Telegram photo: a JPEG rendition is uploaded and
/// registered with Telegram, because an album only accepts media Telegram already holds.
async fn as_photo(service: &TelegramService, home: &Chat, message: &Message) -> Result<Media, String> {
    let directory = crate::paths::cache_dir().join("views");
    tokio::fs::create_dir_all(&directory).await.map_err(|error| error.to_string())?;
    let source = directory.join(format!("album-{}", uuid::Uuid::new_v4()));
    let jpeg = source.with_extension("jpg");
    let photo = async {
        let media = message.media().ok_or("nothing to download")?;
        let mut file = tokio::fs::File::create(&source).await.map_err(|error| error.to_string())?;
        service.downloads.save(&Source::Media(media), &mut file, u64::MAX, |_| {}, |_| async {})
            .await.map_err(|error| format!("download failed: {error}"))?;
        drop(file);
        let (from, to) = (source.clone(), jpeg.clone());
        tokio::task::spawn_blocking(move || flatten(&from, &to)).await.map_err(|error| error.to_string())?.map_err(|error| format!("cannot decode: {error}"))?;
        let file = service.client.upload_file(&jpeg).await.map_err(|error| format!("upload failed: {error}"))?;
        let photo = tl::types::InputMediaUploadedPhoto { spoiler: false, live_photo: false, file: file.raw, stickers: None, ttl_seconds: None, video: None };
        let registered = service.client.upload_media(home, tl::enums::InputMedia::UploadedPhoto(photo)).await
            .map_err(|error| format!("Telegram rejected the photo: {error}"))?;
        registered.filter(|media| matches!(media, Media::Photo(_))).ok_or_else(|| "Telegram did not return a photo".to_owned())
    }.await;
    let _ = tokio::fs::remove_file(&source).await;
    let _ = tokio::fs::remove_file(&jpeg).await;
    photo
}

/// Decodes an image, bounded against decompression bombs, and writes it as a JPEG of at
/// most 1280 px with transparency flattened onto white (JPEG has none; black would hide
/// dark stickers).
fn flatten(source: &Path, destination: &Path) -> Result<(), String> {
    let mut reader = image::ImageReader::open(source).map_err(|error| error.to_string())?.with_guessed_format().map_err(|error| error.to_string())?;
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(16384);
    limits.max_image_height = Some(16384);
    limits.max_alloc = Some(256 * 1024 * 1024);
    reader.limits(limits);
    let rgba = reader.decode().map_err(|error| error.to_string())?.thumbnail(1280, 1280).to_rgba8();
    let mut flat = image::RgbImage::new(rgba.width(), rgba.height());
    for (x, y, pixel) in rgba.enumerate_pixels() {
        let alpha = u32::from(pixel[3]);
        let blend = |channel: u8| ((u32::from(channel) * alpha + 255 * (255 - alpha)) / 255) as u8;
        flat.put_pixel(x, y, image::Rgb([blend(pixel[0]), blend(pixel[1]), blend(pixel[2])]));
    }
    let mut encoded = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut encoded, 90).encode_image(&flat).map_err(|error| error.to_string())?;
    std::fs::write(destination, encoded).map_err(|error| error.to_string())
}
