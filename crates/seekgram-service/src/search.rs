//! Search over the embedding index: ranking, duplicate grouping and index status.

use rusqlite::{Connection, OpenFlags, OptionalExtension};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{sync::Arc, time::{Duration, SystemTime, UNIX_EPOCH}};

use super::workers::{Embed, EmbedError, Kind, Workers};

const MODEL_REPO: &str = "google/embeddinggemma-2";
const MODEL_REVISION: &str = "f9972ac22e696a105c482202646979e33e4b5a1e";
const DIMENSIONS: usize = 768;
fn open_index() -> Result<Connection, String> {
    let path = crate::paths::index_dir().join("index.sqlite3");
    let db = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX)
        .map_err(|e| format!("photo_index_open_failed: {e}"))?;
    db.busy_timeout(Duration::from_secs(3)).map_err(|e| format!("photo_index_error: {e}"))?;
    Ok(db)
}
fn validate_model(db: &Connection) -> Result<(), String> {
    for (key, expected) in [
        ("model", "google/embeddinggemma-2@f9972ac22e696a105c482202646979e33e4b5a1e|omni-v1|query=task: search result | query: |media=no-prefix|audio=16khz-mono|video=1fps|dim=768"),
        ("model_repo", MODEL_REPO),
        ("model_revision", MODEL_REVISION),
        ("processing_version", "omni-v1"),
        ("dimension", "768"),
        ("vector_format", "float32-le-normalized"),
    ] {
        let actual: Option<String> = db.query_row("SELECT value FROM metadata WHERE key=?1", [key], |r| r.get(0))
            .optional().map_err(|e| format!("photo_index_error: {e}"))?;
        if actual.as_deref() != Some(expected) {
            return Err(format!("photo_index_model_mismatch: metadata {key} is missing or incompatible"));
        }
    }
    Ok(())
}
fn index_account_id(db: &Connection) -> Result<i64, String> {
    let value: Option<String> = db.query_row("SELECT value FROM metadata WHERE key='account_id'", [], |r| r.get(0))
        .optional().map_err(|e| format!("photo_index_error: {e}"))?;
    value.and_then(|v| v.parse::<i64>().ok()).filter(|id| *id > 0)
        .ok_or_else(|| "photo_index_account_missing_or_invalid".to_string())
}
fn now() -> u64 { SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs() }
/// Embeds one search input. Searches jump ahead of queued indexing work.
async fn embedding(workers: &Arc<Workers>, kind: Kind, input: String) -> Result<Vec<f32>, String> {
    let workers = workers.clone();
    // A detached task completes the exchange even if the caller disconnects.
    tokio::spawn(async move {
        tokio::time::timeout(Duration::from_secs(240), workers.embed(Embed { kind, input, seconds: None }, true)).await
            .map_err(|_| "the model took too long to answer".to_owned())?
            .map_err(|error| match error {
                EmbedError::Media(reason) => format!("that input could not be read ({reason})"),
                EmbedError::Worker(reason) => reason,
            })
    }).await.map_err(|_| "the search was interrupted".to_owned())?
}

struct SearchHit {
    score: f32,
    chat: i64,
    message: i64,
    segment: i64,
    modality: String,
    start: Option<f64>,
    end: Option<f64>,
    // Only retained image candidates need their vector for exact comparison.
    vector: Option<Vec<u8>>,
    occurrence_count: u64,
    occurrences: Vec<(i64, i64)>,
    last_occurrence: Option<(i64, i64)>,
    /// Other exact groups folded into this one as near duplicates.
    merged: u64,
}

impl SearchHit {
    fn rank(&self) -> (f32, i64, i64, i64) {
        (self.score, self.chat, self.message, self.segment)
    }
}

fn index_error(e: impl std::fmt::Display) -> String { format!("photo_index_error: {e}") }

fn group_id(bytes: &[u8]) -> String { format!("{:x}", Sha256::digest(bytes)) }

fn parse_group_id(id: &str) -> Result<[u8; 32], String> {
    let bytes = id.as_bytes();
    if bytes.len() != 64 || !bytes.iter().all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(c)) {
        return Err("invalid_argument: group_id must be 64 lowercase hexadecimal characters".into());
    }
    let mut digest = [0; 32];
    for (out, hex) in digest.iter_mut().zip(bytes.chunks_exact(2)) {
        let nibble = |c: u8| if c <= b'9' { c - b'0' } else { c - b'a' + 10 };
        *out = (nibble(hex[0]) << 4) | nibble(hex[1]);
    }
    Ok(digest)
}

fn parse_cursor(value: Option<&Value>) -> Result<Option<(i64, i64)>, String> {
    let Some(value) = value else { return Ok(None); };
    let object = value.as_object().ok_or("invalid_argument: cursor must contain after_chat_id and after_message_id")?;
    if object.len() != 2 { return Err("invalid_argument: cursor must contain exactly after_chat_id and after_message_id".into()); }
    let chat = object.get("after_chat_id").and_then(Value::as_i64).ok_or("invalid_argument: cursor after_chat_id must be an integer")?;
    let message = object.get("after_message_id").and_then(Value::as_i64).ok_or("invalid_argument: cursor after_message_id must be an integer")?;
    if message <= 0 { return Err("invalid_argument: cursor after_message_id must be positive".into()); }
    Ok(Some((chat, message)))
}

fn search(vector: Vec<f32>, limit: usize, modality: Option<String>, grouped: bool, similar: Option<f32>) -> Result<Value, String> {
    search_db(&mut open_index()?, &vector, limit, modality.as_deref(), grouped, similar)
}

/// One search input: a text query or a canonical local image path.
fn parse_input(query: Option<&Value>, image_path: Option<&Value>) -> Result<(Kind, String), String> {
    match (query, image_path) {
        (Some(value), None) => {
            let query = value.as_str().ok_or("invalid_argument: query must be a string")?;
            if query.trim().is_empty() || query.len() > 4096 { return Err("invalid_argument: query must contain 1 to 4096 bytes".into()); }
            Ok((Kind::Text, query.to_owned()))
        }
        (None, Some(value)) => {
            let path = value.as_str().ok_or("invalid_argument: image_path must be a string")?;
            if path.is_empty() || path.len() > 32767 || !std::path::Path::new(path).is_absolute() {
                return Err("invalid_argument: image_path must be an absolute local file path".into());
            }
            let path = std::fs::canonicalize(path).map_err(|_| "invalid_argument: image file is unavailable")?;
            let metadata = std::fs::metadata(&path).map_err(|_| "invalid_argument: image file is unavailable")?;
            if !metadata.is_file() || metadata.len() == 0 || metadata.len() > 100 * 1024 * 1024 {
                return Err("invalid_argument: image file must be nonempty and at most 100 MiB".into());
            }
            Ok((Kind::Image, path.to_str().ok_or("invalid_argument: image path is not valid UTF-8")?.to_owned()))
        }
        _ => Err("invalid_argument: each input needs exactly one of query or image_path".into()),
    }
}

/// The query vector for weighted inputs: their weighted sum, renormalized, so
/// several pictures of one subject search for what they share and a negative
/// weight steers away from an input. `None` when the inputs cancel out.
fn combine(parts: &[(f32, Vec<f32>)]) -> Option<Vec<f32>> {
    let mut sum = vec![0f32; DIMENSIONS];
    for (weight, vector) in parts { for (total, value) in sum.iter_mut().zip(vector) { *total += weight * value; } }
    let norm = sum.iter().map(|v| v * v).sum::<f32>().sqrt();
    if !norm.is_finite() || norm < 1e-6 { return None; }
    for value in &mut sum { *value /= norm; }
    Some(sum)
}

/// `similar`: cosine at or above which an image result is folded into a better-ranked
/// one as a re-upload of the same picture. Each result is compared with the kept
/// results only, never with ones already folded, so similarity does not chain.
fn search_db(db: &mut Connection, vector: &[f32], limit: usize, modality: Option<&str>, grouped: bool, similar: Option<f32>) -> Result<Value, String> {
    let similar = similar.filter(|_| grouped);
    let wanted = limit;
    // Folding removes results, so rank a larger pool to still fill `wanted`.
    let limit = if similar.is_some() { (wanted * 4).min(400) } else { wanted };
    let tx = db.transaction().map_err(index_error)?;
    validate_model(&tx)?;
    let account_id = index_account_id(&tx)?;
    let mut best: Vec<SearchHit> = Vec::with_capacity(limit);
    let mut count = 0u64;
    {
        let mut statement = tx.prepare("SELECT chat_id, message_id, segment, modality, start_seconds, end_seconds, vector FROM items WHERE (?1 IS NULL OR modality = ?1)").map_err(index_error)?;
        let mut rows = statement.query([modality]).map_err(index_error)?;
        while let Some(row) = rows.next().map_err(index_error)? {
            let chat: i64 = row.get(0).map_err(index_error)?;
            let message: i64 = row.get(1).map_err(index_error)?;
            let segment: i64 = row.get(2).map_err(index_error)?;
            let bytes = row.get_ref(6).map_err(index_error)?.as_blob().map_err(index_error)?;
            if bytes.len() != DIMENSIONS * 4 { return Err("photo_index_vector_dimension_mismatch".into()); }
            let mut score = 0f32;
            for (chunk, q) in bytes.chunks_exact(4).zip(vector) {
                let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                if !value.is_finite() { return Err("photo_index_invalid_vector".into()); }
                score += value * q;
            }
            if !score.is_finite() { return Err("photo_index_invalid_score".into()); }
            count += 1;
            let rank = (score, chat, message, segment);
            // A lower-ranked duplicate cannot change its group's representative.
            if best.len() == limit && rank < best.last().expect("nonempty").rank() { continue; }
            let kind = row.get_ref(3).map_err(index_error)?.as_str().map_err(index_error)?;
            if grouped && kind == "image" {
                if let Some(existing) = best.iter().position(|hit| hit.vector.as_deref() == Some(bytes)) {
                    if rank <= best[existing].rank() { continue; }
                    best.remove(existing);
                }
            }
            let pos = best.partition_point(|hit| hit.rank() > rank);
            if pos < limit {
                let start: Option<f64> = row.get(4).map_err(index_error)?;
                let end: Option<f64> = row.get(5).map_err(index_error)?;
                if start.is_some_and(|v| !v.is_finite()) || end.is_some_and(|v| !v.is_finite()) {
                    return Err("photo_index_invalid_timestamp".into());
                }
                best.insert(pos, SearchHit {
                    score, chat, message, segment, modality: kind.to_owned(), start, end,
                    vector: if grouped && kind == "image" { Some(bytes.to_vec()) } else { None },
                    occurrence_count: 0, occurrences: Vec::new(), last_occurrence: None, merged: 0,
                });
                if best.len() > limit { best.pop(); }
            }
        }
    }
    // Probe the winning image prefixes in the same snapshot. A prefix is only a lookup
    // hint: two different vectors can share it, so compare the full bytes below.
    if grouped && best.iter().any(|hit| hit.vector.is_some()) {
        let mut prefixes: Vec<([u8; 8], usize)> = best.iter().enumerate().filter_map(|(index, hit)| {
            hit.vector.as_ref().map(|bytes| (bytes[..8].try_into().expect("validated vector"), index))
        }).collect();
        prefixes.sort_unstable();
        let mut sql = String::from("SELECT chat_id, message_id, vector FROM items WHERE modality='image' AND substr(vector,1,8) IN (");
        for i in 0..prefixes.len() {
            if i != 0 { sql.push(','); }
            sql.push('?');
        }
        // Only order within each group matters. This order is also the index order.
        sql.push_str(") ORDER BY substr(vector,1,8), chat_id, message_id");
        let mut statement = tx.prepare(&sql).map_err(index_error)?;
        let mut rows = statement.query(rusqlite::params_from_iter(prefixes.iter().map(|(prefix, _)| prefix.as_slice()))).map_err(index_error)?;
        while let Some(row) = rows.next().map_err(index_error)? {
            let pair = (row.get(0).map_err(index_error)?, row.get(1).map_err(index_error)?);
            let bytes = row.get_ref(2).map_err(index_error)?.as_blob().map_err(index_error)?;
            if bytes.len() != DIMENSIONS * 4 { return Err("photo_index_vector_dimension_mismatch".into()); }
            let prefix: [u8; 8] = bytes[..8].try_into().expect("validated vector");
            let mut position = prefixes.partition_point(|(candidate, _)| *candidate < prefix);
            while position < prefixes.len() && prefixes[position].0 == prefix {
                let hit = &mut best[prefixes[position].1];
                if hit.vector.as_deref() == Some(bytes) {
                    if hit.last_occurrence != Some(pair) {
                        hit.last_occurrence = Some(pair);
                        hit.occurrence_count += 1;
                        if hit.occurrences.len() < 5 { hit.occurrences.push(pair); }
                    }
                    break;
                }
                position += 1;
            }
        }
    }
    tx.commit().map_err(index_error)?;
    if let Some(threshold) = similar {
        let mut kept: Vec<SearchHit> = Vec::with_capacity(wanted);
        for hit in best {
            let closest = hit.vector.as_deref().and_then(|bytes| {
                kept.iter().enumerate().filter_map(|(index, other)| {
                    let cosine: f32 = other.vector.as_deref()?.chunks_exact(4).zip(bytes.chunks_exact(4))
                        .map(|(a, b)| f32::from_le_bytes([a[0], a[1], a[2], a[3]]) * f32::from_le_bytes([b[0], b[1], b[2], b[3]])).sum();
                    (cosine >= threshold).then_some((cosine, index))
                }).max_by(|a, b| a.0.total_cmp(&b.0))
            });
            match closest {
                Some((_, index)) => {
                    let target = &mut kept[index];
                    target.merged += 1 + hit.merged;
                    target.occurrence_count += hit.occurrence_count;
                    let room = 5usize.saturating_sub(target.occurrences.len());
                    target.occurrences.extend(hit.occurrences.into_iter().take(room));
                }
                None if kept.len() < wanted => kept.push(hit),
                None => {}
            }
        }
        best = kept;
    }
    let results: Vec<Value> = best.into_iter().map(|hit| {
        let mut result = json!({"score": hit.score, "chat_id": hit.chat, "message_id": hit.message, "segment": hit.segment, "modality": hit.modality, "start_seconds": hit.start, "end_seconds": hit.end});
        if let Some(bytes) = hit.vector {
            result["group_id"] = json!(group_id(&bytes));
            result["occurrence_count"] = json!(hit.occurrence_count);
            result["occurrences"] = json!(hit.occurrences.iter().map(|(chat_id, message_id)| json!({"chat_id": chat_id, "message_id": message_id})).collect::<Vec<_>>());
            result["occurrences_truncated"] = json!(hit.occurrence_count > hit.occurrences.len() as u64);
            if similar.is_some() { result["similar_merged"] = json!(hit.merged); }
        }
        result
    }).collect();
    Ok(json!({"account_id": account_id, "results": results, "vectors_scanned": count, "model_repo": MODEL_REPO, "model_revision": MODEL_REVISION, "processing_version": "omni-v1", "snapshot_completed_unix": now(), "freshness": "SQLite read snapshot; ongoing indexing and Telegram changes may not be included", "retrieval": "Use Telegram tools with chat_id and message_id to retrieve the original and nearby messages; media is not stored in this index", "similar_threshold": similar, "grouping": "Image groups have byte-identical normalized embeddings. With similar_threshold set, image results at or above that cosine to a better-ranked result are folded into it (similar_merged counts them; occurrence_count includes their messages); group_id and media_occurrences still cover the representative's exact group only. Neither proves identical files, and some re-uploads fall below the threshold."}))
}

fn occurrences_db(db: &mut Connection, id: &str, digest: &[u8; 32], limit: usize, after: Option<(i64, i64)>) -> Result<Value, String> {
    let tx = db.transaction().map_err(index_error)?;
    validate_model(&tx)?;
    let account_id = index_account_id(&tx)?;
    let mut occurrences = Vec::with_capacity(limit + 1);
    {
        let mut statement = tx.prepare("SELECT chat_id, message_id, vector FROM items WHERE modality='image' AND (?1 IS NULL OR chat_id > ?1 OR (chat_id = ?1 AND message_id > ?2)) ORDER BY chat_id, message_id").map_err(index_error)?;
        let mut rows = statement.query(rusqlite::params![after.map(|p| p.0), after.map(|p| p.1)]).map_err(index_error)?;
        let mut last = None;
        while let Some(row) = rows.next().map_err(index_error)? {
            let pair: (i64, i64) = (row.get(0).map_err(index_error)?, row.get(1).map_err(index_error)?);
            if last == Some(pair) { continue; }
            let bytes = row.get_ref(2).map_err(index_error)?.as_blob().map_err(index_error)?;
            if bytes.len() != DIMENSIONS * 4 { return Err("photo_index_vector_dimension_mismatch".into()); }
            if Sha256::digest(bytes).as_slice() == digest {
                last = Some(pair);
                occurrences.push(pair);
                if occurrences.len() > limit { break; }
            }
        }
    }
    tx.commit().map_err(index_error)?;
    let more = occurrences.len() > limit;
    if more { occurrences.pop(); }
    let next = if more { occurrences.last().map(|&(chat, message)| json!({"after_chat_id": chat, "after_message_id": message})) } else { None };
    Ok(json!({"account_id": account_id, "group_id": id, "occurrences": occurrences.iter().map(|(chat_id, message_id)| json!({"chat_id": chat_id, "message_id": message_id})).collect::<Vec<_>>(), "next_cursor": next, "cursor_semantics": "Exclusive (chat_id, message_id) in ascending order; each page uses a new SQLite read snapshot, so concurrent indexing can change later pages."}))
}

fn near_duplicates_db(db: &mut Connection, id: &str, digest: &[u8; 32], limit: usize, threshold: f32) -> Result<Value, String> {
    let tx = db.transaction().map_err(index_error)?;
    validate_model(&tx)?;
    let account_id = index_account_id(&tx)?;
    let reference = {
        // This is a full-vector digest, not a prefix lookup. Read in table order.
        let mut statement = tx.prepare("SELECT vector FROM items NOT INDEXED WHERE modality='image'").map_err(index_error)?;
        let mut rows = statement.query([]).map_err(index_error)?;
        let mut found = None;
        while let Some(row) = rows.next().map_err(index_error)? {
            let bytes = row.get_ref(0).map_err(index_error)?.as_blob().map_err(index_error)?;
            if bytes.len() != DIMENSIONS * 4 { return Err("photo_index_vector_dimension_mismatch".into()); }
            if Sha256::digest(bytes).as_slice() == digest {
                let mut vector = Vec::with_capacity(DIMENSIONS);
                for chunk in bytes.chunks_exact(4) {
                    let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    if !value.is_finite() { return Err("photo_index_invalid_vector".into()); }
                    vector.push(value);
                }
                found = Some(vector);
                break;
            }
        }
        found.ok_or("image_group_not_found")?
    };
    tx.commit().map_err(index_error)?;
    // Grouping remains exact; close groups are suggestions, not transitive merges.
    let mut ranked = search_db(db, &reference, limit + 2, Some("image"), true, None)?;
    if ranked["account_id"].as_i64() != Some(account_id) { return Err("photo_index_account_changed".into()); }
    let scanned = ranked["vectors_scanned"].as_u64().ok_or("photo_index_invalid_response")?;
    let results = ranked["results"].as_array_mut().ok_or("photo_index_invalid_response")?;
    results.retain(|hit| hit["group_id"].as_str() != Some(id) && hit["score"].as_f64().is_some_and(|score| score >= threshold as f64));
    let truncated = results.len() > limit;
    results.truncate(limit);
    Ok(json!({"account_id":account_id, "reference_group_id":id, "threshold":threshold,
              "candidates":results, "truncated":truncated, "vectors_scanned":scanned,
              "notice":"Similarity is a review suggestion, not proof of the same image; groups and messages are not merged.",
              "freshness":"Reference and candidate ranking use separate SQLite read snapshots; concurrent indexing may change results."}))
}
fn status() -> Result<Value, String> {
    let mut db = open_index()?;
    let tx = db.transaction().map_err(|e| format!("photo_index_error: {e}"))?;
    validate_model(&tx)?;
    let account_id = index_account_id(&tx)?;
    let mut counts = serde_json::Map::new();
    {
        let mut statement = tx.prepare("SELECT modality, COUNT(*) FROM items GROUP BY modality").map_err(|e| format!("photo_index_error: {e}"))?;
        let mut rows = statement.query([]).map_err(|e| format!("photo_index_error: {e}"))?;
        while let Some(row) = rows.next().map_err(|e| format!("photo_index_error: {e}"))? {
            let modality: String = row.get(0).map_err(|e| format!("photo_index_error: {e}"))?;
            let count: i64 = row.get(1).map_err(|e| format!("photo_index_error: {e}"))?;
            counts.insert(modality, json!(count));
        }
    }
    tx.commit().map_err(|e| format!("photo_index_error: {e}"))?;
    Ok(json!({"account_id": account_id, "items_by_modality": counts, "model_compatible": true, "model_repo": MODEL_REPO, "model_revision": MODEL_REVISION, "processing_version": "omni-v1", "snapshot_completed_unix": now(), "freshness": "SQLite read snapshot; ongoing indexing may not be included"}))
}

pub async fn dispatch(workers: &Arc<Workers>, action: &str, args: Value) -> Result<Value, String> {
    if !args.is_object() { return Err("invalid_argument: arguments must be an object".into()); }
    match action {
        "media_search" => {
            let inputs: Vec<(Kind, String, f32)> = match args.get("inputs") {
                None => { let (kind, input) = parse_input(args.get("query"), args.get("image_path"))?; vec![(kind, input, 1.0)] }
                Some(_) if args.get("query").is_some() || args.get("image_path").is_some() => return Err("invalid_argument: provide exactly one of query, image_path, or inputs".into()),
                Some(list) => {
                    let list = list.as_array().filter(|list| (1..=8).contains(&list.len())).ok_or("invalid_argument: inputs must hold 1 to 8 entries")?;
                    let mut inputs = Vec::with_capacity(list.len());
                    for entry in list {
                        let entry = entry.as_object().filter(|entry| entry.keys().all(|key| matches!(key.as_str(), "query" | "image_path" | "weight"))).ok_or("invalid_argument: each input is an object with query or image_path and an optional weight")?;
                        let weight = match entry.get("weight") {
                            None => 1.0,
                            Some(value) => value.as_f64().filter(|w| *w != 0.0 && (-10.0..=10.0).contains(w)).ok_or("invalid_argument: weight must be nonzero and between -10 and 10")? as f32,
                        };
                        let (kind, input) = parse_input(entry.get("query"), entry.get("image_path"))?;
                        inputs.push((kind, input, weight));
                    }
                    inputs
                }
            };
            let modality = match args.get("modality") {
                None => None,
                Some(Value::String(value)) if matches!(value.as_str(), "image" | "audio" | "video") => Some(value.clone()),
                _ => return Err("invalid_argument: modality must be image, audio, or video".into()),
            };
            let limit = match args.get("limit") { None => 10, Some(v) => v.as_u64().filter(|n| (1..=100).contains(n)).ok_or("invalid_argument: limit must be between 1 and 100")? as usize };
            let grouped = match args.get("group_duplicates") {
                None => true,
                Some(Value::Bool(value)) => *value,
                _ => return Err("invalid_argument: group_duplicates must be a boolean".into()),
            };
            let similar = match args.get("similar_threshold") {
                None => Some(0.99),
                Some(Value::Null) => None,
                Some(value) => Some(value.as_f64().filter(|v| (0.9..=1.0).contains(v)).ok_or("invalid_argument: similar_threshold must be null or between 0.9 and 1.0")? as f32),
            };
            // Avoid loading the model at all for an absent or incompatible index.
            tokio::task::spawn_blocking(|| { let db = open_index()?; validate_model(&db) }).await.map_err(|_| "photo_index_task_failed")??;
            let mut parts = Vec::with_capacity(inputs.len());
            for (kind, input, weight) in inputs { parts.push((weight, embedding(workers, kind, input).await?)); }
            let vector = combine(&parts).ok_or("invalid_argument: the weighted inputs cancel out")?;
            tokio::task::spawn_blocking(move || search(vector, limit, modality, grouped, similar)).await.map_err(|_| "photo_index_task_failed".to_string())?
        }
        "media_occurrences" => {
            let id = args.get("group_id").and_then(Value::as_str).ok_or("invalid_argument: group_id must be a string")?.to_owned();
            let digest = parse_group_id(&id)?;
            let limit = match args.get("limit") { None => 10, Some(v) => v.as_u64().filter(|n| (1..=100).contains(n)).ok_or("invalid_argument: limit must be between 1 and 100")? as usize };
            let after = parse_cursor(args.get("cursor"))?;
            tokio::task::spawn_blocking(move || occurrences_db(&mut open_index()?, &id, &digest, limit, after)).await.map_err(|_| "photo_index_task_failed".to_string())?
        }
        "media_near_duplicates" => {
            let id = args.get("group_id").and_then(Value::as_str).ok_or("invalid_argument: group_id must be a string")?.to_owned();
            let digest = parse_group_id(&id)?;
            let limit = match args.get("limit") { None => 10, Some(v) => v.as_u64().filter(|n| (1..=100).contains(n)).ok_or("invalid_argument: limit must be between 1 and 100")? as usize };
            let threshold = match args.get("threshold") {
                None => 0.99,
                Some(value) => value.as_f64().filter(|v| v.is_finite() && (0.9..=1.0).contains(v)).ok_or("invalid_argument: threshold must be between 0.9 and 1.0")?,
            };
            tokio::task::spawn_blocking(move || near_duplicates_db(&mut open_index()?, &id, &digest, limit, threshold as f32)).await.map_err(|_| "photo_index_task_failed".to_string())?
        }
        "media_status" => tokio::task::spawn_blocking(status).await.map_err(|_| "photo_index_task_failed".to_string())?,
        _ => Err(format!("unknown_media_action: {action}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Connection {
        let db = Connection::open_in_memory().unwrap();
        db.execute_batch("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE items(chat_id INTEGER, message_id INTEGER, segment INTEGER, modality TEXT,
                start_seconds REAL, end_seconds REAL, vector BLOB,
                PRIMARY KEY(chat_id, message_id, modality, segment));").unwrap();
        for (key, value) in [
            ("model", "google/embeddinggemma-2@f9972ac22e696a105c482202646979e33e4b5a1e|omni-v1|query=task: search result | query: |media=no-prefix|audio=16khz-mono|video=1fps|dim=768"),
            ("model_repo", MODEL_REPO), ("model_revision", MODEL_REVISION),
            ("processing_version", "omni-v1"), ("dimension", "768"),
            ("vector_format", "float32-le-normalized"), ("account_id", "123"),
        ] {
            db.execute("INSERT INTO metadata VALUES (?1, ?2)", [key, value]).unwrap();
        }
        db
    }

    fn vector(first: f32, second: f32) -> Vec<u8> {
        let mut bytes = vec![0; DIMENSIONS * 4];
        bytes[..4].copy_from_slice(&first.to_le_bytes());
        bytes[4..8].copy_from_slice(&second.to_le_bytes());
        bytes
    }

    fn insert(db: &Connection, chat: i64, message: i64, segment: i64, modality: &str, bytes: &[u8]) {
        db.execute("INSERT INTO items VALUES (?1, ?2, ?3, ?4, NULL, NULL, ?5)",
            rusqlite::params![chat, message, segment, modality, bytes]).unwrap();
    }

    #[test]
    fn groups_exact_vectors_and_counts_all_messages() {
        let mut db = fixture();
        let a = vector(1.0, 0.0);
        let b = vector(1.0, 1.0);
        for message in 1..=7 { insert(&db, -1, message, 0, "image", &a); }
        insert(&db, -1, 1, 1, "image", &a);
        insert(&db, 9, 1, 0, "image", &b);
        let mut query = vec![0.0; DIMENSIONS];
        query[0] = 1.0;
        let result = search_db(&mut db, &query, 2, Some("image"), true, None).unwrap();
        let hits = result["results"].as_array().unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0]["group_id"], group_id(&b));
        assert_eq!(hits[1]["group_id"], group_id(&a));
        assert_eq!(hits[1]["message_id"], 7);
        assert_eq!(hits[1]["occurrence_count"], 7);
        assert_eq!(hits[1]["occurrences"].as_array().unwrap().len(), 5);
        assert_eq!(hits[1]["occurrences"][0], json!({"chat_id": -1, "message_id": 1}));
        assert_eq!(hits[1]["occurrences_truncated"], true);
        let ungrouped = search_db(&mut db, &query, 3, Some("image"), false, None).unwrap();
        assert_eq!(ungrouped["results"].as_array().unwrap().len(), 3);
        assert!(ungrouped["results"][1].get("group_id").is_none());
    }

    #[test]
    fn distinct_vectors_with_identical_prefix_keep_separate_occurrences() {
        let mut db = fixture();
        let a = vector(1.0, 0.0);
        let mut b = a.clone();
        b[8..12].copy_from_slice(&1.0f32.to_le_bytes());
        insert(&db, 1, 1, 0, "image", &a);
        insert(&db, 1, 2, 0, "image", &b);
        let mut query = vec![0.0; DIMENSIONS];
        query[0] = 1.0;
        let result = search_db(&mut db, &query, 2, Some("image"), true, None).unwrap();
        let hits = result["results"].as_array().unwrap();
        assert_eq!(hits.len(), 2);
        assert!(hits.iter().all(|hit| hit["occurrence_count"] == 1));
        assert_ne!(hits[0]["group_id"], hits[1]["group_id"]);
    }

    #[test]
    fn close_embeddings_are_suggestions_not_merged_groups() {
        let mut db = fixture();
        let a = vector(1.0, 0.0);
        let b = vector(0.995, (1.0f32 - 0.995f32.powi(2)).sqrt());
        let c = vector(0.98, (1.0f32 - 0.98f32.powi(2)).sqrt());
        let d = vector(0.993, (1.0f32 - 0.993f32.powi(2)).sqrt());
        insert(&db, 1, 1, 0, "image", &a);
        insert(&db, 1, 2, 0, "image", &a);
        insert(&db, 2, 1, 0, "image", &b);
        insert(&db, 2, 2, 0, "image", &b);
        insert(&db, 3, 1, 0, "image", &c);
        insert(&db, 5, 1, 0, "image", &d);
        insert(&db, 4, 1, 0, "video", &b);
        let id = group_id(&a);
        let digest = parse_group_id(&id).unwrap();
        let result = near_duplicates_db(&mut db, &id, &digest, 10, 0.99).unwrap();
        let candidates = result["candidates"].as_array().unwrap();
        assert_eq!(candidates.len(), 2);
        assert_eq!(candidates[0]["group_id"], group_id(&b));
        assert_eq!(candidates[0]["occurrence_count"], 2);
        assert_ne!(candidates[0]["group_id"], id);
        assert!(!result["truncated"].as_bool().unwrap());
        let limited = near_duplicates_db(&mut db, &id, &digest, 1, 0.99).unwrap();
        assert_eq!(limited["candidates"].as_array().unwrap().len(), 1);
        assert!(limited["truncated"].as_bool().unwrap());
        let strict = near_duplicates_db(&mut db, &id, &digest, 10, 0.999).unwrap();
        assert!(strict["candidates"].as_array().unwrap().is_empty());
        assert!(near_duplicates_db(&mut db, &group_id(&vector(0.0, 1.0)), &parse_group_id(&group_id(&vector(0.0, 1.0))).unwrap(), 10, 0.99).is_err());
    }

    #[test]
    fn pagination_is_ordered_exclusive_and_deduplicates_segments() {
        let mut db = fixture();
        let a = vector(1.0, 0.0);
        for (chat, message) in [(2, 2), (-4, 3), (2, 1), (-4, 1), (8, 1)] {
            insert(&db, chat, message, 0, "image", &a);
        }
        insert(&db, -4, 1, 1, "image", &a);
        insert(&db, 1, 1, 0, "image", &vector(0.0, 1.0));
        let id = group_id(&a);
        let digest = parse_group_id(&id).unwrap();
        let first = occurrences_db(&mut db, &id, &digest, 2, None).unwrap();
        assert_eq!(first["account_id"], 123);
        assert_eq!(first["occurrences"], json!([{"chat_id":-4,"message_id":1},{"chat_id":-4,"message_id":3}]));
        let cursor = parse_cursor(first.get("next_cursor")).unwrap();
        let second = occurrences_db(&mut db, &id, &digest, 2, cursor).unwrap();
        assert_eq!(second["occurrences"], json!([{"chat_id":2,"message_id":1},{"chat_id":2,"message_id":2}]));
        let third = occurrences_db(&mut db, &id, &digest, 2, parse_cursor(second.get("next_cursor")).unwrap()).unwrap();
        assert_eq!(third["occurrences"], json!([{"chat_id":8,"message_id":1}]));
        assert!(third["next_cursor"].is_null());
        assert!(parse_group_id(&id.to_uppercase()).is_err());
        assert!(parse_group_id("123").is_err());
        assert!(parse_cursor(Some(&json!({"after_chat_id": 1}))).is_err());
        assert!(parse_cursor(Some(&json!({"after_chat_id": 1, "after_message_id": 2, "extra": 3}))).is_err());
        assert!(parse_cursor(Some(&json!({"after_chat_id": 1, "after_message_id": "2"}))).is_err());
        assert!(parse_cursor(Some(&json!({"after_chat_id": 1, "after_message_id": 0}))).is_err());
    }

    #[test]
    fn audio_and_video_remain_distinct_and_filters_work() {
        let mut db = fixture();
        let a = vector(1.0, 0.0);
        for (message, modality) in [(1, "image"), (2, "image"), (3, "audio"), (4, "audio"), (5, "video"), (6, "video")] {
            insert(&db, 1, message, 0, modality, &a);
        }
        let mut query = vec![0.0; DIMENSIONS];
        query[0] = 1.0;
        let all = search_db(&mut db, &query, 10, None, true, None).unwrap();
        let hits = all["results"].as_array().unwrap();
        assert_eq!(hits.len(), 5);
        assert_eq!(hits.iter().filter(|hit| hit.get("group_id").is_some()).count(), 1);
        let audio = search_db(&mut db, &query, 10, Some("audio"), true, None).unwrap();
        assert_eq!(audio["results"].as_array().unwrap().len(), 2);
        assert!(audio["results"][0].get("group_id").is_none());
    }

    #[test]
    fn similar_results_fold_into_the_best_ranked_copy_without_chaining() {
        let mut db = fixture();
        let a = vector(1.0, 0.0);
        // 0.995 to `a`: a re-upload. `c` is 0.997 to `b` but only 0.985 to `a`.
        let b = vector(0.995, (1.0f32 - 0.995f32.powi(2)).sqrt());
        let c = vector(10f32.to_radians().cos(), 10f32.to_radians().sin());
        insert(&db, 1, 1, 0, "image", &a);
        insert(&db, 1, 2, 0, "image", &a);
        insert(&db, 1, 3, 0, "image", &b);
        insert(&db, 1, 4, 0, "image", &c);
        insert(&db, 1, 5, 0, "video", &b);
        let mut query = vec![0.0; DIMENSIONS];
        query[0] = 1.0;
        let folded = search_db(&mut db, &query, 10, Some("image"), true, Some(0.99)).unwrap();
        let hits = folded["results"].as_array().unwrap();
        assert_eq!(hits.len(), 2, "c is close to the folded b only, so it stays");
        assert_eq!(hits[0]["group_id"], group_id(&a));
        assert_eq!(hits[0]["occurrence_count"], 3);
        assert_eq!(hits[0]["similar_merged"], 1);
        assert_eq!(hits[1]["group_id"], group_id(&c));
        assert_eq!(hits[1]["similar_merged"], 0);
        // Folding must not cost result slots: the pool is ranked beyond `limit`.
        let two = search_db(&mut db, &query, 2, Some("image"), true, Some(0.99)).unwrap();
        assert_eq!(two["results"].as_array().unwrap().len(), 2);
        let exact = search_db(&mut db, &query, 10, Some("image"), true, None).unwrap();
        assert_eq!(exact["results"].as_array().unwrap().len(), 3);
        // Video segments are never folded into images or each other.
        let all = search_db(&mut db, &query, 10, None, true, Some(0.99)).unwrap();
        assert_eq!(all["results"].as_array().unwrap().len(), 3);
    }

    #[test]
    fn combined_inputs_are_a_normalized_weighted_sum() {
        let unit = |index: usize| { let mut v = vec![0f32; DIMENSIONS]; v[index] = 1.0; v };
        let mixed = combine(&[(1.0, unit(0)), (1.0, unit(1))]).unwrap();
        assert!((mixed[0] - 0.5f32.sqrt()).abs() < 1e-6 && (mixed[1] - 0.5f32.sqrt()).abs() < 1e-6);
        let steered = combine(&[(2.0, unit(0)), (-1.0, unit(1))]).unwrap();
        assert!(steered[0] > 0.0 && steered[1] < 0.0);
        assert!((steered.iter().map(|v| v * v).sum::<f32>() - 1.0).abs() < 1e-5);
        assert_eq!(combine(&[(1.0, unit(0))]).unwrap(), unit(0));
        assert!(combine(&[(1.0, unit(0)), (-1.0, unit(0))]).is_none());
    }
}
