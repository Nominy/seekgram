//! Size cap for locally cached Telegram media. Everything under the cache folder can be
//! fetched again, so the least recently used files are deleted once the cap is exceeded.
//! The index, drafts and login are not cache and are never touched.

use std::{fs, path::{Path, PathBuf}, sync::atomic::{AtomicU64, Ordering}, time::{Duration, SystemTime}};

const LIMIT: u64 = 4 * 1024 * 1024 * 1024;
/// Evicting below the cap lets one scan buy headroom instead of rescanning on every export.
const TARGET: u64 = LIMIT / 10 * 9;
/// A file this young may still be read by the request or worker that created it.
const GRACE: Duration = Duration::from_secs(600);
/// Cache size as of the last scan plus bytes this process wrote since; `u64::MAX` until the first scan.
static ESTIMATE: AtomicU64 = AtomicU64::new(u64::MAX);

type Entry = (SystemTime, u64, PathBuf);

fn collect(dir: &Path, out: &mut Vec<Entry>) {
    let Ok(entries) = fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let Ok(meta) = entry.metadata() else { continue };
        if meta.is_dir() { collect(&entry.path(), out); }
        else if meta.is_file() { out.push((meta.modified().unwrap_or(SystemTime::UNIX_EPOCH), meta.len(), entry.path())); }
    }
}

/// Deletes least recently used files until `files` fits `target`, but only when
/// it exceeds `limit`. Returns the remaining size, files deleted and bytes freed.
fn evict(mut files: Vec<Entry>, limit: u64, target: u64, now: SystemTime) -> (u64, u64, u64) {
    let mut total: u64 = files.iter().map(|file| file.1).sum();
    let (mut evicted, mut freed) = (0u64, 0u64);
    if total > limit {
        files.sort_unstable_by_key(|file| file.0);
        for (modified, size, path) in &files {
            // Sorted oldest first: once one file is inside the grace period, all later ones are too.
            if total <= target || now.duration_since(*modified).map_or(true, |age| age < GRACE) { break; }
            if fs::remove_file(path).is_ok() { total -= size; evicted += 1; freed += size; }
        }
    }
    (total, evicted, freed)
}

/// Scans the cache and applies the cap. Recency is the modification time:
/// `touch` refreshes it on every reuse, because Windows does not reliably
/// maintain access times.
pub fn enforce() {
    let mut files = Vec::new();
    collect(&crate::paths::cache_dir(), &mut files);
    let (total, evicted, freed) = evict(files, LIMIT, TARGET, SystemTime::now());
    if evicted > 0 { eprintln!("Media cache: evicted {evicted} least recently used files ({freed} bytes); {total} bytes remain of a {LIMIT}-byte cap"); }
    ESTIMATE.store(total, Ordering::Relaxed);
}

/// Accounts for `added` bytes written by this process, scanning only when the
/// estimate is unknown or crosses the cap. Other processes' writes are picked
/// up by their own `enforce` call and by the next scan here.
pub fn grew(added: u64) {
    let estimate = ESTIMATE.load(Ordering::Relaxed);
    if estimate == u64::MAX || estimate.saturating_add(added) > LIMIT { enforce(); }
    else { ESTIMATE.fetch_add(added, Ordering::Relaxed); }
}

/// Marks a cached file as just used so it is evicted last.
pub fn touch(path: &Path) {
    if let Ok(file) = fs::OpenOptions::new().write(true).open(path) { let _ = file.set_modified(SystemTime::now()); }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Creates `name` with `size` bytes last used `age` seconds before `now`.
    fn file(dir: &Path, name: &str, size: usize, now: SystemTime, age: u64) -> PathBuf {
        let path = dir.join(name);
        fs::write(&path, vec![0u8; size]).unwrap();
        fs::OpenOptions::new().write(true).open(&path).unwrap().set_modified(now - Duration::from_secs(age)).unwrap();
        path
    }

    #[test]
    fn evicts_least_recently_used_down_to_target_and_spares_recent_files() {
        let dir = std::env::temp_dir().join(format!("telegram-cache-test-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(dir.join("nested")).unwrap();
        let now = SystemTime::now();
        let oldest = file(&dir, "oldest", 100, now, 5000);
        let old = file(&dir.join("nested"), "old", 100, now, 4000);
        let reused = file(&dir, "reused", 100, now, 3000);
        let fresh = file(&dir, "fresh", 100, now, 5);
        touch(&reused);

        let mut files = Vec::new();
        collect(&dir, &mut files);
        assert_eq!(evict(files.clone(), 400, 250, now), (400, 0, 0), "a cache at its limit is left alone");
        // Over the limit: the two oldest go, which reaches the target, so nothing newer is deleted.
        assert_eq!(evict(files, 350, 250, now), (200, 2, 200));
        assert!(!oldest.exists() && !old.exists() && reused.exists() && fresh.exists());

        // Both survivors are now inside the grace period, so even an impossible target deletes nothing.
        let mut files = Vec::new();
        collect(&dir, &mut files);
        assert_eq!(evict(files, 0, 0, SystemTime::now()), (200, 0, 0));
        fs::remove_dir_all(dir).unwrap();
    }
}
