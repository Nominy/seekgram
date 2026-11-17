//! The Telegram login on disk: grammers' session (datacenter keys, known peers and update
//! state) kept in SQLite through rusqlite, the SQLite build the rest of the service links.
//! Every change is written through; reads come from memory.

use std::{collections::HashMap, fmt, net::{SocketAddrV4, SocketAddrV6}, path::Path, sync::{Mutex, MutexGuard}};

use grammers_session::{
    BoxFuture, Session, SessionData,
    types::{ChannelKind, ChannelState, DcOption, PeerAuth, PeerId, PeerInfo, PeerKind, UpdateState, UpdatesState},
};
use rusqlite::{Connection, OptionalExtension, params};

const SCHEMA: &str = "
    PRAGMA journal_mode=WAL;
    PRAGMA synchronous=NORMAL;
    CREATE TABLE IF NOT EXISTS home(dc_id INTEGER NOT NULL);
    CREATE TABLE IF NOT EXISTS dc_option(dc_id INTEGER PRIMARY KEY, ipv4 TEXT NOT NULL, ipv6 TEXT NOT NULL, auth_key BLOB);
    CREATE TABLE IF NOT EXISTS peer(peer_id INTEGER PRIMARY KEY, auth INTEGER, bot INTEGER, is_self INTEGER, channel_kind INTEGER);
    CREATE TABLE IF NOT EXISTS update_state(pts INTEGER NOT NULL, qts INTEGER NOT NULL, date INTEGER NOT NULL, seq INTEGER NOT NULL);
    CREATE TABLE IF NOT EXISTS channel_state(channel_id INTEGER PRIMARY KEY, pts INTEGER NOT NULL);
";

#[derive(Debug)]
pub enum StoreError {
    Sql(rusqlite::Error),
    Poisoned,
    Corrupt(&'static str),
}

impl fmt::Display for StoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sql(error) => write!(f, "{error}"),
            Self::Poisoned => f.write_str("the saved Telegram login is locked by a crashed task"),
            Self::Corrupt(what) => write!(f, "the saved Telegram login is damaged ({what})"),
        }
    }
}

impl std::error::Error for StoreError {}

impl From<rusqlite::Error> for StoreError {
    fn from(error: rusqlite::Error) -> Self { Self::Sql(error) }
}

pub struct SessionStore {
    data: Mutex<SessionData>,
    db: Mutex<Connection>,
}

impl SessionStore {
    /// Opens the session database, creating it when missing. A database that has no
    /// home datacenter yet takes the login of grammers 0.7's session file, if there is one,
    /// so updating Seekgram does not log the user out.
    pub fn open(path: &Path, legacy: &Path) -> Result<Self, StoreError> {
        let db = Connection::open(path)?;
        db.busy_timeout(std::time::Duration::from_secs(5))?;
        db.execute_batch(SCHEMA)?;
        let mut data = SessionData::default();
        let home: Option<i32> = db.query_row("SELECT dc_id FROM home LIMIT 1", [], |row| row.get(0)).optional()?;
        let store = match home {
            Some(home) => {
                data.home_dc = home;
                load(&db, &mut data)?;
                Self { data: Mutex::new(data), db: Mutex::new(db) }
            }
            None => {
                let store = Self { data: Mutex::new(data), db: Mutex::new(db) };
                if let Some(login) = std::fs::read(legacy).ok().and_then(|bytes| legacy_login(&bytes)) {
                    store.import(&login)?;
                    eprintln!("Account: moved the saved login to the new session format");
                }
                store
            }
        };
        Ok(store)
    }

    pub fn home_dc(&self) -> Result<i32, StoreError> {
        Ok(self.data()?.home_dc)
    }

    fn data(&self) -> Result<MutexGuard<'_, SessionData>, StoreError> {
        self.data.lock().map_err(|_| StoreError::Poisoned)
    }

    fn db(&self) -> Result<MutexGuard<'_, Connection>, StoreError> {
        self.db.lock().map_err(|_| StoreError::Poisoned)
    }

    fn import(&self, login: &LegacyLogin) -> Result<(), StoreError> {
        let mut data = self.data()?;
        let mut db = self.db()?;
        let transaction = db.transaction()?;
        for (dc_id, key) in &login.keys {
            let Some(option) = data.dc_options.get_mut(dc_id) else { continue };
            option.auth_key = Some(*key);
            write_dc(&transaction, option)?;
        }
        data.home_dc = login.home_dc;
        transaction.execute("DELETE FROM home", [])?;
        transaction.execute("INSERT INTO home(dc_id) VALUES (?1)", [login.home_dc])?;
        transaction.commit()?;
        Ok(())
    }

    fn put_peer(&self, db: &Connection, data: &mut SessionData, peer: PeerInfo) -> Result<(), StoreError> {
        let merged = match data.peer_infos.get_mut(&peer.id()) {
            Some(existing) => { existing.extend_info(&peer); existing.clone() }
            None => { data.peer_infos.insert(peer.id(), peer.clone()); peer }
        };
        let (auth, bot, is_self, kind) = match &merged {
            PeerInfo::User { auth, bot, is_self, .. } => (auth.map(PeerAuth::hash), *bot, *is_self, None),
            PeerInfo::Chat { .. } => (None, None, None, None),
            PeerInfo::Channel { auth, kind, .. } => (auth.map(PeerAuth::hash), None, None, kind.map(|kind| kind as i64)),
        };
        db.execute(
            "INSERT OR REPLACE INTO peer(peer_id, auth, bot, is_self, channel_kind) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![merged.id().bot_api_dialog_id_unchecked(), auth, bot, is_self, kind],
        )?;
        Ok(())
    }
}

fn load(db: &Connection, data: &mut SessionData) -> Result<(), StoreError> {
    let mut options = db.prepare("SELECT dc_id, ipv4, ipv6, auth_key FROM dc_option")?;
    let rows = options.query_map([], |row| Ok((row.get::<_, i32>(0)?, row.get::<_, String>(1)?, row.get::<_, String>(2)?, row.get::<_, Option<Vec<u8>>>(3)?)))?;
    for row in rows {
        let (id, ipv4, ipv6, key) = row?;
        let auth_key = match key {
            None => None,
            Some(key) => Some(<[u8; 256]>::try_from(key).map_err(|_| StoreError::Corrupt("authorization key length"))?),
        };
        data.dc_options.insert(id, DcOption {
            id,
            ipv4: ipv4.parse::<SocketAddrV4>().map_err(|_| StoreError::Corrupt("datacenter address"))?,
            ipv6: ipv6.parse::<SocketAddrV6>().map_err(|_| StoreError::Corrupt("datacenter address"))?,
            auth_key,
        });
    }
    let mut peers = db.prepare("SELECT peer_id, auth, bot, is_self, channel_kind FROM peer")?;
    let rows = peers.query_map([], |row| Ok((row.get::<_, i64>(0)?, row.get::<_, Option<i64>>(1)?, row.get::<_, Option<bool>>(2)?, row.get::<_, Option<bool>>(3)?, row.get::<_, Option<i64>>(4)?)))?;
    for row in rows {
        let (marked, auth, bot, is_self, kind) = row?;
        let id = PeerId::from_bot_api_dialog_id(marked).ok_or(StoreError::Corrupt("peer id"))?;
        let bare = id.bare_id_unchecked();
        let info = match id.kind() {
            PeerKind::User => PeerInfo::User { id: bare, auth: auth.map(PeerAuth::from_hash), bot, is_self },
            PeerKind::Chat => PeerInfo::Chat { id: bare },
            PeerKind::Channel => PeerInfo::Channel {
                id: bare,
                auth: auth.map(PeerAuth::from_hash),
                kind: match kind {
                    Some(1) => Some(ChannelKind::Broadcast),
                    Some(2) => Some(ChannelKind::Megagroup),
                    Some(3) => Some(ChannelKind::Gigagroup),
                    Some(4) => Some(ChannelKind::Community),
                    _ => None,
                },
            },
        };
        data.peer_infos.insert(id, info);
    }
    if let Some((pts, qts, date, seq)) = db.query_row("SELECT pts, qts, date, seq FROM update_state LIMIT 1", [], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))).optional()? {
        data.updates_state = UpdatesState { pts, qts, date, seq, channels: Vec::new() };
    }
    let mut channels = db.prepare("SELECT channel_id, pts FROM channel_state")?;
    for row in channels.query_map([], |row| Ok(ChannelState { id: row.get(0)?, pts: row.get(1)? }))? {
        data.updates_state.channels.push(row?);
    }
    Ok(())
}

fn write_dc(db: &Connection, option: &DcOption) -> rusqlite::Result<usize> {
    db.execute(
        "INSERT OR REPLACE INTO dc_option(dc_id, ipv4, ipv6, auth_key) VALUES (?1, ?2, ?3, ?4)",
        params![option.id, option.ipv4.to_string(), option.ipv6.to_string(), option.auth_key.as_ref().map(|key| key.as_slice())],
    )
}

fn write_updates(db: &Connection, state: &UpdatesState) -> rusqlite::Result<()> {
    db.execute("DELETE FROM update_state", [])?;
    db.execute("INSERT INTO update_state(pts, qts, date, seq) VALUES (?1, ?2, ?3, ?4)", params![state.pts, state.qts, state.date, state.seq])?;
    Ok(())
}

impl Session for SessionStore {
    type Error = StoreError;

    fn home_dc_id(&self) -> Result<i32, StoreError> {
        self.home_dc()
    }

    fn set_home_dc_id(&self, dc_id: i32) -> BoxFuture<'_, Result<(), StoreError>> {
        Box::pin(async move {
            self.data()?.home_dc = dc_id;
            let mut db = self.db()?;
            let transaction = db.transaction()?;
            transaction.execute("DELETE FROM home", [])?;
            transaction.execute("INSERT INTO home(dc_id) VALUES (?1)", [dc_id])?;
            transaction.commit()?;
            Ok(())
        })
    }

    fn dc_option(&self, dc_id: i32) -> Result<Option<DcOption>, StoreError> {
        Ok(self.data()?.dc_options.get(&dc_id).cloned())
    }

    fn set_dc_option(&self, dc_option: &DcOption) -> BoxFuture<'_, Result<(), StoreError>> {
        let dc_option = dc_option.clone();
        Box::pin(async move {
            let mut data = self.data()?;
            let db = self.db()?;
            write_dc(&db, &dc_option)?;
            data.dc_options.insert(dc_option.id, dc_option);
            Ok(())
        })
    }

    fn peer(&self, peer: PeerId) -> BoxFuture<'_, Result<Option<PeerInfo>, StoreError>> {
        Box::pin(async move {
            let data = self.data()?;
            Ok(if peer == PeerId::self_user() {
                data.peer_infos.values().find(|info| matches!(info, PeerInfo::User { is_self: Some(true), .. })).cloned()
            } else {
                data.peer_infos.get(&peer).cloned()
            })
        })
    }

    fn cache_peer(&self, peer: PeerInfo) -> BoxFuture<'_, Result<(), StoreError>> {
        Box::pin(async move {
            let mut data = self.data()?;
            let db = self.db()?;
            self.put_peer(&db, &mut data, peer)
        })
    }

    fn cache_peers(&self, peers: Vec<PeerInfo>) -> BoxFuture<'_, Result<(), StoreError>> {
        Box::pin(async move {
            let mut data = self.data()?;
            let mut db = self.db()?;
            let transaction = db.transaction()?;
            for peer in peers {
                self.put_peer(&transaction, &mut data, peer)?;
            }
            transaction.commit()?;
            Ok(())
        })
    }

    fn updates_state(&self) -> BoxFuture<'_, Result<UpdatesState, StoreError>> {
        Box::pin(async move { Ok(self.data()?.updates_state.clone()) })
    }

    fn set_update_state(&self, update: UpdateState) -> BoxFuture<'_, Result<(), StoreError>> {
        Box::pin(async move {
            let mut data = self.data()?;
            let mut db = self.db()?;
            let transaction = db.transaction()?;
            let state = &mut data.updates_state;
            match update {
                UpdateState::All(all) => {
                    *state = all;
                    write_updates(&transaction, state)?;
                    transaction.execute("DELETE FROM channel_state", [])?;
                    for channel in &state.channels {
                        transaction.execute("INSERT INTO channel_state(channel_id, pts) VALUES (?1, ?2)", params![channel.id, channel.pts])?;
                    }
                }
                UpdateState::Primary { pts, date, seq } => {
                    (state.pts, state.date, state.seq) = (pts, date, seq);
                    write_updates(&transaction, state)?;
                }
                UpdateState::Secondary { qts } => {
                    state.qts = qts;
                    write_updates(&transaction, state)?;
                }
                UpdateState::Channel { id, pts } => {
                    state.channels.retain(|channel| channel.id != id);
                    state.channels.push(ChannelState { id, pts });
                    transaction.execute("INSERT OR REPLACE INTO channel_state(channel_id, pts) VALUES (?1, ?2)", params![id, pts])?;
                }
            }
            transaction.commit()?;
            Ok(())
        })
    }
}

/// The parts of a grammers 0.7 session file needed to stay logged in.
#[derive(Debug, PartialEq)]
struct LegacyLogin {
    home_dc: i32,
    keys: HashMap<i32, [u8; 256]>,
}

/// Reads grammers 0.7's session file, the TL object
/// `session flags:# dcs:Vector<DataCenter> user:flags.0?User state:flags.1?UpdateState`
/// with `dataCenter flags:# id:int ipv4:flags.0?int ipv6:flags.1?int128 port:int auth:flags.2?bytes`
/// and `user id:long dc:int bot:Bool`. `None` when it holds no logged-in user.
fn legacy_login(bytes: &[u8]) -> Option<LegacyLogin> {
    struct Reader<'a>(&'a [u8]);
    impl Reader<'_> {
        fn take(&mut self, count: usize) -> Option<&[u8]> {
            if self.0.len() < count { return None; }
            let (head, tail) = self.0.split_at(count);
            self.0 = tail;
            Some(head)
        }
        fn u32(&mut self) -> Option<u32> { Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?)) }
        fn i32(&mut self) -> Option<i32> { Some(i32::from_le_bytes(self.take(4)?.try_into().ok()?)) }
        fn i64(&mut self) -> Option<i64> { Some(i64::from_le_bytes(self.take(8)?.try_into().ok()?)) }
        fn bytes(&mut self) -> Option<Vec<u8>> {
            let first = *self.take(1)?.first()?;
            let (length, header) = if first < 254 { (first as usize, 1) } else {
                let length = self.take(3)?;
                (length[0] as usize | (length[1] as usize) << 8 | (length[2] as usize) << 16, 4)
            };
            let data = self.take(length)?.to_vec();
            self.take((4 - (header + length) % 4) % 4)?;
            Some(data)
        }
    }
    let mut reader = Reader(bytes);
    reader.u32()?;
    let flags = reader.u32()?;
    if reader.u32()? != 0x1cb5_c415 { return None; }
    let count = reader.i32()?;
    let mut keys = HashMap::new();
    for _ in 0..count {
        reader.u32()?;
        let dc_flags = reader.u32()?;
        let id = reader.i32()?;
        if dc_flags & 1 != 0 { reader.i32()?; }
        if dc_flags & 2 != 0 { reader.take(16)?; }
        reader.i32()?;
        if dc_flags & 4 != 0 {
            let key = reader.bytes()?;
            keys.insert(id, <[u8; 256]>::try_from(key).ok()?);
        }
    }
    if flags & 1 == 0 { return None; }
    reader.u32()?;
    reader.i64()?;
    let home_dc = reader.i32()?;
    keys.contains_key(&home_dc).then_some(LegacyLogin { home_dc, keys })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes_field(out: &mut Vec<u8>, data: &[u8]) {
        out.extend([254, data.len() as u8, (data.len() >> 8) as u8, (data.len() >> 16) as u8]);
        out.extend(data);
        while out.len() % 4 != 0 { out.push(0); }
    }

    fn legacy_file(user: Option<i32>) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend(0x1111_1111u32.to_le_bytes());
        out.extend(u32::from(user.is_some()).to_le_bytes());
        out.extend(0x1cb5_c415u32.to_le_bytes());
        out.extend(2i32.to_le_bytes());
        for (id, fill, ipv6) in [(2i32, 7u8, false), (4, 9u8, true)] {
            out.extend(0x2222_2222u32.to_le_bytes());
            out.extend((1u32 | if ipv6 { 2 } else { 0 } | 4).to_le_bytes());
            out.extend(id.to_le_bytes());
            out.extend(0x0100_007fi32.to_le_bytes());
            if ipv6 { out.extend([0u8; 16]); }
            out.extend(443i32.to_le_bytes());
            bytes_field(&mut out, &[fill; 256]);
        }
        if let Some(dc) = user {
            out.extend(0x3333_3333u32.to_le_bytes());
            out.extend(42i64.to_le_bytes());
            out.extend(dc.to_le_bytes());
            out.extend(0x997275b5u32.to_le_bytes());
        }
        out
    }

    #[test]
    fn legacy_login_keeps_every_key_and_the_home_datacenter() {
        let login = legacy_login(&legacy_file(Some(4))).unwrap();
        assert_eq!(login.home_dc, 4);
        assert_eq!(login.keys[&2], [7; 256]);
        assert_eq!(login.keys[&4], [9; 256]);
        assert_eq!(legacy_login(&legacy_file(None)), None);
        assert_eq!(legacy_login(&legacy_file(Some(5))), None);
        let file = legacy_file(Some(2));
        assert_eq!(legacy_login(&file[..file.len() - 9]), None);
    }

    #[tokio::test]
    async fn imported_login_and_peers_survive_reopening() {
        let dir = std::env::temp_dir().join(format!("seekgram-session-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let (path, legacy) = (dir.join("session.sqlite3"), dir.join("telegram.session"));
        std::fs::write(&legacy, legacy_file(Some(4))).unwrap();
        {
            let store = SessionStore::open(&path, &legacy).unwrap();
            assert_eq!(store.home_dc_id().unwrap(), 4);
            store.cache_peers(vec![
                PeerInfo::User { id: 42, auth: Some(PeerAuth::from_hash(-5)), bot: Some(false), is_self: Some(true) },
                PeerInfo::Channel { id: 77, auth: Some(PeerAuth::from_hash(9)), kind: Some(ChannelKind::Megagroup) },
            ]).await.unwrap();
            store.set_update_state(UpdateState::Channel { id: 77, pts: 3 }).await.unwrap();
        }
        // The legacy file no longer matters once the database holds a login.
        std::fs::write(&legacy, legacy_file(Some(2))).unwrap();
        let store = SessionStore::open(&path, &legacy).unwrap();
        assert_eq!(store.home_dc_id().unwrap(), 4);
        assert_eq!(store.dc_option(4).unwrap().unwrap().auth_key, Some([9; 256]));
        assert_eq!(store.dc_option(2).unwrap().unwrap().auth_key, Some([7; 256]));
        let me = store.peer(PeerId::self_user()).await.unwrap().unwrap();
        assert_eq!(me.id(), PeerId::user(42).unwrap());
        let group = store.peer(PeerId::channel(77).unwrap()).await.unwrap().unwrap();
        assert_eq!(group, PeerInfo::Channel { id: 77, auth: Some(PeerAuth::from_hash(9)), kind: Some(ChannelKind::Megagroup) });
        assert_eq!(store.updates_state().await.unwrap().channels, vec![ChannelState { id: 77, pts: 3 }]);
        drop(store);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
