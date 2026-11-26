//! Boundary between the Telegram MCP transport and the media index implementation.

use futures_util::future::BoxFuture;
use serde_json::Value;
use std::path::PathBuf;
#[cfg(windows)]
use std::sync::Arc;

pub trait IndexQuery: Send + Sync {
    fn query<'a>(&'a self, action: &'a str, args: Value) -> BoxFuture<'a, Result<Value, String>>;
}

pub struct LocalIndex {
    pub(crate) path: PathBuf,
    #[cfg(windows)]
    pub(crate) workers: Arc<super::workers::Workers>,
}

impl LocalIndex {
    #[cfg(windows)]
    pub fn with_workers(workers: Arc<super::workers::Workers>) -> Self { Self { workers, path: default_path() } }

    #[cfg(not(windows))]
    pub fn headless() -> Self { Self { path: default_path() } }
}

fn default_path() -> PathBuf { crate::paths::index_dir().join("index.sqlite3") }

impl IndexQuery for LocalIndex {
    fn query<'a>(&'a self, action: &'a str, args: Value) -> BoxFuture<'a, Result<Value, String>> {
        Box::pin(super::search::query(self, action, args))
    }
}

/// Keep the account boundary in the adapter entry point; it applies to every
/// backend and uses the identity from the same snapshot as the returned rows.
pub(crate) async fn for_account(index: &dyn IndexQuery, account: i64, action: &str, args: Value) -> Result<Value, String> {
    match index.query(action, args).await {
        Ok(value) if value.get("account_id").and_then(Value::as_i64) == Some(account) => Ok(value),
        Ok(_) => Err("media_index_account_mismatch_or_missing".to_owned()),
        Err(error) => Err(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    struct Fixture(Value);

    impl IndexQuery for Fixture {
        fn query<'a>(&'a self, _: &'a str, _: Value) -> BoxFuture<'a, Result<Value, String>> {
            Box::pin(async { Ok(self.0.clone()) })
        }
    }

    #[tokio::test]
    async fn adapter_cannot_return_another_accounts_index() {
        let good = json!({"account_id":123,"hits":[{"chat_id":456}]});
        assert_eq!(for_account(&Fixture(good.clone()), 123, "media_status", json!({})).await.unwrap(), good);
        for unverified in [json!({"account_id":999,"hits":[{"chat_id":456}]}), json!({"hits":[]})] {
            assert_eq!(for_account(&Fixture(unverified), 123, "media_status", json!({})).await.unwrap_err(),
                       "media_index_account_mismatch_or_missing");
        }
    }
}
