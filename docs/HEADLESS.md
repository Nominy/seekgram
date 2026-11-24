# Seekgram on Linux, without the desktop app

The `seekgram` package builds a headless service on Linux. It uses the same Telegram client, local drafts and Streamable HTTP MCP protocol as the Windows app. It does not start the desktop app, updater, indexer, model workers or a Telegram search-chat responder. Nothing is migrated from your Windows installation. The Windows app, its settings, and existing MCP tool schemas are unchanged.

## Build and run privately

```sh
cargo build --release --locked -p seekgram
target/release/seekgram service
```

Use the same user and environment in a second terminal for `target/release/seekgram status`, `login`, or `stop`. `start` also runs the service detached; use `service` in the foreground to see startup errors. The service reserves TCP port 28741 on `127.0.0.1` by default, even while MCP is disabled. If needed, set a free `port` in `settings.json` before a restart.

The data directory is `$XDG_DATA_HOME/seekgram` or `~/.local/share/seekgram`. Set `SEEKGRAM_DATA_DIR` to an absolute path for a separate instance. On Linux it must be owned by your user and have mode `0700`; files must have mode `0600`. Do not point it at an existing desktop installation. Symlinks and unsafe permissions are rejected. Unix socket paths are limited by the OS; use a shorter absolute data directory if the socket path is too long.

**Use a trusted host and encrypted storage.** On Linux `secrets.bin` is an owner-readable file, not encrypted by Seekgram. It contains the API key and MCP bearer token. `telegram.sqlite3` contains the reusable login session, and drafts are kept in a separate database. Never upload these files, put them under source control, or send them to an assistant. Start a separate login using your own [Telegram API credentials](https://core.telegram.org/api/obtaining_api_id). Run `seekgram login` yourself and enter the API key, phone number, code, and two-step password as prompted in your own terminal. There is no QR login in this build. To revoke the session, use **Settings → Devices** in an official Telegram client.

## Use MCP

MCP remains off by default. To enable it, set `"mcp": true` in your private `settings.json`; an otherwise empty settings file can contain just `{"mcp":true}`. Restart the headless service after editing the file. MCP listens only on `http://127.0.0.1:28741/mcp` (substitute your configured port). Your trusted local MCP client must send the token from `secrets.bin` in the `Authorization: Bearer` header. Have the client read it from a private file or secret store, not from a command argument or chat. This token permits MCP reads and `draft_send`; only give it to trusted clients. Sending still requires an explicit instruction to send the particular draft. TCP does not expose login or other owner controls on Linux; those use `control.sock` in the private data directory.

The host and any private connection to it must be configured separately. Do not expose this port directly to the internet. Obtain any permissions and consent required by [Telegram's API and data-use terms](https://core.telegram.org/api/terms) before relying on AI tools with chats or media.

## Query an existing media index

The `IndexQuery` interface separates MCP from the index backend. Windows retains its existing text/image embedding and ranking flow. The headless adapter can open a compatible index at `index/index.sqlite3` in the private data directory, read-only. Only use a SQLite-consistent snapshot you are authorized to use, owned by the same Telegram account. The adapter validates model metadata and the account binding; no media index is created automatically.

`media_status`, `media_occurrences`, and `media_near_duplicates` query existing data without a model. New Linux-only `media_search_vector` accepts a precomputed 768-value query vector from the same EmbeddingGemma 2 revision and query preprocessing as the index. It normalizes valid input and uses the same ranking options and response shape as `media_search`. The headless host cannot embed a new text or image query, and the existing `media_search` returns a clear backend-unavailable error there.

## Verify without a Telegram login

```sh
cargo test --locked --profile ci -p seekgram -p seekgram-core -p seekgram-service
cargo build --locked --profile ci -p seekgram
python3 tests/headless_mcp.py target/ci/seekgram
```

The smoke test uses only an isolated data directory, a generated test token and local sockets. It does not connect to Telegram or download models.
