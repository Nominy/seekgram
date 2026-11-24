//! The loopback control API the panel and CLI use, and the mount point of `/mcp`.
//!
//! Everything here is reachable only from this machine and only with the access token,
//! which is stored encrypted for the Windows user or in the private Linux data directory.
//! On Linux owner controls use a Unix socket; only the MCP route is on loopback TCP.
//! Browser pages cannot reach it either:
//! requests must name a loopback host and, if they carry an Origin, a loopback origin.

use std::{net::{Ipv4Addr, SocketAddr}, sync::Arc, time::Duration};

use axum::{
    extract::{DefaultBodyLimit, Request, State},
    http::{header, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{any, get, post},
    Json, Router,
};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;
use tokio::sync::oneshot;

use crate::config::Settings;

use super::{index_query::IndexQuery, mcp, state::{LoginCommand, LoginRequest, Shared, Status}};
#[cfg(windows)]
use super::{chat_search, provision};

const MAX_BODY: usize = 1024 * 1024;

#[derive(Clone)]
pub struct AppState {
    pub shared: Arc<Shared>,
    pub index: Arc<dyn IndexQuery>,
    token_hash: [u8; 32],
    port: u16,
    /// Held for reading by requests that outlive their connection; shutdown waits for it.
    pub in_flight: Arc<tokio::sync::RwLock<()>>,
}

/// A failure the panel shows to the user as is.
struct ApiError(StatusCode, String);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response { (self.0, Json(json!({"error": self.1}))).into_response() }
}

fn refused(message: impl Into<String>) -> ApiError { ApiError(StatusCode::BAD_REQUEST, message.into()) }
fn failed(message: impl Into<String>) -> ApiError { ApiError(StatusCode::INTERNAL_SERVER_ERROR, message.into()) }

type Reply = Result<Json<Status>, ApiError>;

async fn guard(State(state): State<AppState>, request: Request, next: Next) -> Response {
    let headers = request.headers();
    if !mcp::single_header(headers, "host").is_some_and(|host| mcp::loopback_authority(host, state.port)) {
        return StatusCode::FORBIDDEN.into_response();
    }
    if headers.contains_key(header::ORIGIN) && !mcp::single_header(headers, "origin").is_some_and(|origin| mcp::valid_origin(origin, state.port)) {
        return StatusCode::FORBIDDEN.into_response();
    }
    if !authorized(headers, &state.token_hash) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    next.run(request).await
}

fn authorized(headers: &axum::http::HeaderMap, hash: &[u8; 32]) -> bool {
    let Some(supplied) = mcp::single_header(headers, "authorization").and_then(|value| value.strip_prefix("Bearer ")) else { return false; };
    let supplied: [u8; 32] = Sha256::digest(supplied.as_bytes()).into();
    bool::from(hash.ct_eq(&supplied))
}

#[cfg(target_os = "linux")]
async fn guard_owner(State(state): State<AppState>, request: Request, next: Next) -> Response {
    // The private socket is the origin boundary. Do not couple its request Host
    // to the TCP port because a changed setting takes effect on next start.
    if request.headers().contains_key(header::ORIGIN) {
        return StatusCode::FORBIDDEN.into_response();
    }
    if !authorized(request.headers(), &state.token_hash) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    next.run(request).await
}

async fn status(State(state): State<AppState>) -> Json<Status> { Json(state.shared.status()) }

/// Hands a login step to the account task and waits until Telegram has answered it.
async fn login(state: &AppState, command: LoginCommand) -> Reply {
    let (done, answered) = oneshot::channel();
    state.shared.login.send(LoginRequest { command, done }).await.map_err(|_| failed("Seekgram is shutting down"))?;
    match tokio::time::timeout(Duration::from_secs(90), answered).await {
        Ok(_) => Ok(Json(state.shared.status())),
        Err(_) => Err(failed("Telegram did not answer in time. Check your internet connection and try again.")),
    }
}

#[derive(Deserialize)]
struct ApiKey { api_id: i64, api_hash: String }

async fn login_api_key(State(state): State<AppState>, Json(body): Json<ApiKey>) -> Reply {
    let api_id = i32::try_from(body.api_id).ok().filter(|id| *id > 0).ok_or_else(|| refused("The api_id is a number, usually 6 to 8 digits."))?;
    let api_hash = body.api_hash.trim().to_ascii_lowercase();
    if api_hash.len() != 32 || !api_hash.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(refused("The api_hash is 32 letters and digits. Copy it exactly from my.telegram.org."));
    }
    login(&state, LoginCommand::ApiKey { api_id, api_hash }).await
}

#[derive(Deserialize)]
struct Phone { phone: String }

async fn login_phone(State(state): State<AppState>, Json(body): Json<Phone>) -> Reply {
    let phone: String = body.phone.chars().filter(|c| c.is_ascii_digit() || *c == '+').collect();
    if phone.trim_start_matches('+').len() < 7 { return Err(refused("Enter the phone number with its country code, like +44 7700 900123.")); }
    login(&state, LoginCommand::Phone(phone)).await
}

#[derive(Deserialize)]
struct Code { code: String }

async fn login_code(State(state): State<AppState>, Json(body): Json<Code>) -> Reply {
    let code: String = body.code.chars().filter(char::is_ascii_digit).collect();
    if code.is_empty() { return Err(refused("Enter the code Telegram sent you.")); }
    login(&state, LoginCommand::Code(code)).await
}

#[derive(Deserialize)]
struct Password { password: String }

async fn login_password(State(state): State<AppState>, Json(body): Json<Password>) -> Reply {
    if body.password.is_empty() { return Err(refused("Enter your two-step verification password.")); }
    login(&state, LoginCommand::Password(body.password)).await
}

async fn logout(State(state): State<AppState>) -> Reply { login(&state, LoginCommand::Logout).await }

/// Applies any subset of the settings. Unknown fields are refused so a typo cannot pass silently.
async fn settings(State(state): State<AppState>, Json(body): Json<Value>) -> Reply {
    let changes = body.as_object().ok_or_else(|| refused("Settings must be an object."))?;
    let mut merged = serde_json::to_value(state.shared.settings()).map_err(|error| failed(error.to_string()))?;
    for (key, value) in changes {
        // The search chat is only ever set by Seekgram itself, after checking the chat is private.
        if matches!(key.as_str(), "search_chat" | "search_group") || merged.get(key).is_none() { return Err(refused(format!("Unknown setting: {key}"))); }
        merged[key] = value.clone();
    }
    let new: Settings = serde_json::from_value(merged).map_err(|_| refused("A setting has the wrong kind of value."))?;
    if new.port < 1024 { return Err(refused("The port must be 1024 or higher.")); }
    if new.workers > 4 { return Err(refused("At most 4 model workers.")); }
    if !(1..=16).contains(&new.concurrency) { return Err(refused("Concurrency must be between 1 and 16.")); }
    if new.autostart != state.shared.settings().autostart {
        crate::autostart::set(new.autostart).map_err(failed)?;
    }
    state.shared.set_settings(|settings| *settings = new).map_err(failed)?;
    Ok(Json(state.shared.status()))
}

#[cfg(windows)]
async fn search_chat_create(State(state): State<AppState>) -> Reply {
    let telegram = state.shared.telegram().await.map_err(refused)?;
    chat_search::use_group(&state.shared, &telegram).await.map_err(failed)?;
    Ok(Json(state.shared.status()))
}

#[cfg(windows)]
async fn search_chat_saved(State(state): State<AppState>) -> Reply {
    state.shared.set_settings(|settings| {
        if settings.search_chat.is_some() { settings.search_group = settings.search_chat; }
        settings.search_chat = None;
    }).map_err(failed)?;
    Ok(Json(state.shared.status()))
}

#[cfg(windows)]
async fn setup_repair(State(state): State<AppState>) -> Json<Status> {
    provision::repair();
    Json(state.shared.status())
}

async fn shutdown(State(state): State<AppState>) -> Json<Status> {
    state.shared.shutdown.send_replace(true);
    Json(state.shared.status())
}

fn controls() -> Router<AppState> {
    let control = Router::new()
        .route("/api/status", get(status))
        .route("/api/login/api_key", post(login_api_key))
        .route("/api/login/phone", post(login_phone))
        .route("/api/login/code", post(login_code))
        .route("/api/login/password", post(login_password))
        .route("/api/logout", post(logout))
        .route("/api/settings", post(settings));
    #[cfg(windows)]
    let control = control
        .route("/api/search_chat/create", post(search_chat_create))
        .route("/api/search_chat/saved", post(search_chat_saved))
        .route("/api/setup/repair", post(setup_repair));
    control.route("/api/shutdown", post(shutdown))
}

fn tcp_routes(state: AppState) -> Router {
    let routes = Router::new().route("/mcp", any(mcp::endpoint));
    #[cfg(not(target_os = "linux"))]
    let routes = controls().merge(routes);
    routes.layer(middleware::from_fn_with_state(state.clone(), guard))
        .layer(DefaultBodyLimit::max(MAX_BODY)).with_state(state)
}

#[cfg(target_os = "linux")]
fn owner_routes(state: AppState) -> Router {
    controls().layer(middleware::from_fn_with_state(state.clone(), guard_owner))
        .layer(DefaultBodyLimit::max(MAX_BODY)).with_state(state)
}

/// Serves until shutdown, then lets requests that outlived their connection finish.
pub async fn serve(shared: Arc<Shared>, index: Arc<dyn IndexQuery>) -> Result<(), String> {
    let port = shared.settings().port;
    let token = shared.secrets.lock().unwrap_or_else(|e| e.into_inner()).token.clone();
    let listener = tokio::net::TcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, port))).await
        .map_err(|error| format!("Port {port} on this PC is not available ({error}). Another program may be using it."))?;
    let in_flight = Arc::new(tokio::sync::RwLock::new(()));
    let state = AppState { shared: shared.clone(), index, token_hash: Sha256::digest(token.as_bytes()).into(), port, in_flight: in_flight.clone() };
    let app = tcp_routes(state.clone());
    #[cfg(target_os = "linux")]
    let local = bind_control_socket()?;
    eprintln!("{} listening on loopback port {port}", if cfg!(target_os = "linux") { "MCP" } else { "Control API" });
    let stopping = shared.clone();
    let serve_tcp = async move {
        axum::serve(listener, app).with_graceful_shutdown(async move { stopping.stopped().await }).await
            .map_err(|error| format!("The control API stopped unexpectedly: {error}"))
    };
    #[cfg(not(target_os = "linux"))]
    let result = serve_tcp.await;
    #[cfg(target_os = "linux")]
    let result = {
        let owner = owner_routes(state);
        let serve_local = async move {
            axum::serve(local, owner).with_graceful_shutdown(async move { shared.stopped().await }).await
                .map_err(|error| format!("The owner control socket stopped unexpectedly: {error}"))
        };
        let result = tokio::try_join!(serve_tcp, serve_local).map(|_| ());
        let _ = std::fs::remove_file(crate::paths::control_socket());
        result
    };
    // A send detached from a closed connection is allowed to finish, within reason.
    let _ = tokio::time::timeout(Duration::from_secs(30), in_flight.write()).await;
    result
}

/// Called while the service owns its data-directory lock. A previous process may
/// have left a socket after a crash; never remove an unrelated file or symlink.
#[cfg(target_os = "linux")]
fn bind_control_socket() -> Result<tokio::net::UnixListener, String> {
    use std::os::unix::fs::{FileTypeExt, PermissionsExt};

    let path = crate::paths::control_socket();
    match std::fs::symlink_metadata(&path) {
        Ok(meta) if meta.file_type().is_socket() => std::fs::remove_file(&path).map_err(|e| e.to_string())?,
        Ok(_) => return Err("The owner control socket path is not a socket".into()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {},
        Err(e) => return Err(e.to_string()),
    }
    let socket = tokio::net::UnixListener::bind(&path).map_err(|e| format!("Cannot bind owner control socket: {e}"))?;
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))
        .map_err(|e| format!("Cannot restrict owner control socket: {e}"))?;
    Ok(socket)
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;
    use axum::body::{Body, to_bytes};
    use seekgram_core::config::Secrets;
    use tower::ServiceExt;

    fn state(enabled: bool) -> AppState {
        const MOCK_TOKEN: &str = "fixture-test-only";
        let settings = Settings { mcp: enabled, ..Settings::default() };
        let port = settings.port;
        let (send, _) = tokio::sync::mpsc::channel(1);
        let shared = Shared::new(settings, Secrets { token: MOCK_TOKEN.into(), ..Secrets::default() }, send);
        AppState { shared, index: Arc::new(super::super::index_query::LocalIndex::headless()),
            token_hash: Sha256::digest(MOCK_TOKEN.as_bytes()).into(), port,
            in_flight: Arc::new(tokio::sync::RwLock::new(())) }
    }

    async fn request(router: Router, method: &str, path: &str, auth: bool, origin: Option<&str>, body: Value) -> (StatusCode, Value) {
        let builder = Request::builder().method(method).uri(path)
            .header("host", "127.0.0.1:28741").header("content-type", "application/json");
        let builder = if auth { builder.header("authorization", "Bearer fixture-test-only") } else { builder };
        let builder = if let Some(origin) = origin { builder.header("origin", origin) } else { builder };
        let response = router.oneshot(builder.body(Body::from(body.to_string())).unwrap()).await.unwrap();
        let status = response.status();
        let bytes = to_bytes(response.into_body(), MAX_BODY).await.unwrap();
        (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
    }

    #[tokio::test]
    async fn headless_mcp_auth_protocol_and_no_telegram_session() {
        let hello = json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25"}});
        let app = tcp_routes(state(true));
        assert_eq!(request(app.clone(), "POST", "/mcp", false, None, hello.clone()).await.0, StatusCode::UNAUTHORIZED);
        assert_eq!(request(app.clone(), "POST", "/mcp", true, Some("http://example.invalid"), hello.clone()).await.0, StatusCode::FORBIDDEN);
        assert_eq!(request(tcp_routes(state(false)), "POST", "/mcp", true, None, hello.clone()).await.0, StatusCode::NOT_FOUND);
        for version in ["2025-03-26", "2025-06-18", "2025-11-25"] {
            let mut init = hello.clone();
            init["params"]["protocolVersion"] = json!(version);
            let (status, reply) = request(app.clone(), "POST", "/mcp", true, None, init).await;
            assert_eq!(status, StatusCode::OK);
            assert_eq!(reply["result"]["protocolVersion"], version);
        }
        let (_, response) = request(app.clone(), "POST", "/mcp", true, None,
            json!({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"history","arguments":{"chat":"123"}}})).await;
        assert_eq!(response["result"]["isError"], true);
        assert!(response["result"]["content"][0]["text"].as_str().unwrap().contains("not logged in"));
        assert_eq!(request(app, "GET", "/api/status", true, None, Value::Null).await.0, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn owner_socket_routes_are_separate() {
        let app = owner_routes(state(true));
        assert_eq!(request(app.clone(), "GET", "/api/status", false, None, Value::Null).await.0, StatusCode::UNAUTHORIZED);
        assert_eq!(request(app.clone(), "GET", "/api/status", true, Some("http://127.0.0.1:28741"), Value::Null).await.0, StatusCode::FORBIDDEN);
        assert_eq!(request(app.clone(), "GET", "/api/status", true, None, Value::Null).await.0, StatusCode::OK);
        assert_eq!(request(app, "POST", "/mcp", true, None, Value::Null).await.0, StatusCode::NOT_FOUND);
    }
}
