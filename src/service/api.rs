//! The loopback control API the panel and CLI use, and the mount point of `/mcp`.
//!
//! Everything here is reachable only from this machine and only with the access token,
//! which is stored encrypted for the Windows user. Browser pages cannot reach it either:
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

use super::{chat_search, mcp, provision, state::{LoginCommand, LoginRequest, Shared, Status}, workers::Workers};

const MAX_BODY: usize = 1024 * 1024;

#[derive(Clone)]
pub struct AppState {
    pub shared: Arc<Shared>,
    pub workers: Arc<Workers>,
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
    let Some(supplied) = mcp::single_header(headers, "authorization").and_then(|value| value.strip_prefix("Bearer ")) else {
        return StatusCode::UNAUTHORIZED.into_response();
    };
    let supplied: [u8; 32] = Sha256::digest(supplied.as_bytes()).into();
    if !bool::from(state.token_hash.ct_eq(&supplied)) {
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
        if key == "search_chat" || merged.get(key).is_none() { return Err(refused(format!("Unknown setting: {key}"))); }
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

async fn search_chat_create(State(state): State<AppState>) -> Reply {
    let telegram = state.shared.telegram().await.map_err(refused)?;
    chat_search::create_group(&state.shared, &telegram).await.map_err(failed)?;
    Ok(Json(state.shared.status()))
}

async fn search_chat_saved(State(state): State<AppState>) -> Reply {
    state.shared.set_settings(|settings| settings.search_chat = None).map_err(failed)?;
    Ok(Json(state.shared.status()))
}

async fn setup_repair(State(state): State<AppState>) -> Json<Status> {
    provision::repair();
    Json(state.shared.status())
}

async fn shutdown(State(state): State<AppState>) -> Json<Status> {
    state.shared.shutdown.send_replace(true);
    Json(state.shared.status())
}

/// Serves until shutdown, then lets requests that outlived their connection finish.
pub async fn serve(shared: Arc<Shared>, workers: Arc<Workers>) -> Result<(), String> {
    let port = shared.settings().port;
    let token = shared.secrets.lock().unwrap_or_else(|e| e.into_inner()).token.clone();
    let listener = tokio::net::TcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, port))).await
        .map_err(|error| format!("Port {port} on this PC is not available ({error}). Another program may be using it."))?;
    let in_flight = Arc::new(tokio::sync::RwLock::new(()));
    let state = AppState { shared: shared.clone(), workers, token_hash: Sha256::digest(token.as_bytes()).into(), port, in_flight: in_flight.clone() };
    let app = Router::new()
        .route("/api/status", get(status))
        .route("/api/login/api_key", post(login_api_key))
        .route("/api/login/phone", post(login_phone))
        .route("/api/login/code", post(login_code))
        .route("/api/login/password", post(login_password))
        .route("/api/logout", post(logout))
        .route("/api/settings", post(settings))
        .route("/api/search_chat/create", post(search_chat_create))
        .route("/api/search_chat/saved", post(search_chat_saved))
        .route("/api/setup/repair", post(setup_repair))
        .route("/api/shutdown", post(shutdown))
        .route("/mcp", any(mcp::endpoint))
        .layer(middleware::from_fn_with_state(state.clone(), guard))
        .layer(DefaultBodyLimit::max(MAX_BODY))
        .with_state(state);
    eprintln!("Control API listening on loopback port {port}");
    let stopping = shared.clone();
    let result = axum::serve(listener, app).with_graceful_shutdown(async move { stopping.stopped().await }).await
        .map_err(|error| format!("The control API stopped unexpectedly: {error}"));
    // A send detached from a closed connection is allowed to finish, within reason.
    let _ = tokio::time::timeout(Duration::from_secs(30), in_flight.write()).await;
    result
}
