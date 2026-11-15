//! Typed client for the background service's control API.
//!
//! Every method blocks on loopback HTTP, so the panel only ever calls them from
//! gpui's background executor, never from the UI thread.

use std::time::Duration;

use reqwest::blocking::{RequestBuilder, Response};
use serde_json::{Value, json};

use crate::service::state::Status;

/// The status poll must fail fast so a stopped service is noticed within a few seconds.
const STATUS_TIMEOUT: Duration = Duration::from_secs(4);
/// Login steps wait for Telegram (the service gives up after 90 s) and creating the
/// search group takes several Telegram round trips.
const ACTION_TIMEOUT: Duration = Duration::from_secs(120);

/// Why a request produced no `Status`.
#[derive(Debug)]
pub enum ApiError {
    /// Nothing is listening on the port: the service is not running, or not yet.
    Unreachable,
    /// The service answered and said no. The text is a sentence for the user.
    Refused(String),
    /// The connection broke or timed out, or the answer was not understood.
    Broken(String),
}

impl ApiError {
    /// The sentence to show next to the control that caused the request.
    pub fn message(&self) -> String {
        match self {
            Self::Unreachable => "Seekgram is not running.".to_owned(),
            Self::Refused(message) | Self::Broken(message) => message.clone(),
        }
    }
}

pub type Reply = Result<Status, ApiError>;

#[derive(Clone)]
pub struct Client {
    http: reqwest::blocking::Client,
    base: String,
    token: String,
}

impl Client {
    pub fn new(port: u16, token: String) -> Result<Self, String> {
        // A system proxy must never see loopback traffic that carries the access token.
        let http = reqwest::blocking::Client::builder()
            .no_proxy()
            .connect_timeout(Duration::from_secs(2))
            .build()
            .map_err(|error| format!("The panel could not set up its connection to Seekgram: {error}"))?;
        Ok(Self { http, base: format!("http://127.0.0.1:{port}"), token })
    }

    /// The bearer token, for the "Copy access token" button.
    pub fn token(&self) -> &str {
        &self.token
    }

    pub fn status(&self) -> Reply {
        self.send(self.http.get(self.url("/api/status")).timeout(STATUS_TIMEOUT))
    }

    pub fn login_api_key(&self, api_id: i64, api_hash: &str) -> Reply {
        self.post("/api/login/api_key", Some(json!({"api_id": api_id, "api_hash": api_hash})))
    }

    pub fn login_phone(&self, phone: &str) -> Reply {
        self.post("/api/login/phone", Some(json!({"phone": phone})))
    }

    pub fn login_code(&self, code: &str) -> Reply {
        self.post("/api/login/code", Some(json!({"code": code})))
    }

    pub fn login_password(&self, password: &str) -> Reply {
        self.post("/api/login/password", Some(json!({"password": password})))
    }

    pub fn logout(&self) -> Reply {
        self.post("/api/logout", None)
    }

    pub fn set_indexing(&self, indexing: bool) -> Reply {
        self.post("/api/settings", Some(json!({"indexing": indexing})))
    }

    pub fn set_autostart(&self, autostart: bool) -> Reply {
        self.post("/api/settings", Some(json!({"autostart": autostart})))
    }

    pub fn set_check_updates(&self, check_updates: bool) -> Reply {
        self.post("/api/settings", Some(json!({"check_updates": check_updates})))
    }

    pub fn set_mcp(&self, mcp: bool) -> Reply {
        self.post("/api/settings", Some(json!({"mcp": mcp})))
    }

    pub fn set_workers(&self, workers: u8) -> Reply {
        self.post("/api/settings", Some(json!({"workers": workers})))
    }

    pub fn set_concurrency(&self, concurrency: u8) -> Reply {
        self.post("/api/settings", Some(json!({"concurrency": concurrency})))
    }

    pub fn create_search_group(&self) -> Reply {
        self.post("/api/search_chat/create", None)
    }

    pub fn use_saved_messages(&self) -> Reply {
        self.post("/api/search_chat/saved", None)
    }

    pub fn repair_setup(&self) -> Reply {
        self.post("/api/setup/repair", None)
    }

    pub fn shutdown(&self) -> Reply {
        self.post("/api/shutdown", None)
    }

    fn url(&self, path: &str) -> String {
        format!("{}{path}", self.base)
    }

    fn post(&self, path: &str, body: Option<Value>) -> Reply {
        let request = self.http.post(self.url(path)).timeout(ACTION_TIMEOUT);
        self.send(match body {
            Some(body) => request.json(&body),
            None => request,
        })
    }

    fn send(&self, request: RequestBuilder) -> Reply {
        let response = request.bearer_auth(&self.token).send().map_err(|error| {
            if error.is_connect() {
                ApiError::Unreachable
            } else if error.is_timeout() {
                ApiError::Broken("Seekgram took too long to answer. Try again.".to_owned())
            } else {
                ApiError::Broken("The connection to Seekgram broke. Try again.".to_owned())
            }
        })?;
        if response.status().is_success() {
            response
                .json::<Status>()
                .map_err(|_| ApiError::Broken("Seekgram sent an answer this panel does not understand. Both may need updating.".to_owned()))
        } else {
            Err(ApiError::Refused(refusal(response)))
        }
    }
}

/// The service's own sentence when it sent one; otherwise what the status code means
/// for someone who only sees the panel.
fn refusal(response: Response) -> String {
    let code = response.status();
    let stated = response
        .json::<Value>()
        .ok()
        .and_then(|body| body.get("error").and_then(Value::as_str).map(str::to_owned))
        .filter(|message| !message.is_empty());
    if let Some(message) = stated {
        return message;
    }
    match code.as_u16() {
        401 | 403 => "Seekgram did not accept this panel. Another copy with a different data folder may be using the same port.".to_owned(),
        _ => format!("Seekgram could not do that (error {}).", code.as_u16()),
    }
}
