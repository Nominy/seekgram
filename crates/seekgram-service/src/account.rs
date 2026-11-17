//! The Telegram login: from "no API key" to a ready client, one user-supplied step at
//! a time. Owns `Status.account` and the lifetime of the logged-in client.

use std::sync::Arc;

use grammers_client::{InvocationError, SignInError, client::{LoginToken, PasswordToken}, peer::User, tl};
use tokio::{sync::mpsc, task::JoinHandle};

use crate::paths;

use super::{chat_search, state::{AccountStep, LoginCommand, LoginRequest, Shared}, telegram::{self, Connection, TelegramService, marked_id}, workers::Workers};

struct Account {
    shared: Arc<Shared>,
    workers: Arc<Workers>,
    /// Connected, though not necessarily logged in yet.
    connection: Option<Connection>,
    code: Option<LoginToken>,
    password: Option<PasswordToken>,
    /// The chat search listener of the logged-in client.
    listener: Option<JoinHandle<()>>,
}

/// Turns a Telegram error into a sentence the user can act on.
fn explain(error: &InvocationError) -> String {
    let rpc = match error {
        InvocationError::Rpc(rpc) => rpc,
        InvocationError::Deserialize(_) => return telegram::UNREADABLE.to_owned(),
        InvocationError::Session(error) => return format!("The saved Telegram login cannot be used: {error}"),
        _ => return "Could not reach Telegram. Check your internet connection and try again.".to_owned(),
    };
    match rpc.name.as_str() {
        "API_ID_INVALID" | "API_ID_PUBLISHED_FLOOD" => "Telegram does not accept this API key. Check the api_id and api_hash on my.telegram.org.".to_owned(),
        "PHONE_NUMBER_INVALID" => "Telegram does not recognise this phone number. Include the country code, like +44 7700 900123.".to_owned(),
        "PHONE_NUMBER_BANNED" => "Telegram has banned this phone number.".to_owned(),
        "PHONE_NUMBER_FLOOD" | "PHONE_PASSWORD_FLOOD" => "Too many attempts. Telegram asks you to try again later.".to_owned(),
        "PHONE_CODE_EXPIRED" => "That code has expired. Enter your number again to get a new one.".to_owned(),
        "FLOOD_WAIT" | "FLOOD_PREMIUM_WAIT" => match rpc.value {
            Some(seconds) if seconds >= 120 => format!("Too many attempts. Telegram asks you to wait {} minutes.", seconds.div_ceil(60)),
            Some(seconds) => format!("Too many attempts. Telegram asks you to wait {seconds} seconds."),
            None => "Too many attempts. Telegram asks you to wait a while.".to_owned(),
        },
        name => format!("Telegram refused the request ({name})."),
    }
}

impl Account {
    fn step(&self, step: AccountStep, error: Option<String>) {
        self.shared.update(|status| {
            status.account.step = step;
            status.account.error = error;
            if step != AccountStep::Password { status.account.password_hint = None; }
            if step != AccountStep::Ready { status.account.name = None; status.account.id = None; }
        });
        self.shared.notify();
    }

    fn key(&self) -> Option<(i32, String)> {
        let secrets = self.shared.secrets.lock().unwrap_or_else(|e| e.into_inner());
        Some((secrets.api_id?, secrets.api_hash.clone()?))
    }

    fn client(&self) -> Option<grammers_client::Client> {
        self.connection.as_ref().map(|connection| connection.client.clone())
    }

    /// Drops the client and its connections to Telegram.
    async fn disconnect(&mut self) {
        if let Some(connection) = self.connection.take() { connection.close().await; }
    }

    /// Connects with the stored key and moves to whichever step applies. Returns whether
    /// Telegram was reached; an unreachable Telegram is retried by the caller.
    async fn connect(&mut self) -> bool {
        let Some((api_id, _)) = self.key() else {
            self.step(AccountStep::ApiKey, None);
            return true;
        };
        self.step(AccountStep::Connecting, None);
        self.disconnect().await;
        let connection = match telegram::connect(api_id) {
            Ok(connection) => connection,
            Err(error) => {
                self.step(AccountStep::Connecting, Some(error));
                return false;
            }
        };
        let authorized = connection.client.is_authorized().await;
        self.connection = Some(connection);
        match authorized {
            Ok(true) => {
                self.ready().await;
                true
            }
            Ok(false) => {
                self.step(AccountStep::Phone, None);
                true
            }
            Err(error) => {
                let invalid_key = matches!(&error, InvocationError::Rpc(rpc) if rpc.name.starts_with("API_ID"));
                self.step(if invalid_key { AccountStep::ApiKey } else { AccountStep::Connecting }, Some(explain(&error)));
                invalid_key
            }
        }
    }

    /// The client is logged in: publish it and start answering searches. The session
    /// store has saved the login as it changed.
    async fn ready(&mut self) {
        let Some(connection) = &self.connection else { return };
        let me: User = match connection.client.get_me().await {
            Ok(me) => me,
            Err(error) => return self.step(AccountStep::Connecting, Some(explain(&error))),
        };
        let service = match TelegramService::new(connection) {
            Ok(service) => Arc::new(service),
            Err(error) => return self.step(AccountStep::Connecting, Some(format!("The drafts database could not be opened: {error}"))),
        };
        let Some(updates) = self.connection.as_mut().and_then(|connection| connection.updates.take()) else {
            return self.step(AccountStep::Connecting, Some("Telegram's updates are already being listened to; restart Seekgram.".to_owned()));
        };
        *self.shared.telegram.write().await = Some(service.clone());
        self.shared.update(|status| {
            status.account.name = Some(me.full_name());
            status.account.id = Some(marked_id(me.id()));
        });
        self.shared.update(|status| { status.account.step = AccountStep::Ready; status.account.error = None; status.account.password_hint = None; });
        self.shared.notify();
        eprintln!("Account: logged in");
        self.listener = Some(tokio::spawn(chat_search::run(self.shared.clone(), service, self.workers.clone(), me, updates)));
    }

    async fn forget(&mut self) {
        if let Some(listener) = self.listener.take() { listener.abort(); }
        *self.shared.telegram.write().await = None;
        self.code = None;
        self.password = None;
    }

    async fn handle(&mut self, command: LoginCommand) {
        match command {
            LoginCommand::ApiKey { api_id, api_hash } => {
                self.forget().await;
                self.disconnect().await;
                let saved = {
                    let mut secrets = self.shared.secrets.lock().unwrap_or_else(|e| e.into_inner());
                    secrets.api_id = Some(api_id);
                    secrets.api_hash = Some(api_hash);
                    secrets.save()
                };
                match saved {
                    Ok(()) => { self.connect().await; }
                    Err(error) => self.step(AccountStep::ApiKey, Some(error)),
                }
            }
            LoginCommand::Phone(phone) => {
                let (Some(client), Some((_, api_hash))) = (self.client(), self.key()) else { self.connect().await; return };
                if self.shared.read(|status| status.account.step) == AccountStep::Ready { return; }
                match client.request_login_code(&phone, &api_hash).await {
                    Ok(token) => { self.code = Some(token); self.step(AccountStep::Code, None); }
                    Err(error) => {
                        let invalid_key = matches!(&error, InvocationError::Rpc(rpc) if rpc.name.starts_with("API_ID"));
                        self.step(if invalid_key { AccountStep::ApiKey } else { AccountStep::Phone }, Some(explain(&error)));
                    }
                }
            }
            LoginCommand::Code(code) => {
                let (Some(client), Some(token)) = (self.client(), self.code.as_ref()) else {
                    return self.step(AccountStep::Phone, Some("Enter your phone number first.".to_owned()));
                };
                match client.sign_in(token, &code).await {
                    Ok(_) => self.ready().await,
                    Err(SignInError::PasswordRequired(token)) => {
                        let hint = token.hint().map(str::to_owned);
                        self.password = Some(token);
                        self.step(AccountStep::Password, None);
                        self.shared.update(|status| status.account.password_hint = hint);
                    }
                    Err(SignInError::InvalidCode) => self.step(AccountStep::Code, Some("That code is not right. Check the newest message from Telegram.".to_owned())),
                    Err(SignInError::SignUpRequired) => self.step(AccountStep::Phone, Some("There is no Telegram account for this number. Create one in the Telegram app first.".to_owned())),
                    Err(SignInError::Other(error)) => {
                        let expired = matches!(&error, InvocationError::Rpc(rpc) if rpc.name == "PHONE_CODE_EXPIRED");
                        self.step(if expired { AccountStep::Phone } else { AccountStep::Code }, Some(explain(&error)));
                    }
                    Err(SignInError::InvalidPassword(_)) => self.step(AccountStep::Code, Some("Telegram refused the code.".to_owned())),
                }
            }
            LoginCommand::Password(password) => {
                let Some(client) = self.client() else { self.connect().await; return };
                // A token is used up by one attempt, so a retry asks Telegram for a fresh one.
                let token = match self.password.take() {
                    Some(token) => token,
                    None => match client.invoke(&tl::functions::account::GetPassword {}).await {
                        Ok(tl::enums::account::Password::Password(password)) => PasswordToken::new(password),
                        Err(error) => return self.step(AccountStep::Password, Some(explain(&error))),
                    },
                };
                let hint = token.hint().map(str::to_owned);
                match client.check_password(token, password.as_bytes()).await {
                    Ok(_) => self.ready().await,
                    Err(SignInError::InvalidPassword(_)) => {
                        self.step(AccountStep::Password, Some("That password is not right.".to_owned()));
                        self.shared.update(|status| status.account.password_hint = hint);
                    }
                    Err(SignInError::Other(error)) => {
                        self.step(AccountStep::Password, Some(explain(&error)));
                        self.shared.update(|status| status.account.password_hint = hint);
                    }
                    Err(_) => self.step(AccountStep::Phone, Some("Telegram wants the login started again.".to_owned())),
                }
            }
            LoginCommand::Logout => {
                self.forget().await;
                if let Some(client) = self.client() {
                    // Ends the session on Telegram's side too, so it disappears from "Devices".
                    if let Err(error) = client.sign_out().await { eprintln!("Account: Telegram did not confirm the logout: {error}"); }
                }
                // The session database is open until its connection is gone.
                self.disconnect().await;
                let session = paths::session_db();
                let mut files = vec![paths::legacy_session_file(), session.clone()];
                files.extend(["-wal", "-shm"].map(|suffix| { let mut name = session.clone().into_os_string(); name.push(suffix); name.into() }));
                for file in files {
                    if let Err(error) = std::fs::remove_file(&file) {
                        if error.kind() != std::io::ErrorKind::NotFound { eprintln!("Account: the saved login could not be deleted: {error}"); }
                    }
                }
                self.connect().await;
            }
        }
    }
}

/// Runs until shutdown, serving login steps from the control API.
pub async fn run(shared: Arc<Shared>, workers: Arc<Workers>, mut commands: mpsc::Receiver<LoginRequest>) {
    let mut account = Account { shared: shared.clone(), workers, connection: None, code: None, password: None, listener: None };
    let mut reachable = account.connect().await;
    loop {
        tokio::select! {
            request = commands.recv() => {
                let Some(request) = request else { break };
                account.handle(request.command).await;
                reachable = true;
                let _ = request.done.send(());
            }
            _ = tokio::time::sleep(std::time::Duration::from_secs(15)), if !reachable => reachable = account.connect().await,
            _ = shared.stopped() => break,
        }
    }
    account.forget().await;
    account.disconnect().await;
}
