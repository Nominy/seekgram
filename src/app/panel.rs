//! The panel's state: its link to the background service, the last status it saw,
//! and the requests in flight.
//!
//! All HTTP runs on gpui's background executor. The UI thread only starts requests
//! and applies their results, so the window stays responsive even when Telegram or
//! the service is slow to answer.

use std::{
    collections::{HashMap, HashSet},
    time::Duration,
};

use gpui::{App, ClipboardItem, Context, Entity, FontWeight, IntoElement, Render, Subscription, Window, div, prelude::*, px};
use gpui_component::{
    TitleBar,
    button::{Button, ButtonVariants as _},
    input::{InputEvent, InputState},
    spinner::Spinner,
};

use crate::{
    cli,
    config::{Secrets, Settings},
    paths,
    service::state::{AccountStep, IndexerState, SetupState, Status},
};

use super::{
    client::{ApiError, Client, Reply},
    theme::{self, color},
    widgets::{self, Tone, centered, column, note, problem, row},
};

const POLL_INTERVAL: Duration = Duration::from_secs(1);
/// Failed polls in a row before a connected panel decides the service is gone.
/// One slow answer must not throw the user out of the dashboard.
const LOST_AFTER: u8 = 3;
/// How long a "Copied" confirmation stays on a button.
const COPIED_FOR: Duration = Duration::from_secs(2);

/// The control a request belongs to, so its spinner and its error appear next to it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Control {
    Login,
    Repair,
    Indexing,
    SearchChat,
    Autostart,
    DataFolder,
    Logout,
    Shutdown,
    Mcp,
    Workers,
    Concurrency,
}

/// Which copy button was pressed last, to confirm it on the button itself.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Copied {
    McpUrl,
    Token,
}

/// How the panel stands with the background service.
pub enum Link {
    /// Waiting for the first answer. `launched` is set once the panel has started the
    /// service itself; `problem` once that is known to have failed.
    Starting { launched: bool, problem: Option<String> },
    Connected { misses: u8 },
    /// Not running. `requested`: the user stopped it from this panel. `down_seen`:
    /// a poll has found the port closed since, so the next answer is a new service
    /// and not the old one still finishing its shutdown.
    Stopped { requested: bool, down_seen: bool },
    /// The panel cannot even try: its own settings or secrets are unreadable.
    Unusable(String),
}

/// The text fields of the login wizard. They live as long as the panel so a value
/// survives a failed attempt.
pub struct LoginInputs {
    pub api_id: Entity<InputState>,
    pub api_hash: Entity<InputState>,
    pub phone: Entity<InputState>,
    pub code: Entity<InputState>,
    pub password: Entity<InputState>,
}

impl LoginInputs {
    fn new(window: &mut Window, cx: &mut Context<Panel>) -> Self {
        let mut field = |placeholder: &'static str, masked: bool| cx.new(|cx| InputState::new(window, cx).placeholder(placeholder).masked(masked));
        Self {
            api_id: field("api_id, for example 1234567", false),
            api_hash: field("api_hash", false),
            phone: field("+44 7700 900123", false),
            code: field("Login code", false),
            password: field("Password", true),
        }
    }

    fn all(&self) -> [&Entity<InputState>; 5] {
        [&self.api_id, &self.api_hash, &self.phone, &self.code, &self.password]
    }

    /// The field the user should be typing in when `step` appears.
    fn first(&self, step: AccountStep) -> Option<&Entity<InputState>> {
        match step {
            AccountStep::ApiKey => Some(&self.api_id),
            AccountStep::Phone => Some(&self.phone),
            AccountStep::Code => Some(&self.code),
            AccountStep::Password => Some(&self.password),
            AccountStep::Connecting | AccountStep::Ready => None,
        }
    }
}

pub struct Panel {
    client: Option<Client>,
    pub(super) link: Link,
    /// The last status the service reported; kept while it is briefly unreachable.
    pub(super) status: Option<Status>,
    /// Bumped whenever an action starts or finishes. A poll that began under an older
    /// value may carry a status from before the action and is dropped, so a switch
    /// the user just flipped does not flicker back.
    epoch: u64,
    pending: HashSet<Control>,
    errors: HashMap<Control, String>,
    pub(super) inputs: LoginInputs,
    /// The last step that asked for input; keeps the stepper in place while the
    /// service reports `connecting`. `None` until a question has been asked: a
    /// logged-in account reconnecting at startup is not shown the login wizard.
    pub(super) wizard_step: Option<AccountStep>,
    /// The step whose field was given focus, so focus moves once per step and not
    /// on every poll.
    focused_step: Option<AccountStep>,
    /// The user asked to enter another phone number while Telegram waits for a code.
    pub(super) change_number: bool,
    /// The service's error at the moment the user went back to the phone step. It is
    /// about the code, so it is not shown under the phone field.
    pub(super) hidden_error: Option<String>,
    pub(super) confirm_logout: bool,
    pub(super) advanced_open: bool,
    pub(super) copied: Option<Copied>,
    copied_serial: u64,
    log_path: String,
    _subscriptions: Vec<Subscription>,
}

impl Panel {
    pub fn new(window: &mut Window, cx: &mut Context<Self>) -> Self {
        let inputs = LoginInputs::new(window, cx);
        let subscriptions = inputs.all().map(|input| cx.subscribe_in(input, window, Self::on_input)).into();
        let connection = Secrets::load()
            .map_err(|reason| format!("Seekgram could not read its saved access key on this PC ({reason})."))
            .and_then(|secrets| Client::new(Settings::load().port, secrets.token));
        let (client, link) = match connection {
            Ok(client) => (Some(client), Link::Starting { launched: false, problem: None }),
            Err(reason) => (None, Link::Unusable(reason)),
        };
        if let Some(client) = client.clone() {
            Self::poll(client, window, cx);
        }
        Self {
            client,
            link,
            status: None,
            epoch: 0,
            pending: HashSet::new(),
            errors: HashMap::new(),
            inputs,
            wizard_step: None,
            focused_step: None,
            change_number: false,
            hidden_error: None,
            confirm_logout: false,
            advanced_open: false,
            copied: None,
            copied_serial: 0,
            log_path: paths::logs_dir().join("service.log").to_string_lossy().into_owned(),
            _subscriptions: subscriptions,
        }
    }

    /// Asks for the status once a second for as long as the window lives.
    fn poll(client: Client, window: &mut Window, cx: &mut Context<Self>) {
        cx.spawn_in(window, async move |this, cx| {
            loop {
                let Ok(epoch) = this.update(cx, |panel, _| panel.epoch) else { return };
                let request = client.clone();
                let reply = cx.background_executor().spawn(async move { request.status() }).await;
                let delivered = cx.update(|window, cx| this.update(cx, |panel, cx| panel.polled(epoch, reply, window, cx)).is_ok());
                if !matches!(delivered, Ok(true)) {
                    return;
                }
                cx.background_executor().timer(POLL_INTERVAL).await;
            }
        })
        .detach();
    }

    fn polled(&mut self, epoch: u64, reply: Reply, window: &mut Window, cx: &mut Context<Self>) {
        match reply {
            // A service that was told to stop keeps answering until it has shut down.
            Ok(_) if matches!(self.link, Link::Stopped { down_seen: false, .. }) => {}
            Ok(status) => {
                self.link = Link::Connected { misses: 0 };
                if epoch == self.epoch {
                    self.apply(status, window, cx);
                }
            }
            Err(error) => self.missed(error, cx),
        }
        cx.notify();
    }

    fn missed(&mut self, error: ApiError, cx: &mut Context<Self>) {
        let (mut launch, mut lost) = (false, false);
        match &mut self.link {
            Link::Starting { launched, problem } => match error {
                ApiError::Unreachable if !*launched => {
                    *launched = true;
                    launch = true;
                }
                ApiError::Refused(message) => *problem = Some(message),
                ApiError::Unreachable | ApiError::Broken(_) => {}
            },
            Link::Connected { misses } => {
                *misses += 1;
                lost = *misses >= LOST_AFTER;
            }
            Link::Stopped { down_seen, .. } => *down_seen |= matches!(error, ApiError::Unreachable),
            Link::Unusable(_) => {}
        }
        if launch {
            self.launch(cx);
        }
        if lost {
            self.link = Link::Stopped { requested: false, down_seen: true };
            self.pending.clear();
        }
    }

    /// Starts the background service. `cli::start_service` blocks until the service
    /// answers or 20 seconds have passed, so it runs off the UI thread; the poll loop
    /// notices on its own when the service is up.
    fn launch(&mut self, cx: &mut Context<Self>) {
        cx.spawn(async move |this, cx| {
            let outcome = cx.background_executor().spawn(async { cli::start_service() }).await;
            if let Err(reason) = outcome {
                this.update(cx, |panel, cx| {
                    if let Link::Starting { problem, .. } = &mut panel.link {
                        *problem = Some(reason);
                        cx.notify();
                    }
                })
                .ok();
            }
        })
        .detach();
    }

    /// Takes in a status from the service and moves the wizard along with it.
    fn apply(&mut self, status: Status, window: &mut Window, cx: &mut Context<Self>) {
        let step = status.account.step;
        if self.status.as_ref().map(|previous| previous.account.step) != Some(step) {
            // What was typed for a finished step must not linger in a hidden field.
            self.clear(&[&self.inputs.code, &self.inputs.password], window, cx);
            if step == AccountStep::Ready {
                self.clear(&[&self.inputs.api_hash], window, cx);
            }
            if step != AccountStep::Code {
                self.change_number = false;
            }
            self.confirm_logout = false;
        }
        match step {
            AccountStep::Connecting => {}
            AccountStep::Ready => self.wizard_step = None,
            asking => self.wizard_step = Some(asking),
        }
        self.status = Some(status);
        self.focus_step(window, cx);
    }

    fn clear(&self, inputs: &[&Entity<InputState>], window: &mut Window, cx: &mut Context<Self>) {
        for input in inputs {
            input.update(cx, |state, cx| state.set_value("", window, cx));
        }
    }

    /// The wizard step on screen: the service's, unless the user went back to change
    /// the phone number.
    pub(super) fn visible_step(&self) -> AccountStep {
        if self.change_number {
            return AccountStep::Phone;
        }
        self.status.as_ref().map(|status| status.account.step).unwrap_or_default()
    }

    fn focus_step(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let step = self.visible_step();
        if self.focused_step == Some(step) {
            return;
        }
        self.focused_step = Some(step);
        if let Some(input) = self.inputs.first(step) {
            input.update(cx, |state, cx| state.focus(window, cx));
        }
    }

    fn on_input(&mut self, _: &Entity<InputState>, event: &InputEvent, window: &mut Window, cx: &mut Context<Self>) {
        if matches!(event, InputEvent::PressEnter { .. }) {
            self.submit_login(window, cx);
        }
    }

    pub(super) fn busy(&self, control: Control) -> bool {
        self.pending.contains(&control)
    }

    pub(super) fn error(&self, control: Control) -> Option<&str> {
        self.errors.get(&control).map(String::as_str)
    }

    fn fail(&mut self, control: Control, message: impl Into<String>, cx: &mut Context<Self>) {
        self.errors.insert(control, message.into());
        cx.notify();
    }

    /// Runs one API call in the background and applies its result. A control that
    /// already has a request in flight ignores further presses until it returns.
    fn act(&mut self, control: Control, call: impl FnOnce(&Client) -> Reply + Send + 'static, window: &mut Window, cx: &mut Context<Self>) {
        let Some(client) = self.client.clone() else { return };
        if !self.pending.insert(control) {
            return;
        }
        self.errors.remove(&control);
        self.epoch += 1;
        cx.notify();
        cx.spawn_in(window, async move |this, cx| {
            let reply = cx.background_executor().spawn(async move { call(&client) }).await;
            cx.update(|window, cx| this.update(cx, |panel, cx| panel.acted(control, reply, window, cx)).ok()).ok();
        })
        .detach();
    }

    fn acted(&mut self, control: Control, reply: Reply, window: &mut Window, cx: &mut Context<Self>) {
        self.pending.remove(&control);
        self.epoch += 1;
        match reply {
            Ok(status) => {
                if control == Control::Shutdown {
                    self.link = Link::Stopped { requested: true, down_seen: false };
                }
                // A new number was accepted: back to the service's own step, the code.
                if control == Control::Login && status.account.error.is_none() {
                    self.change_number = false;
                }
                self.apply(status, window, cx);
            }
            // A service that goes away without answering the request to stop has stopped.
            Err(ApiError::Unreachable | ApiError::Broken(_)) if control == Control::Shutdown => {
                self.link = Link::Stopped { requested: true, down_seen: false };
            }
            Err(error) => {
                self.errors.insert(control, error.message());
            }
        }
        cx.notify();
    }

    pub(super) fn submit_login(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        match self.visible_step() {
            AccountStep::ApiKey => {
                let api_hash = typed(&self.inputs.api_hash, cx).trim().to_owned();
                let Ok(api_id) = typed(&self.inputs.api_id, cx).trim().parse::<i64>() else {
                    return self.fail(Control::Login, "The api_id is a number, usually 6 to 8 digits.", cx);
                };
                if api_hash.is_empty() {
                    return self.fail(Control::Login, "Paste the api_hash from my.telegram.org as well.", cx);
                }
                self.act(Control::Login, move |client| client.login_api_key(api_id, &api_hash), window, cx);
            }
            AccountStep::Phone => {
                let phone = typed(&self.inputs.phone, cx).trim().to_owned();
                if phone.is_empty() {
                    return self.fail(Control::Login, "Enter your phone number with its country code, like +44 7700 900123.", cx);
                }
                self.act(Control::Login, move |client| client.login_phone(&phone), window, cx);
            }
            AccountStep::Code => {
                let code = typed(&self.inputs.code, cx).trim().to_owned();
                if code.is_empty() {
                    return self.fail(Control::Login, "Enter the code Telegram sent you.", cx);
                }
                self.act(Control::Login, move |client| client.login_code(&code), window, cx);
            }
            AccountStep::Password => {
                let password = typed(&self.inputs.password, cx);
                if password.is_empty() {
                    return self.fail(Control::Login, "Enter your two-step verification password.", cx);
                }
                self.act(Control::Login, move |client| client.login_password(&password), window, cx);
            }
            AccountStep::Connecting | AccountStep::Ready => {}
        }
    }

    /// Shows the phone step again while Telegram is waiting for a code, or returns
    /// to the code. Submitting a new number makes the service ask for a new code.
    pub(super) fn set_change_number(&mut self, change: bool, window: &mut Window, cx: &mut Context<Self>) {
        self.change_number = change;
        self.hidden_error = self.status.as_ref().filter(|_| change).and_then(|status| status.account.error.clone());
        self.errors.remove(&Control::Login);
        self.focus_step(window, cx);
        cx.notify();
    }

    pub(super) fn open_telegram_site(&mut self, cx: &mut Context<Self>) {
        cx.open_url("https://my.telegram.org/apps");
    }

    pub(super) fn repair(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.act(Control::Repair, Client::repair_setup, window, cx);
    }

    pub(super) fn set_indexing(&mut self, on: bool, window: &mut Window, cx: &mut Context<Self>) {
        self.change_setting(Control::Indexing, |settings| settings.indexing = on, move |client| client.set_indexing(on), window, cx);
    }

    pub(super) fn set_autostart(&mut self, on: bool, window: &mut Window, cx: &mut Context<Self>) {
        self.change_setting(Control::Autostart, |settings| settings.autostart = on, move |client| client.set_autostart(on), window, cx);
    }

    pub(super) fn set_mcp(&mut self, on: bool, window: &mut Window, cx: &mut Context<Self>) {
        self.change_setting(Control::Mcp, |settings| settings.mcp = on, move |client| client.set_mcp(on), window, cx);
    }

    pub(super) fn set_workers(&mut self, workers: u8, window: &mut Window, cx: &mut Context<Self>) {
        self.change_setting(Control::Workers, |settings| settings.workers = workers, move |client| client.set_workers(workers), window, cx);
    }

    pub(super) fn set_concurrency(&mut self, concurrency: u8, window: &mut Window, cx: &mut Context<Self>) {
        self.change_setting(Control::Concurrency, |settings| settings.concurrency = concurrency, move |client| client.set_concurrency(concurrency), window, cx);
    }

    /// Shows the new value at once and sends it. If the service refuses, its answer
    /// (or the next poll) puts the real value back and the error appears by the control.
    fn change_setting(
        &mut self,
        control: Control,
        show: impl FnOnce(&mut Settings),
        send: impl FnOnce(&Client) -> Reply + Send + 'static,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.busy(control) {
            return;
        }
        if let Some(status) = &mut self.status {
            show(&mut status.settings);
        }
        self.act(control, send, window, cx);
    }

    pub(super) fn create_search_group(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.act(Control::SearchChat, Client::create_search_group, window, cx);
    }

    pub(super) fn use_saved_messages(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.act(Control::SearchChat, Client::use_saved_messages, window, cx);
    }

    pub(super) fn open_data_folder(&mut self, cx: &mut Context<Self>) {
        let Some(folder) = self.status.as_ref().map(|status| status.data_dir.clone()) else { return };
        match std::process::Command::new("explorer.exe").arg(folder).spawn() {
            Ok(_) => {
                self.errors.remove(&Control::DataFolder);
                cx.notify();
            }
            Err(error) => self.fail(Control::DataFolder, format!("Windows could not open the folder ({error})."), cx),
        }
    }

    pub(super) fn set_confirm_logout(&mut self, confirm: bool, cx: &mut Context<Self>) {
        self.confirm_logout = confirm;
        cx.notify();
    }

    pub(super) fn logout(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.confirm_logout = false;
        self.act(Control::Logout, Client::logout, window, cx);
    }

    pub(super) fn shutdown(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.act(Control::Shutdown, Client::shutdown, window, cx);
    }

    /// Goes back to waiting for the service; the next poll starts it if nothing answers.
    fn start(&mut self, cx: &mut Context<Self>) {
        self.link = Link::Starting { launched: false, problem: None };
        self.errors.clear();
        cx.notify();
    }

    pub(super) fn toggle_advanced(&mut self, cx: &mut Context<Self>) {
        self.advanced_open = !self.advanced_open;
        cx.notify();
    }

    pub(super) fn copy_mcp_url(&mut self, cx: &mut Context<Self>) {
        if let Some(url) = self.status.as_ref().map(|status| status.mcp_url.clone()) {
            self.copy(Copied::McpUrl, url, cx);
        }
    }

    pub(super) fn copy_token(&mut self, cx: &mut Context<Self>) {
        if let Some(token) = self.client.as_ref().map(|client| client.token().to_owned()) {
            self.copy(Copied::Token, token, cx);
        }
    }

    fn copy(&mut self, what: Copied, text: String, cx: &mut Context<Self>) {
        cx.write_to_clipboard(ClipboardItem::new_string(text));
        self.copied = Some(what);
        self.copied_serial += 1;
        let serial = self.copied_serial;
        cx.notify();
        cx.spawn(async move |this, cx| {
            cx.background_executor().timer(COPIED_FOR).await;
            this.update(cx, |panel, cx| {
                if panel.copied_serial == serial {
                    panel.copied = None;
                    cx.notify();
                }
            })
            .ok();
        })
        .detach();
    }

    /// The header pill: one word for how things stand, shown once logged in.
    fn health(status: &Status) -> (&'static str, Tone) {
        if status.setup.state == SetupState::Failed || status.indexer.state == IndexerState::Failed {
            ("Needs attention", Tone::Danger)
        } else if status.setup.state != SetupState::Ready {
            ("Setting up", Tone::Warning)
        } else if !status.settings.indexing || status.indexer.state == IndexerState::Off {
            ("Paused", Tone::Quiet)
        } else {
            ("Running", Tone::Accent)
        }
    }

    fn render_header(&self) -> impl IntoElement {
        let account = match (&self.link, &self.status) {
            (Link::Connected { .. }, Some(status)) if status.account.step == AccountStep::Ready => Some(status),
            _ => None,
        };
        let brand = row()
            .gap_2()
            .child(widgets::icon("icons/mark.svg").size_5().text_color(color(theme::ACCENT)))
            .child(div().text_sm().font_weight(FontWeight::SEMIBOLD).child("Seekgram"));
        let state = row().gap_3().pr_3().when_some(account, |state, status| {
            let (label, tone) = Self::health(status);
            state
                .when_some(status.account.name.clone(), |state, name| state.child(note(name).max_w(px(260.)).truncate()))
                .child(widgets::pill(label, tone))
        });
        TitleBar::new().bg(color(theme::SURFACE)).child(brand).child(state)
    }

    fn render_starting(&self, reason: Option<&str>, cx: &mut Context<Self>) -> impl IntoElement {
        let Some(reason) = reason else {
            return centered(row().gap_3().child(Spinner::new().color(color(theme::ACCENT))).child(widgets::text("Starting Seekgram...")));
        };
        centered(
            column()
                .max_w(px(520.))
                .gap_3()
                .items_center()
                .text_center()
                .child(widgets::title("Seekgram did not start"))
                .child(note(reason.to_owned()))
                // The start-up error names the log itself when it knows nothing better.
                .when(!reason.contains(self.log_path.as_str()), |message| message.child(note(format!("The log may say why: {}", self.log_path))))
                .child(Button::new("start-again").primary().label("Try again").on_click(cx.listener(|panel, _, _, cx| panel.start(cx)))),
        )
    }

    fn render_stopped(&self, requested: bool, down_seen: bool, cx: &mut Context<Self>) -> impl IntoElement {
        if !down_seen {
            return centered(row().gap_3().child(Spinner::new().color(color(theme::ACCENT))).child(widgets::text("Stopping Seekgram...")));
        }
        let (title, explanation) = if requested {
            ("Seekgram is stopped", "Nothing is being indexed and searches are not answered until you start it again.".to_owned())
        } else {
            ("Seekgram is not running", format!("The background service stopped answering. The log may say why: {}", self.log_path))
        };
        centered(
            column()
                .max_w(px(520.))
                .gap_3()
                .items_center()
                .text_center()
                .child(widgets::title(title))
                .child(note(explanation))
                .child(Button::new("start").primary().label("Start").on_click(cx.listener(|panel, _, _, cx| panel.start(cx)))),
        )
    }

    fn render_unusable(reason: &str) -> impl IntoElement {
        centered(
            column()
                .max_w(px(520.))
                .gap_3()
                .items_center()
                .text_center()
                .child(widgets::title("Seekgram cannot open"))
                .child(problem(reason.to_owned())),
        )
    }

    /// A logged-in account getting back online; no question to ask.
    fn render_connecting(status: &Status) -> impl IntoElement {
        centered(
            column()
                .max_w(px(520.))
                .gap_3()
                .items_center()
                .text_center()
                .child(row().gap_3().child(Spinner::new().color(color(theme::ACCENT))).child(widgets::text("Connecting to Telegram...")))
                .children(status.account.error.clone().map(problem)),
        )
    }
}

/// What the user typed into a wizard field.
fn typed(input: &Entity<InputState>, cx: &App) -> String {
    input.read(cx).value().to_string()
}

impl Render for Panel {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let body = match (&self.link, &self.status) {
            (Link::Unusable(reason), _) => Self::render_unusable(reason).into_any_element(),
            (Link::Starting { problem, .. }, _) => self.render_starting(problem.as_deref(), cx).into_any_element(),
            (Link::Stopped { requested, down_seen }, _) => self.render_stopped(*requested, *down_seen, cx).into_any_element(),
            (Link::Connected { .. }, None) => self.render_starting(None, cx).into_any_element(),
            (Link::Connected { .. }, Some(status)) if status.account.step == AccountStep::Ready => self.render_dashboard(status, cx).into_any_element(),
            (Link::Connected { .. }, Some(status)) if status.account.step == AccountStep::Connecting && self.wizard_step.is_none() => {
                Self::render_connecting(status).into_any_element()
            }
            (Link::Connected { .. }, Some(status)) => self.render_wizard(status, cx).into_any_element(),
        };
        column().size_full().bg(color(theme::SURFACE)).text_color(color(theme::TEXT)).child(self.render_header()).child(body)
    }
}
