//! The dashboard shown once logged in: one card per thing the user can look at or
//! change. Everything on it is read from the last status; every control posts to
//! the control API through `Panel`.

use gpui::{App, Context, Div, IntoElement, div, prelude::*, px};
use gpui_component::{
    Disableable as _, Icon, Sizable as _,
    button::{Button, ButtonVariants as _},
    progress::Progress,
    switch::Switch,
};

use crate::service::state::{IndexCounts, Indexer, IndexerState, Setup, SetupState, SetupStep, Status};

use super::{
    format,
    panel::{Control, Copied, Panel},
    widgets::{Tone, card, code, column, divider, dot, heading, icon, note, page, problem, row, setting, stat, text},
};

/// The panel offers automatic, one or two model workers; more rarely fits in VRAM.
const MAX_WORKERS: u8 = 2;
/// The service accepts 1 to 16 media messages prepared at once.
const MAX_CONCURRENCY: u8 = 16;

const CHEAT_SHEET: [(&str, &str); 6] = [
    ("?pink dragon", "Find photos, videos and audio that match the words."),
    ("?", "As the caption of a photo you send, or as a reply to one: find more like it."),
    ("?video ...", "Search only videos."),
    ("?audio ...", "Search only audio and voice messages."),
    ("?more", "Show the next results of the last search."),
    ("?7", "Forward result number 7 to the chat."),
];

impl Panel {
    pub(super) fn render_dashboard(&self, status: &Status, cx: &mut Context<Self>) -> impl IntoElement {
        page(
            "dashboard",
            column()
                .w_full()
                .max_w(px(780.))
                .gap_4()
                .child(self.setup_card(status, cx))
                .child(self.indexing_card(status, cx))
                .child(self.search_card(status, cx))
                .child(self.settings_card(status, cx)),
        )
    }

    fn control_problem(&self, control: Control) -> Option<Div> {
        self.error(control).map(|error| problem(error.to_owned()))
    }

    /// Preparing the model worker. Prominent while it is not ready, one line afterwards.
    fn setup_card(&self, status: &Status, cx: &mut Context<Self>) -> Div {
        let setup = &status.setup;
        let busy = self.busy(Control::Repair);
        if setup.state == SetupState::Ready {
            let line = match &setup.gpu {
                Some(gpu) if !gpu.name.is_empty() => format!("Model ready on {}", gpu.name),
                _ => "Model ready".to_owned(),
            };
            return card()
                .flex_row()
                .items_center()
                .py_3()
                .child(dot(Tone::Accent))
                .child(text(line).flex_1().min_w_0())
                .children(self.control_problem(Control::Repair))
                .child(Button::new("repair").ghost().small().label("Repair").loading(busy).on_click(cx.listener(|panel, _, window, cx| panel.repair(window, cx))));
        }
        let failed = setup.state == SetupState::Failed;
        let (number, name) = setup_step(setup.step);
        card()
            .child(
                row()
                    .gap_3()
                    .justify_between()
                    .child(heading(if failed { "The search model could not be set up" } else { "Getting the search model ready" }))
                    .child(note(format!("Step {number} of 6")).flex_shrink_0()),
            )
            .child(text(if setup.state == SetupState::Pending { "Waiting to check this PC" } else { name }))
            .when(!setup.detail.is_empty(), |card| card.child(note(setup.detail.clone())))
            .children(download_progress(setup))
            .children(gpu_line(setup))
            .when(failed, |card| {
                card.child(problem(setup.error.clone().unwrap_or_else(|| "Setup stopped because of an error.".to_owned())))
                    .children(self.control_problem(Control::Repair))
                    .child(row().child(Button::new("repair").primary().label("Try again").loading(busy).on_click(cx.listener(|panel, _, window, cx| panel.repair(window, cx)))))
            })
    }

    fn indexing_card(&self, status: &Status, cx: &mut Context<Self>) -> Div {
        let indexer = &status.indexer;
        let (words, failed) = indexing_words(status);
        let active = matches!(indexer.state, IndexerState::Starting | IndexerState::Running | IndexerState::FloodWait | IndexerState::Stopping);
        card()
            .child(
                row()
                    .items_start()
                    .gap_4()
                    .child(column().flex_1().min_w_0().gap_1().child(heading("Indexing")).child(if failed { problem(words) } else { note(words) }))
                    .child(
                        Switch::new("indexing")
                            .checked(status.settings.indexing)
                            .on_click(cx.listener(|panel, on: &bool, window, cx| panel.set_indexing(*on, window, cx))),
                    ),
            )
            .children(self.control_problem(Control::Indexing))
            .when(active, |card| card.children(current_chat(indexer)).child(activity(indexer)))
            .child(totals(&status.index))
    }

    fn search_card(&self, status: &Status, cx: &mut Context<Self>) -> Div {
        let chat = &status.search_chat;
        let busy = self.busy(Control::SearchChat);
        let (place, change) = if chat.kind == "group" {
            let place = if chat.title.is_empty() { "your private Search group".to_owned() } else { format!("your private group \"{}\"", chat.title) };
            let change = Button::new("use-saved-messages")
                .label("Use Saved Messages instead")
                .on_click(cx.listener(|panel, _, window, cx| panel.use_saved_messages(window, cx)));
            (place, change)
        } else {
            let change = Button::new("create-search-group")
                .label("Create a private Search group")
                .on_click(cx.listener(|panel, _, window, cx| panel.create_search_group(window, cx)));
            ("Saved Messages".to_owned(), change)
        };
        card()
            .child(
                column()
                    .gap_1()
                    .child(heading("Search"))
                    .child(note(format!("Send a search to {place} in Telegram and Seekgram answers there."))),
            )
            .child(row().child(change.outline().small().loading(busy)))
            .children(self.control_problem(Control::SearchChat))
            .child(divider())
            .child(cheat_sheet(cx))
    }

    fn settings_card(&self, status: &Status, cx: &mut Context<Self>) -> Div {
        card()
            .gap_4()
            .child(heading("Settings"))
            .child(
                row()
                    .gap_4()
                    .child(setting("Start with Windows", "Seekgram keeps working in the background after you sign in to Windows."))
                    .child(
                        Switch::new("autostart")
                            .checked(status.settings.autostart)
                            .on_click(cx.listener(|panel, on: &bool, window, cx| panel.set_autostart(*on, window, cx))),
                    ),
            )
            .children(self.control_problem(Control::Autostart))
            .child(self.settings_actions(cx))
            .children(self.control_problem(Control::DataFolder))
            .children(self.control_problem(Control::Logout))
            .children(self.control_problem(Control::Shutdown))
            .child(divider())
            .child(self.advanced_toggle(cx))
            .when(self.advanced_open, |card| card.child(self.advanced(status, cx)))
    }

    /// The data folder, logging out and stopping. Logging out asks first, in place.
    fn settings_actions(&self, cx: &mut Context<Self>) -> Div {
        if self.confirm_logout {
            return row()
                .gap_3()
                .flex_wrap()
                .child(text("Log out of Telegram on this PC? Your index is kept.").flex_1().min_w_0())
                .child(Button::new("logout-confirm").danger().small().label("Log out").on_click(cx.listener(|panel, _, window, cx| panel.logout(window, cx))))
                .child(Button::new("logout-cancel").ghost().small().label("Cancel").on_click(cx.listener(|panel, _, _, cx| panel.set_confirm_logout(false, cx))));
        }
        row()
            .gap_2()
            .flex_wrap()
            .child(
                Button::new("open-data-folder")
                    .outline()
                    .small()
                    .icon(Icon::default().path("icons/folder.svg"))
                    .label("Open data folder")
                    .on_click(cx.listener(|panel, _, _, cx| panel.open_data_folder(cx))),
            )
            .child(
                Button::new("logout")
                    .outline()
                    .small()
                    .label("Log out of Telegram")
                    .loading(self.busy(Control::Logout))
                    .on_click(cx.listener(|panel, _, _, cx| panel.set_confirm_logout(true, cx))),
            )
            .child(
                Button::new("stop")
                    .outline()
                    .small()
                    .label("Stop Seekgram")
                    .loading(self.busy(Control::Shutdown))
                    .on_click(cx.listener(|panel, _, window, cx| panel.shutdown(window, cx))),
            )
    }

    fn advanced_toggle(&self, cx: &mut Context<Self>) -> impl IntoElement {
        row()
            .id("advanced")
            .gap_2()
            .cursor_pointer()
            .child(icon(if self.advanced_open { "icons/chevron-down.svg" } else { "icons/chevron-right.svg" }))
            .child(text("Advanced"))
            .on_click(cx.listener(|panel, _, _, cx| panel.toggle_advanced(cx)))
    }

    /// What most people never need: the MCP endpoint for AI agents and performance knobs.
    fn advanced(&self, status: &Status, cx: &mut Context<Self>) -> Div {
        let settings = &status.settings;
        let (workers, concurrency) = (settings.workers, settings.concurrency);
        column()
            .gap_4()
            .child(
                row()
                    .gap_4()
                    .child(setting("Let AI agents search (MCP)", "Lets an AI assistant on this PC, such as Claude or Cursor, search your indexed media. Leave it off if you do not use one."))
                    .child(
                        Switch::new("mcp")
                            .checked(settings.mcp)
                            .on_click(cx.listener(|panel, on: &bool, window, cx| panel.set_mcp(*on, window, cx))),
                    ),
            )
            .children(self.control_problem(Control::Mcp))
            .when(settings.mcp, |section| section.child(self.mcp_details(status, cx)))
            .child(
                row()
                    .gap_4()
                    .child(setting("Model workers", "Each worker keeps the model in graphics memory (about 3 GB). Automatic picks from the memory that is free."))
                    .child(number_stepper(
                        Button::new("workers-less")
                            .disabled(workers == 0)
                            .on_click(cx.listener(move |panel, _, window, cx| panel.set_workers(workers.saturating_sub(1), window, cx))),
                        if workers == 0 { "Automatic".to_owned() } else { workers.to_string() },
                        Button::new("workers-more")
                            .disabled(workers >= MAX_WORKERS)
                            .on_click(cx.listener(move |panel, _, window, cx| panel.set_workers((workers + 1).min(MAX_WORKERS), window, cx))),
                    )),
            )
            .children(self.control_problem(Control::Workers))
            .child(
                row()
                    .gap_4()
                    .child(setting("Downloads at once", "How many media messages are downloaded and prepared at the same time."))
                    .child(number_stepper(
                        Button::new("concurrency-less")
                            .disabled(concurrency <= 1)
                            .on_click(cx.listener(move |panel, _, window, cx| panel.set_concurrency(concurrency.saturating_sub(1).max(1), window, cx))),
                        concurrency.to_string(),
                        Button::new("concurrency-more")
                            .disabled(concurrency >= MAX_CONCURRENCY)
                            .on_click(cx.listener(move |panel, _, window, cx| panel.set_concurrency(concurrency.saturating_add(1).min(MAX_CONCURRENCY), window, cx))),
                    )),
            )
            .children(self.control_problem(Control::Concurrency))
    }

    /// Where an agent connects and the token it needs; only shown while MCP is on.
    fn mcp_details(&self, status: &Status, cx: &mut Context<Self>) -> Div {
        let copied = |what: Copied, label: &'static str| if self.copied == Some(what) { "Copied" } else { label };
        column()
            .gap_2()
            .child(
                row()
                    .gap_2()
                    .flex_wrap()
                    .child(code(status.mcp_url.clone(), cx))
                    .child(
                        Button::new("copy-mcp-url")
                            .outline()
                            .xsmall()
                            .icon(Icon::default().path("icons/copy.svg"))
                            .label(copied(Copied::McpUrl, "Copy"))
                            .on_click(cx.listener(|panel, _, _, cx| panel.copy_mcp_url(cx))),
                    )
                    .child(
                        Button::new("copy-token")
                            .outline()
                            .xsmall()
                            .label(copied(Copied::Token, "Copy access token"))
                            .on_click(cx.listener(|panel, _, _, cx| panel.copy_token(cx))),
                    ),
            )
            .child(note("Add this address to the assistant as an MCP server, with the access token as its Bearer token. Keep the token private: whoever has it can search your media."))
    }
}

/// Position and name of a setup step, in the order the service runs them.
fn setup_step(step: SetupStep) -> (usize, &'static str) {
    match step {
        SetupStep::Checking => (1, "Checking this PC"),
        SetupStep::Uv => (2, "Downloading the Python manager"),
        SetupStep::Ffmpeg => (3, "Downloading the video tools"),
        SetupStep::Packages => (4, "Installing the model's software (about 3 GB)"),
        SetupStep::Model => (5, "Downloading the search model (about 1.5 GB)"),
        SetupStep::SelfTest => (6, "Testing the model on your graphics card"),
    }
}

/// A bar for the current download, when its size is known.
fn download_progress(setup: &Setup) -> Option<Div> {
    let total = setup.bytes_total.filter(|total| *total > 0)?;
    let done = setup.bytes_done.unwrap_or(0).min(total);
    Some(
        column()
            .gap_1p5()
            .child(Progress::new("setup-progress").value(format::percent(done, total)))
            .child(note(format!("{} of {}", format::bytes(done), format::bytes(total)))),
    )
}

fn gpu_line(setup: &Setup) -> Option<Div> {
    match &setup.gpu {
        Some(gpu) => Some(note(format!("Graphics card: {} ({} of memory)", gpu.name, format::bytes(gpu.vram_mb.saturating_mul(1024 * 1024))))),
        None if setup.state == SetupState::Failed => Some(note("No NVIDIA graphics card was found on this PC.")),
        None => None,
    }
}

/// The indexer's state as a sentence, and whether it is a failure.
fn indexing_words(status: &Status) -> (String, bool) {
    let indexer = &status.indexer;
    let reason = indexer.reason.as_deref().filter(|reason| !reason.is_empty()).map(sentence);
    match indexer.state {
        IndexerState::Off if !status.settings.indexing => ("Switched off. Nothing new is being indexed.".to_owned(), false),
        IndexerState::Off => (reason.unwrap_or_else(|| "Waiting to start.".to_owned()), false),
        IndexerState::Starting => ("Starting...".to_owned(), false),
        IndexerState::Running => ("Indexing your Telegram media.".to_owned(), false),
        IndexerState::Idle => ("Up to date. New media is indexed as it arrives.".to_owned(), false),
        IndexerState::FloodWait => match indexer.flood_wait_seconds {
            Some(seconds) => (format!("Telegram asked for a pause. Continuing in {}.", format::wait(seconds)), false),
            None => ("Telegram asked for a short pause.".to_owned(), false),
        },
        IndexerState::Stopping => ("Finishing the items in progress...".to_owned(), false),
        IndexerState::Failed => (reason.unwrap_or_else(|| "Indexing stopped because of an error.".to_owned()), true),
    }
}

/// Reasons arrive as fragments ("waiting for login"); shown alone they start a sentence.
fn sentence(text: &str) -> String {
    let mut characters = text.chars();
    match characters.next() {
        Some(first) => first.to_uppercase().chain(characters).collect(),
        None => String::new(),
    }
}

/// The chat being scanned and how far its messages have been read.
fn current_chat(indexer: &Indexer) -> Option<Div> {
    let chat = indexer.chat.as_ref()?;
    let name = if chat.title.is_empty() { "Untitled chat".to_owned() } else { chat.title.clone() };
    let position = (indexer.chats_total > 0).then(|| {
        let number = (indexer.chats_done + 1).min(indexer.chats_total);
        note(format!("Chat {} of {}", format::count(number.into()), format::count(indexer.chats_total.into()))).flex_shrink_0()
    });
    let scanned = format::percent(chat.message_id.max(0) as u64, chat.top_message_id.max(0) as u64);
    Some(
        column()
            .gap_1p5()
            .child(row().gap_3().justify_between().child(text(name).flex_1().min_w_0().truncate()).children(position))
            .child(Progress::new("chat-progress").value(scanned)),
    )
}

fn activity(indexer: &Indexer) -> Div {
    let between_chats = indexer.chat.is_none() && indexer.chats_total > 0;
    row()
        .gap_4()
        .flex_wrap()
        .when(between_chats, |line| line.child(note(format!("{} of {} chats checked", format::count(indexer.chats_done.into()), format::count(indexer.chats_total.into())))))
        .child(note(format!("{} items per minute", format::rate(indexer.items_per_minute))))
        .child(note(format!("{} in progress", format::count(indexer.in_flight.into()))))
}

fn totals(index: &IndexCounts) -> Div {
    row()
        .w_full()
        .flex_wrap()
        .gap_2()
        .child(stat("Images", format::count(index.images)))
        .child(stat("Video segments", format::count(index.video_segments)))
        .child(stat("Audio segments", format::count(index.audio_segments)))
        .child(stat("Chats", format::count(index.chats)))
        .child(stat("Failures", format::count(index.failures)))
        .child(stat("Index size", format::bytes(index.size_bytes)))
}

fn cheat_sheet(cx: &App) -> Div {
    column().gap_2().children(CHEAT_SHEET.iter().map(|(command, meaning)| {
        row()
            .items_start()
            .gap_3()
            .child(div().flex().w(px(132.)).flex_shrink_0().child(code(*command, cx)))
            .child(note(*meaning).flex_1().min_w_0())
    }))
}

/// A value with a button on each side to lower or raise it.
fn number_stepper(less: Button, value: String, more: Button) -> Div {
    row()
        .flex_shrink_0()
        .gap_2()
        .child(less.outline().xsmall().icon(Icon::default().path("icons/minus.svg")))
        .child(div().min_w(px(84.)).text_center().text_sm().child(value))
        .child(more.outline().xsmall().icon(Icon::default().path("icons/plus.svg")))
}
