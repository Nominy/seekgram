//! The login wizard: one question at a time until Telegram lets Seekgram in.
//!
//! Which question is on screen comes from the service (`status.account.step`); the
//! panel only collects the answer and posts it.

use std::cmp::Ordering;

use gpui::{Context, Div, FontWeight, IntoElement, div, prelude::*, px};
use gpui_component::{
    Icon, Sizable as _,
    button::{Button, ButtonVariants as _},
    input::Input,
    spinner::Spinner,
};

use seekgram_core::status::{AccountStep, Status};

use super::{
    panel::{Control, Panel},
    theme::{self, color, tint},
    widgets::{card, column, heading, icon, note, page, problem, row, text, title},
};

impl Panel {
    pub(super) fn render_wizard(&self, status: &Status, cx: &mut Context<Self>) -> impl IntoElement {
        let step = self.visible_step();
        let question = match step {
            AccountStep::ApiKey => self.step_api_key(status, cx),
            AccountStep::Phone => self.step_phone(status, cx),
            AccountStep::Code => self.step_code(status, cx),
            AccountStep::Password => self.step_password(status, cx),
            AccountStep::Connecting | AccountStep::Ready => self.step_connecting(status),
        };
        page(
            "wizard",
            column()
                .w_full()
                .max_w(px(540.))
                .gap_5()
                .child(
                    column()
                        .gap_1()
                        .child(title("Welcome to Seekgram"))
                        .child(note("Log in to Telegram once. Seekgram then indexes your photos, videos and audio on this PC so you can search them.")),
                )
                .child(self.stepper(step))
                .child(question),
        )
    }

    /// Where the login stands. The password step only appears for accounts that
    /// have two-step verification, so it is not promised up front.
    fn stepper(&self, visible: AccountStep) -> Div {
        let current = if matches!(visible, AccountStep::Connecting | AccountStep::Ready) { self.wizard_step.unwrap_or_default() } else { visible };
        let mut steps = vec![(AccountStep::ApiKey, "Telegram key"), (AccountStep::Phone, "Phone number"), (AccountStep::Code, "Login code")];
        if current == AccountStep::Password {
            steps.push((AccountStep::Password, "Password"));
        }
        let position = steps.iter().position(|(step, _)| *step == current).unwrap_or(0);
        let mut line = row().w_full().gap_3();
        for (index, (_, label)) in steps.into_iter().enumerate() {
            if index > 0 {
                line = line.child(div().flex_1().h(px(1.)).bg(color(theme::BORDER)));
            }
            line = line.child(step_marker(index + 1, label, index.cmp(&position)));
        }
        line
    }

    /// What went wrong with the last answer: the panel's own check, or Telegram's verdict.
    fn login_problem(&self, status: &Status) -> Option<Div> {
        let reported = status.account.error.clone().filter(|error| !(self.change_number && self.hidden_error.as_ref() == Some(error)));
        self.error(Control::Login).map(str::to_owned).or(reported).map(problem)
    }

    fn continue_button(&self, cx: &mut Context<Self>) -> Button {
        Button::new("continue")
            .primary()
            .label("Continue")
            .loading(self.busy(Control::Login))
            .on_click(cx.listener(|panel, _, window, cx| panel.submit_login(window, cx)))
    }

    fn step_api_key(&self, status: &Status, cx: &mut Context<Self>) -> Div {
        card()
            .gap_4()
            .child(question("Get your Telegram key", "Telegram gives a key to every app that connects to it. Getting yours takes about a minute."))
            .child(
                column()
                    .gap_2()
                    .child(numbered(1, "Open my.telegram.org and log in with your phone number."))
                    .child(numbered(2, "Choose \"API development tools\" and create an app with any name."))
                    .child(numbered(3, "Copy the api_id and the api_hash it shows into the boxes below.")),
            )
            .child(
                row().child(
                    Button::new("open-telegram-site")
                        .outline()
                        .small()
                        .icon(Icon::default().path("icons/external-link.svg"))
                        .label("Open my.telegram.org")
                        .on_click(cx.listener(|panel, _, _, cx| panel.open_telegram_site(cx))),
                ),
            )
            .child(column().gap_2().child(Input::new(&self.inputs.api_id)).child(Input::new(&self.inputs.api_hash)))
            .children(self.login_problem(status))
            .child(note("The key stays on this PC, stored encrypted, and is only ever sent to Telegram."))
            .child(row().child(self.continue_button(cx)))
    }

    fn step_phone(&self, status: &Status, cx: &mut Context<Self>) -> Div {
        card()
            .gap_4()
            .child(question("Your phone number", "The number of your Telegram account in international format: a plus sign and the country code first."))
            .child(Input::new(&self.inputs.phone))
            .children(self.login_problem(status))
            .child(row().gap_4().child(self.continue_button(cx)).when(self.change_number, |actions| {
                actions.child(
                    Button::new("back-to-code")
                        .link()
                        .label("Back to the code")
                        .on_click(cx.listener(|panel, _, window, cx| panel.set_change_number(false, window, cx))),
                )
            }))
    }

    fn step_code(&self, status: &Status, cx: &mut Context<Self>) -> Div {
        card()
            .gap_4()
            .child(question("Enter the login code", "Telegram sent a code to your Telegram app, in the chat named Telegram. It is not a text message."))
            .child(Input::new(&self.inputs.code))
            .children(self.login_problem(status))
            .child(
                row().gap_4().child(self.continue_button(cx)).child(
                    Button::new("change-number")
                        .link()
                        .label("Use a different number")
                        .on_click(cx.listener(|panel, _, window, cx| panel.set_change_number(true, window, cx))),
                ),
            )
    }

    fn step_password(&self, status: &Status, cx: &mut Context<Self>) -> Div {
        let hint = status.account.password_hint.as_deref().filter(|hint| !hint.is_empty());
        card()
            .gap_4()
            .child(question("Two-step verification", "Your account is protected by a password of your own. Enter it to finish logging in."))
            .child(Input::new(&self.inputs.password))
            .children(hint.map(|hint| note(format!("Hint: {hint}"))))
            .children(self.login_problem(status))
            .child(row().child(self.continue_button(cx)))
    }

    fn step_connecting(&self, status: &Status) -> Div {
        card()
            .gap_4()
            .child(row().gap_3().child(Spinner::new().color(color(theme::ACCENT))).child(text("Connecting to Telegram...")))
            .children(self.login_problem(status))
    }
}

/// The heading of a step and one sentence on what is being asked.
fn question(name: &'static str, explanation: &'static str) -> Div {
    column().gap_1().child(heading(name)).child(note(explanation))
}

fn numbered(number: usize, instruction: &'static str) -> Div {
    row()
        .items_start()
        .gap_3()
        .child(div().w(px(16.)).flex_shrink_0().text_sm().text_color(color(theme::ACCENT)).font_weight(FontWeight::SEMIBOLD).child(format!("{number}.")))
        .child(text(instruction).flex_1().min_w_0())
}

/// One entry of the stepper: done, current or still to come.
fn step_marker(number: usize, label: &'static str, order: Ordering) -> Div {
    let badge = row().size(px(22.)).flex_shrink_0().justify_center().rounded_full().text_xs().font_weight(FontWeight::SEMIBOLD);
    let badge = match order {
        Ordering::Less => badge.bg(tint(theme::ACCENT)).child(icon("icons/check.svg").size_3().text_color(color(theme::ACCENT))),
        Ordering::Equal => badge.bg(color(theme::ACCENT)).text_color(color(0xffffff)).child(number.to_string()),
        Ordering::Greater => badge.border_1().border_color(color(theme::BORDER)).text_color(color(theme::MUTED)).child(number.to_string()),
    };
    let label = if order == Ordering::Equal { text(label).font_weight(FontWeight::MEDIUM) } else { note(label) };
    row().flex_shrink_0().gap_2().child(badge).child(label)
}
