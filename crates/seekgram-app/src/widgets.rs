//! The few visual building blocks every screen shares, so cards, notes and errors
//! look the same wherever they appear.

use gpui::{App, Div, ElementId, FontWeight, IntoElement, SharedString, Svg, div, prelude::*, px, svg};
use gpui_component::{ActiveTheme as _, scroll::ScrollableElement as _};

use super::theme::{self, color, tint};

pub fn row() -> Div {
    div().flex().flex_row().items_center()
}

pub fn column() -> Div {
    div().flex().flex_col()
}

/// A screen's content, centred and scrolling vertically when the window is short.
/// `id` keeps each screen's scroll position its own.
pub fn page(id: impl Into<ElementId>, content: impl IntoElement) -> impl IntoElement {
    div().flex_1().min_h_0().w_full().child(
        column().items_center().w_full().px_6().py_6().child(content).overflow_y_scrollbar().id(id),
    )
}

/// A message that fills the window: starting, stopped, or unable to run.
pub fn centered(content: impl IntoElement) -> Div {
    column().flex_1().min_h_0().w_full().items_center().justify_center().p_8().child(content)
}

pub fn card() -> Div {
    column()
        .w_full()
        .gap_3()
        .p_5()
        .rounded(px(10.))
        .bg(color(theme::CARD))
        .border_1()
        .border_color(color(theme::BORDER))
}

pub fn heading(text: impl Into<SharedString>) -> Div {
    div().text_base().font_weight(FontWeight::SEMIBOLD).child(text.into())
}

/// The large line at the top of a full-window message or the wizard.
pub fn title(text: impl Into<SharedString>) -> Div {
    div().text_xl().font_weight(FontWeight::SEMIBOLD).child(text.into())
}

pub fn text(text: impl Into<SharedString>) -> Div {
    div().text_sm().child(text.into())
}

/// Secondary text: explanations and captions.
pub fn note(text: impl Into<SharedString>) -> Div {
    div().text_sm().text_color(color(theme::MUTED)).child(text.into())
}

/// Why something failed, next to the control that caused it.
pub fn problem(text: impl Into<SharedString>) -> Div {
    div().text_sm().text_color(color(theme::DANGER)).child(text.into())
}

/// An icon from `assets/icons`, tinted with the surrounding text colour unless told otherwise.
pub fn icon(path: &'static str) -> Svg {
    svg().path(path).size_4().flex_shrink_0().text_color(color(theme::MUTED))
}

/// Something to type or copy exactly, set in the monospace font.
pub fn code(text: impl Into<SharedString>, cx: &App) -> Div {
    div()
        .flex_shrink_0()
        .px_2()
        .py_0p5()
        .rounded(px(5.))
        .bg(color(theme::INSET))
        .font_family(cx.theme().mono_font_family.clone())
        .text_sm()
        .child(text.into())
}

#[derive(Clone, Copy)]
pub enum Tone {
    Accent,
    Quiet,
    Warning,
    Danger,
}

impl Tone {
    fn hex(self) -> u32 {
        match self {
            Self::Accent => theme::ACCENT,
            Self::Quiet => theme::MUTED,
            Self::Warning => theme::WARNING,
            Self::Danger => theme::DANGER,
        }
    }
}

/// A small coloured dot, the quietest way to show a state.
pub fn dot(tone: Tone) -> Div {
    div().size(px(8.)).flex_shrink_0().rounded_full().bg(color(tone.hex()))
}

/// The status pill in the header.
pub fn pill(label: impl Into<SharedString>, tone: Tone) -> Div {
    row()
        .gap_1p5()
        .px_2p5()
        .py_0p5()
        .rounded_full()
        .bg(tint(tone.hex()))
        .text_xs()
        .font_weight(FontWeight::MEDIUM)
        .child(dot(tone))
        .child(label.into())
}

/// One number with its caption, for the index totals.
pub fn stat(label: &'static str, value: String) -> Div {
    column()
        .flex_1()
        .min_w(px(112.))
        .gap_0p5()
        .px_3()
        .py_2()
        .rounded(px(8.))
        .bg(color(theme::INSET))
        .child(div().text_base().font_weight(FontWeight::SEMIBOLD).child(value))
        .child(div().text_xs().text_color(color(theme::MUTED)).child(label))
}

/// The words of a settings row: what the control is, and what it means for the user.
pub fn setting(name: &'static str, meaning: &'static str) -> Div {
    column().flex_1().min_w_0().gap_0p5().child(text(name)).child(note(meaning))
}

/// A hairline between groups inside a card.
pub fn divider() -> Div {
    div().w_full().h(px(1.)).flex_shrink_0().bg(color(theme::BORDER))
}
