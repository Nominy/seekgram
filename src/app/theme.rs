//! The panel's palette: dark and quiet, with one accent colour.
//!
//! gpui-component's controls read their colours from its global theme, so the palette
//! is written into that theme once at startup; the panel's own surfaces use the same
//! constants directly.

use gpui::{App, Hsla, rgb, rgba};
use gpui_component::{Theme, ThemeMode};

/// Telegram-adjacent blue; the only accent.
pub const ACCENT: u32 = 0x3a9fe0;
const ACCENT_HOVER: u32 = 0x52adea;
const ACCENT_ACTIVE: u32 = 0x2f8bc9;
/// Window background.
pub const SURFACE: u32 = 0x101215;
/// Card background, one step above the window.
pub const CARD: u32 = 0x171a1f;
/// Insets inside a card: code samples, stat tiles.
pub const INSET: u32 = 0x1f242b;
pub const BORDER: u32 = 0x2a2f37;
pub const TEXT: u32 = 0xf2f2f2;
pub const MUTED: u32 = 0xa0a0a0;
pub const DANGER: u32 = 0xf0686c;
pub const WARNING: u32 = 0xe2b34c;

pub fn color(hex: u32) -> Hsla {
    rgb(hex).into()
}

/// A translucent wash of the accent, for pills and selected surfaces.
pub fn tint(hex: u32) -> Hsla {
    rgba((hex << 8) | 0x2e).into()
}

/// Switches gpui-component to its dark theme and recolours it with the palette.
pub fn apply(cx: &mut App) {
    Theme::change(ThemeMode::Dark, None, cx);
    Theme::update(cx, |theme| {
        let colors = &mut theme.colors;
        colors.background = color(SURFACE);
        colors.foreground = color(TEXT);
        colors.muted = color(INSET);
        colors.muted_foreground = color(MUTED);
        colors.border = color(BORDER);
        colors.input = color(0x363c46);
        colors.caret = color(TEXT);
        colors.selection = rgba((ACCENT << 8) | 0x59).into();
        colors.ring = color(ACCENT);

        colors.primary = color(ACCENT);
        colors.primary_hover = color(ACCENT_HOVER);
        colors.primary_active = color(ACCENT_ACTIVE);
        colors.primary_foreground = color(0xffffff);
        colors.button_primary = color(ACCENT);
        colors.button_primary_hover = color(ACCENT_HOVER);
        colors.button_primary_active = color(ACCENT_ACTIVE);
        colors.button_primary_foreground = color(0xffffff);

        colors.secondary = color(INSET);
        colors.secondary_hover = color(0x283038);
        colors.secondary_active = color(0x2f3842);
        colors.secondary_foreground = color(TEXT);

        colors.link = color(ACCENT);
        colors.link_hover = color(ACCENT_HOVER);
        colors.link_active = color(ACCENT_ACTIVE);
        colors.progress_bar = color(ACCENT);
        colors.switch = color(0x3a414c);
        colors.switch_thumb = color(0xffffff);

        colors.danger = color(DANGER);
        colors.title_bar = color(SURFACE);
        colors.title_bar_border = color(BORDER);
    });
}
