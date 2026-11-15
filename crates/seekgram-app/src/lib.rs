//! The control panel: a window over the background service's control API.
//!
//! The panel owns no data and does no work of its own. It starts the service when
//! it is not running, shows what `GET /api/status` reports, and turns clicks into
//! API calls. Closing the window quits only the panel; the service keeps running.
//!
//! - `client`: the typed API client (blocking; only ever called off the UI thread).
//! - `panel`: state, polling, actions and the full-window messages.
//! - `wizard`, `dashboard`: the two screens, one small function per step or card.
//! - `widgets`, `theme`, `format`, `assets`: shared looks, colours, wording of numbers, icons.

mod assets;
mod client;
mod dashboard;
mod format;
mod panel;
mod theme;
mod widgets;
mod wizard;

use gpui::{App, AppContext as _, Bounds, TitlebarOptions, WindowBounds, WindowOptions, px, size};
use gpui_component::{Root, TitleBar};

/// Opens the panel and returns when the user closes it.
pub fn run() {
    gpui_platform::application().with_assets(assets::Assets).run(|cx: &mut App| {
        gpui_component::init(cx);
        theme::apply(cx);
        cx.on_window_closed(|cx, _| {
            if cx.windows().is_empty() {
                cx.quit();
            }
        })
        .detach();

        let options = WindowOptions {
            window_bounds: Some(WindowBounds::Windowed(Bounds::centered(None, size(px(900.), px(680.)), cx))),
            titlebar: Some(TitlebarOptions { title: Some("Seekgram".into()), ..TitleBar::title_bar_options() }),
            window_min_size: Some(size(px(720.), px(520.))),
            app_id: Some("seekgram".into()),
            ..TitleBar::window_options()
        };
        let opened = cx.open_window(options, |window, cx| {
            let panel = cx.new(|cx| panel::Panel::new(window, cx));
            cx.new(|cx| Root::new(panel, window, cx))
        });
        match opened {
            Ok(_) => cx.activate(true),
            Err(error) => {
                eprintln!("Seekgram could not open its window: {error}");
                cx.quit();
            }
        }
    });
}
