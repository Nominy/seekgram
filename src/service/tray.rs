//! The notification-area icon: the visible sign that Seekgram is running in the
//! background, and the quickest way to open the panel, pause indexing or quit.
//!
//! It belongs to the service, not the panel, so it is there from logon whether or not
//! a window was ever opened. Windows requires the icon's owner to pump messages, so it
//! lives on its own thread.

use std::{sync::Arc, time::Duration};

use tray_icon::{
    menu::{Menu, MenuEvent, MenuItem, PredefinedMenuItem},
    Icon, MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent,
};

use super::state::{AccountStep, IndexerState, SetupState, Shared, Status, UpdateState};

const OPEN: &str = "open";
const INDEXING: &str = "indexing";
const QUIT: &str = "quit";

/// Opens the control panel: this same program without arguments.
fn open_panel() {
    match std::env::current_exe() {
        Ok(exe) => { if let Err(error) = std::process::Command::new(exe).spawn() { eprintln!("Tray: the panel could not be opened: {error}"); } }
        Err(error) => eprintln!("Tray: the program file could not be found: {error}"),
    }
}

/// The program file a process runs, if it can be asked.
#[cfg(windows)]
fn process_image(pid: u32) -> Option<std::path::PathBuf> {
    use windows::Win32::{Foundation::CloseHandle, System::Threading::{OpenProcess, QueryFullProcessImageNameW, PROCESS_NAME_WIN32, PROCESS_QUERY_LIMITED_INFORMATION}};
    // SAFETY: plain Win32 calls; the buffer outlives the call and the handle is closed.
    unsafe {
        let process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid).ok()?;
        let mut buffer = [0u16; 1024];
        let mut length = buffer.len() as u32;
        let named = QueryFullProcessImageNameW(process, PROCESS_NAME_WIN32, windows::core::PWSTR(buffer.as_mut_ptr()), &mut length).is_ok();
        let _ = CloseHandle(process);
        named.then(|| std::path::PathBuf::from(String::from_utf16_lossy(&buffer[..length as usize])))
    }
}

/// "Quit Seekgram" means all of it: open control panels are closed too, not left showing
/// a stopped service. So does installing an update. A panel is any visible window of
/// another process running this same program; the service has no visible window.
#[cfg(windows)]
pub fn close_panels() {
    use windows::{core::BOOL, Win32::{Foundation::{HWND, LPARAM, WPARAM}, UI::WindowsAndMessaging::{EnumWindows, GetWindowThreadProcessId, IsWindowVisible, PostMessageW, WM_CLOSE}}};
    unsafe extern "system" fn visit(window: HWND, _: LPARAM) -> BOOL {
        // SAFETY: called by EnumWindows with a valid window handle.
        unsafe {
            if !IsWindowVisible(window).as_bool() { return BOOL(1); }
            let mut pid = 0;
            GetWindowThreadProcessId(window, Some(&mut pid));
            if pid == 0 || pid == std::process::id() { return BOOL(1); }
            let ours = process_image(pid).is_some_and(|image| std::env::current_exe().is_ok_and(|exe| image == exe));
            if ours { let _ = PostMessageW(Some(window), WM_CLOSE, WPARAM(0), LPARAM(0)); }
            BOOL(1)
        }
    }
    // SAFETY: EnumWindows only calls `visit` during this call.
    let _ = unsafe { EnumWindows(Some(visit), LPARAM(0)) };
}

#[cfg(not(windows))]
pub fn close_panels() {}

/// What Seekgram is doing, shown when the pointer rests on the icon: one line, and a
/// second while an update is being checked for, downloaded or installed.
fn summary(status: &Status) -> String {
    let doing = if status.account.step != AccountStep::Ready {
        "waiting for you to log in".to_owned()
    } else if status.setup.state == SetupState::Failed {
        "setup needs attention".to_owned()
    } else if status.setup.state != SetupState::Ready {
        "setting up the search model".to_owned()
    } else {
        match status.indexer.state {
            IndexerState::Running | IndexerState::Starting => format!("indexing, chat {} of {}", status.indexer.chats_done + 1, status.indexer.chats_total.max(1)),
            IndexerState::Idle => "up to date".to_owned(),
            IndexerState::FloodWait => "paused by Telegram for a moment".to_owned(),
            IndexerState::Failed => "indexing needs attention".to_owned(),
            IndexerState::Off | IndexerState::Stopping => "indexing paused".to_owned(),
        }
    };
    let update = status.update.version.as_deref().unwrap_or("");
    let updating = match status.update.state {
        UpdateState::Idle => String::new(),
        UpdateState::Checking => "\nChecking for a new version".to_owned(),
        UpdateState::Downloading => format!("\nDownloading version {update}"),
        UpdateState::Ready => format!("\nInstalling version {update}"),
        UpdateState::Failed => "\nThe last update attempt failed".to_owned(),
    };
    format!("Seekgram: {doing}{updating}")
}

fn indexing_label(on: bool) -> &'static str { if on { "Pause indexing" } else { "Resume indexing" } }

#[cfg(windows)]
fn run(shared: Arc<Shared>) -> Result<(), String> {
    use windows::Win32::UI::WindowsAndMessaging::{DispatchMessageW, PeekMessageW, TranslateMessage, MSG, PM_REMOVE};

    let open = MenuItem::with_id(OPEN, "Open Seekgram", true, None);
    let indexing = MenuItem::with_id(INDEXING, indexing_label(shared.settings().indexing), true, None);
    let quit = MenuItem::with_id(QUIT, "Quit Seekgram", true, None);
    let menu = Menu::new();
    menu.append_items(&[&open, &indexing, &PredefinedMenuItem::separator(), &quit]).map_err(|e| format!("the tray menu could not be built: {e}"))?;
    // Resource 1 is the application icon embedded by build.rs.
    let icon = Icon::from_resource(1, None).map_err(|e| format!("the tray icon could not be loaded: {e}"))?;
    let tray = TrayIconBuilder::new()
        .with_icon(icon)
        .with_menu(Box::new(menu))
        .with_menu_on_left_click(false)
        .with_tooltip(summary(&shared.status()))
        .build()
        .map_err(|e| format!("the tray icon could not be shown: {e}"))?;

    let actions = shared.clone();
    MenuEvent::set_event_handler(Some(move |event: MenuEvent| match event.id.as_ref() {
        OPEN => open_panel(),
        INDEXING => {
            if let Err(error) = actions.set_settings(|settings| settings.indexing = !settings.indexing) { eprintln!("Tray: {error}"); }
        }
        QUIT => {
            close_panels();
            actions.shutdown.send_replace(true);
        }
        _ => {}
    }));
    TrayIconEvent::set_event_handler(Some(|event: TrayIconEvent| {
        if matches!(event, TrayIconEvent::Click { button: MouseButton::Left, button_state: MouseButtonState::Up, .. }) { open_panel(); }
    }));

    let mut shown = (String::new(), true);
    while !shared.stopping() {
        // SAFETY: a standard message pump for this thread's own windows.
        unsafe {
            let mut message = MSG::default();
            while PeekMessageW(&mut message, None, 0, 0, PM_REMOVE).as_bool() {
                let _ = TranslateMessage(&message);
                DispatchMessageW(&message);
            }
        }
        let status = shared.status();
        let now = (summary(&status), status.settings.indexing);
        if now != shown {
            let _ = tray.set_tooltip(Some(&now.0));
            indexing.set_text(indexing_label(now.1));
            shown = now;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    // Dropping the icon removes it at once instead of leaving a ghost until the pointer passes over it.
    drop(tray);
    Ok(())
}

#[cfg(not(windows))]
fn run(_: Arc<Shared>) -> Result<(), String> { Ok(()) }

/// Shows the icon until shutdown. A tray that cannot be shown is logged, not fatal:
/// the panel and the service work without it.
pub fn spawn(shared: Arc<Shared>) {
    let spawned = std::thread::Builder::new().name("tray".into()).spawn(move || {
        if let Err(error) = run(shared) { eprintln!("Tray: {error}"); }
    });
    if let Err(error) = spawned { eprintln!("Tray: the icon thread could not start: {error}"); }
}
