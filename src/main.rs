// A GUI-subsystem binary: started at logon as the background service, it must not
// open a console window. The CLI modes attach to the caller's console themselves.
#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

mod cli;

fn main() {
    // Seekgram Setup's install, update and uninstall callbacks exit here.
    #[cfg(windows)]
    seekgram_service::install::handle_setup_events();
    std::process::exit(cli::run());
}
