// A GUI-subsystem binary: started at logon as the background service, it must not
// open a console window. The CLI modes attach to the caller's console themselves.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app;
mod autostart;
mod cli;
mod config;
mod embed;
mod install;
mod paths;
mod service;

fn main() {
    // Seekgram Setup's install, update and uninstall callbacks exit here.
    install::handle_setup_events();
    std::process::exit(cli::run());
}
