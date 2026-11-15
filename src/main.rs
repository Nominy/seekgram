// A GUI-subsystem binary: started at logon as the background service, it must not
// open a console window. The CLI modes attach to the caller's console themselves.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app;
mod autostart;
mod cli;
mod config;
mod paths;
mod service;

fn main() {
    std::process::exit(cli::run());
}
