//! Icons compiled into the binary, so the panel is a single file.
//!
//! gpui-component looks its own icons up by path (`icons/loader.svg` for spinners,
//! `icons/window-*.svg` for the title bar buttons); those names are fixed by it.

use std::borrow::Cow;

use anyhow::Result;
use gpui::{AssetSource, SharedString};

macro_rules! icons {
    ($($name:literal),* $(,)?) => {
        &[$((concat!("icons/", $name, ".svg"), include_bytes!(concat!("../../../assets/icons/", $name, ".svg")))),*]
    };
}

const ICONS: &[(&str, &[u8])] = icons![
    "check",
    "chevron-down",
    "chevron-right",
    "copy",
    "external-link",
    "folder",
    "loader",
    "mark",
    "minus",
    "plus",
    "window-close",
    "window-maximize",
    "window-minimize",
    "window-restore",
];

pub struct Assets;

impl AssetSource for Assets {
    fn load(&self, path: &str) -> Result<Option<Cow<'static, [u8]>>> {
        Ok(ICONS.iter().find(|(name, _)| *name == path).map(|(_, bytes)| Cow::Borrowed(*bytes)))
    }

    fn list(&self, path: &str) -> Result<Vec<SharedString>> {
        Ok(ICONS.iter().filter(|(name, _)| name.starts_with(path)).map(|(name, _)| SharedString::from(*name)).collect())
    }
}
