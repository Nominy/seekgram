fn main() {
    println!("cargo:rerun-if-changed=assets/windows/seekgram.rc");
    println!("cargo:rerun-if-changed=assets/app/seekgram.ico");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        // The version resource takes its numbers from Cargo.toml so the two cannot disagree.
        let part = |name: &str| std::env::var(format!("CARGO_PKG_VERSION_{name}")).unwrap();
        let macros = [
            format!("SEEKGRAM_MAJOR={}", part("MAJOR")),
            format!("SEEKGRAM_MINOR={}", part("MINOR")),
            format!("SEEKGRAM_PATCH={}", part("PATCH")),
        ];
        // Icon resource id 1 is what gpui loads for the window class.
        embed_resource::compile("assets/windows/seekgram.rc", &macros)
            .manifest_optional()
            .unwrap();
    }
}
