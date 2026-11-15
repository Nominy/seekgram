<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/brand/header-dark.svg">
    <img src="assets/brand/header-light.svg" alt="Seekgram" width="420">
  </picture>
</p>

<p align="center">
  Find any photo, video or voice message in your Telegram account by describing it. Runs on your own PC.
</p>

<p align="center">
  <a href="https://github.com/Nominy/seekgram/releases/latest"><img src="https://img.shields.io/github/v/release/Nominy/seekgram?style=flat-square&color=8fd3fb&labelColor=0f2740" alt="Latest release"></a>
  <a href="https://github.com/Nominy/seekgram/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/Nominy/seekgram/ci.yml?branch=main&style=flat-square&labelColor=0f2740" alt="CI"></a>
  <img src="https://img.shields.io/badge/platform-windows%2010%20%2F%2011-3a9fe0?style=flat-square&labelColor=0f2740" alt="Windows 10 or 11">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-3a9fe0?style=flat-square&labelColor=0f2740" alt="MIT license"></a>
</p>

<p align="center">
  <a href="https://nominy.github.io/seekgram">Website</a> ·
  <a href="#install">Install</a> ·
  <a href="#usage">Usage</a> ·
  <a href="#privacy-and-security">Privacy</a> ·
  <a href="https://github.com/Nominy/seekgram/releases">Releases</a>
</p>

Somewhere in your Telegram there is that one picture. Send `?pink dragon` to your
Saved Messages and Seekgram sends it back, with a link to the message it came
from. A model running on your own graphics card has looked at every photo, video
and audio file in your account, so you search by what is *in* them, not by
captions or file names. Nothing is uploaded anywhere.

## Features

- **Search by meaning.** Describe a picture in a few words. `?video` and
  `?audio` search videos and audio the same way.
- **Search by picture.** Send or reply to a photo with `?` to find similar ones,
  and add words to steer it: this, but *in snow*.
- **Search from Telegram itself.** Any of your devices, no extra app on your
  phone. Results arrive ten at a time as an album; each number under it opens the
  original message.
- **Runs on your PC.** A local model on your graphics card (or, slowly, on the
  processor) does all the looking. There is no Seekgram server and no account.
- **Stores vectors, not media.** Files are downloaded temporarily, embedded and
  deleted. The index holds only vectors and message IDs.
- **Stays out of the way.** A background service starts with Windows, indexes
  when it can, pauses when Telegram asks it to, and resumes where it stopped. A
  small control panel shows what it is doing.

## Install

**[Download Seekgram Setup](https://github.com/Nominy/seekgram/releases/latest/download/Seekgram.App-win-Setup.exe)**
and run it. It installs Seekgram for your Windows user (no admin) with a Start
menu and desktop shortcut, an entry in *Apps & features* and a start at Windows
logon, then opens Seekgram.

Seekgram isn't code-signed yet, so the first run needs one extra click: Windows
SmartScreen says *Windows protected your PC*: click **More info → Run anyway**.

**Or from a terminal:** open **PowerShell** (press <kbd>Win</kbd>, type
*PowerShell*, press <kbd>Enter</kbd>), paste this line and press <kbd>Enter</kbd>:

```powershell
irm https://nominy.github.io/seekgram/install.ps1 | iex
```

The script downloads the same Setup, verifies its checksum and runs it. It also
removes a copy installed by its earlier versions (in
`%LOCALAPPDATA%\Programs\Seekgram`), keeping the login, index and model.

The first start downloads the search model and its tools, **about 1.6 GB**, once.
The window shows the progress, and you can log in to Telegram in the meantime:

1. Get a free API key at [my.telegram.org](https://my.telegram.org) (*API
   development tools*; any app name will do) and enter its `api_id` and
   `api_hash`.
2. Enter your phone number, then the login code Telegram sends you, then your
   two-step verification password if you have one.

Indexing starts by itself as soon as the login and the downloads are done.

**Updates install themselves.** Two minutes after it starts, and then every six
hours, Seekgram asks GitHub for a new release. A new version is downloaded in the
background, then Seekgram closes its window, stops the service cleanly and
restarts on the new version. Your login, index and model are untouched. Switch
this off in *Settings → Install updates automatically*.

<details>
<summary>Script options and release files</summary>

Set environment variables before the one-liner, or pass the matching parameters
when running the script as a file:

| Environment variable | Parameter | |
|--|--|--|
| `$env:SEEKGRAM_VERSION = 'v0.1.0'` | `-Version v0.1.0` | install a specific release |
| `$env:SEEKGRAM_NO_LAUNCH = '1'` | `-NoLaunch` | don't open the app when done |
| | `-Setup <file>` | run a Setup you already downloaded |
| | `-Uninstall` | remove Seekgram, keep its data |
| | `-Uninstall -Purge` | remove Seekgram and delete its data |

To pass parameters without saving the script:

```powershell
& ([scriptblock]::Create((irm https://nominy.github.io/seekgram/install.ps1))) -Uninstall
```

Every [release](https://github.com/Nominy/seekgram/releases) has
`Seekgram.App-win-Setup.exe`, `SHA256SUMS`, `install.ps1` and the update
packages Seekgram downloads itself (`*.nupkg`, `releases.win.json`). The program
goes to `%LOCALAPPDATA%\Seekgram.App\current`; the data stays separately in
`%LOCALAPPDATA%\Seekgram`.
</details>

## Usage

Open **Seekgram** from the Start menu to see the control panel: the login, the
download and indexing progress, how much is indexed, and the settings. Closing
the window does not stop anything; the service keeps running in the background.

Searching happens in Telegram. Every search is a message starting with `?`, sent
in your **search chat**. That is Saved Messages to begin with; the app can
create a private group called *Search* for it instead, so results do not fill up
your Saved Messages.

| You send | You get |
|--|--|
| `?pink dragon` | pictures matching the words |
| `?video cat on a skateboard` | videos |
| `?audio rain` | voice messages and audio |
| a photo with the caption `?` | pictures similar to it (an album mixes its photos into one search) |
| `?` in reply to a photo | more like that photo; works on an earlier result too |
| `?in snow` as a caption or a reply | the photo plus the words |
| `?more` | the next 10 results |
| `?7` | result number 7 itself, forwarded, so you can jump to its chat |
| `?here` | sent in a group where you are the only member: search moves to that group (send it in Saved Messages to move back) |
| `?` | the help text |

Results come as one album of the photos and videos themselves. The numbers under
it open each original message; in some Telegram apps the links to private chats
do nothing, and `?7` gets you there instead. Things that cannot be part of an
album (GIFs, video stickers, audio) are forwarded. Only chats that are already
indexed are searched.

From a terminal (the program is
`%LOCALAPPDATA%\Seekgram.App\current\seekgram.exe`; it is not on PATH):

```powershell
$seekgram = "$env:LOCALAPPDATA\Seekgram.App\current\seekgram.exe"
& $seekgram            # open the control panel (starts the service if needed)
& $seekgram status     # print what the service is doing, as JSON
& $seekgram stop       # stop the background service
& $seekgram start      # start it again
```

## Requirements

- **Windows 10 or 11**, 64-bit. There is no macOS or Linux version.
- **A DirectX 12 graphics card** (NVIDIA, AMD or Intel) is recommended. Without
  one the model runs on the processor, which works but is many times slower.
- **About 6 GB of free disk space** for the model, its tools and the index.
- **A Telegram account** and a free API key from
  [my.telegram.org](https://my.telegram.org).

What to expect:

- **The first index of a large account takes days**, a small one hours. It runs
  in the background and survives restarts, and you can search what is already
  indexed while it works. Indexing can be switched off in the app when you need
  the graphics card for something else.
- **Edits and deletions in Telegram are not yet synced to the index.** New media
  is picked up; a message you deleted can still appear as a result that no
  longer opens.

## Privacy and security

**Seekgram logs in as your full Telegram account**, the same way Telegram
Desktop does. It can see everything you can. Only install builds you trust: the
releases on this page, or one you built yourself from this source. You can end
its session at any time in Telegram under **Settings → Devices**.

What it does with that access:

- **Everything runs on your PC.** There is no Seekgram server, no account and no
  telemetry.
- **Media is downloaded temporarily, embedded and deleted.** Only vectors and
  message IDs are stored. Message text and captions are not stored or logged. A
  size-capped cache holds the media a search needs to answer; it is safe to
  delete.
- **Network traffic** goes to Telegram, and once to GitHub, NuGet and Hugging
  Face to download ffmpeg, ONNX Runtime with DirectML, and the model. Every six
  hours it asks GitHub's release list for a new version (unless you switch
  automatic updates off). Nothing else.
- **It reads every chat, but writes to one.** It scans all your chats for media.
  It sends only search results, and only into your search chat: Saved Messages,
  or a group whose only member is you. That is checked when the group is chosen,
  not afterwards: if you add someone to the group later, they see your results.
- **Secrets are encrypted.** The Telegram login session and your API key are
  stored encrypted for your Windows user (DPAPI), in `%LOCALAPPDATA%\Seekgram`.
  Other accounts on the PC and copies of the files on another machine cannot use
  them; programs running as you can.

Seekgram is not affiliated with Telegram.

## How it works

One program, `seekgram.exe`, in two roles:

- **The service** (`seekgram service`) runs in the background from Windows
  logon. It holds the Telegram connection, goes through your chats, downloads
  each media file, hands it to the model, stores the resulting vector in a local
  SQLite index and answers searches in your search chat.
- **The control panel** (`seekgram`) is a small native window that shows the
  service's status and settings. It is only a view; closing it stops nothing.
- **The local model** is Google's EmbeddingGemma 2, run through ONNX Runtime and
  DirectML on your graphics card, in a worker process of `seekgram.exe`. It turns
  pictures, video frames, audio and your search words into vectors in the same
  space, so a search is a nearest-neighbour lookup. Seekgram downloads ffmpeg,
  the runtime and the model by itself on first start; nothing else needs to be
  installed.

Everything it owns is in one folder, `%LOCALAPPDATA%\Seekgram`. The details are
in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Uninstall

**Settings → Apps → Seekgram → Uninstall**, or:

```powershell
& ([scriptblock]::Create((irm https://nominy.github.io/seekgram/install.ps1))) -Uninstall
```

This stops Seekgram and removes the program, its shortcuts and the start at logon.
Your Telegram login, the index and the downloaded model stay in
`%LOCALAPPDATA%\Seekgram`, so a reinstall continues where you left off. To delete
those too, add `-Purge`:

```powershell
& ([scriptblock]::Create((irm https://nominy.github.io/seekgram/install.ps1))) -Uninstall -Purge
```

After a purge, also remove the Seekgram session in Telegram under
**Settings → Devices**.

## For AI agents (optional)

Seekgram can also offer your index to AI agents through an
[MCP](https://modelcontextprotocol.io) endpoint. It is off by default and you do
not need it for anything above. To use it, switch it on in the app under
**Settings → Advanced**, then give your agent:

- the URL `http://127.0.0.1:28741/mcp` (Streamable HTTP; reachable only from
  this PC),
- the bearer token, copied from the same place in the app, sent as
  `Authorization: Bearer <token>`.

Anything holding that token can search your media and read your chats through
the service, so treat it like a password.

## Building from source

You need [Rust](https://rustup.rs) on Windows (MSVC toolchain).

```powershell
git clone https://github.com/Nominy/seekgram
cd seekgram
cargo run --release
```

Set `SEEKGRAM_DATA_DIR` to run a second, isolated instance next to an installed
one while developing. A build run from `target\` does not update itself; only a
copy installed with Seekgram Setup does. To try updates against a local folder of
`vpk pack` output, set `SEEKGRAM_UPDATE_URL` to that folder before starting the
service.

<details>
<summary>Project layout and releasing</summary>

| | |
|--|--|
| `src/service/` | the background service: Telegram, indexer, model workers, search, control API |
| `src/app/` | the control panel ([GPUI](https://gpui.rs)) |
| `src/embed/` | the model worker (`seekgram embed-worker`): preprocessing and ONNX Runtime |
| `assets/model/`, `assets/selftest/` | the model graphs and the self-test files, embedded in the binary |
| `src/install.rs`, `src/service/update.rs` | Setup's install, update and uninstall steps; the automatic updates ([Velopack](https://velopack.io)) |
| `install.ps1` | the PowerShell install script |
| `assets/brand/generate.py` | regenerates the logo, header pictures and icons |
| `site/` | the website at nominy.github.io/seekgram |
| `docs/ARCHITECTURE.md` | how the parts fit together |

To release: bump `version` in `Cargo.toml`, commit, then
`git tag vX.Y.Z && git push --tags`. The *Release* workflow builds Windows x64,
packs it with Velopack's `vpk` (with a delta from the previous release) and
publishes the release with `Seekgram.App-win-Setup.exe`, the update packages,
`SHA256SUMS` and `install.ps1`. Installed copies pick it up within six hours. The
*Pages* workflow publishes `site/` and the installer script.
</details>

## Acknowledgements

Built on [GPUI](https://gpui.rs), [grammers](https://codeberg.org/Lonami/grammers)
and [EmbeddingGemma](https://huggingface.co/google/embeddinggemma-2).
Wordmark set in [Sora](https://fonts.google.com/specimen/Sora) (OFL).

## License

[MIT](LICENSE)
