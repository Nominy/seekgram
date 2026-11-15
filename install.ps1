<#
.SYNOPSIS
    Seekgram installer for Windows.

.DESCRIPTION
    irm https://nominy.github.io/seekgram/install.ps1 | iex

    Downloads the latest release, checks its SHA-256 and installs it for your
    user (no admin): Start menu, PATH, "Apps & features", and a start at Windows
    logon so indexing and search keep working in the background. Then it opens
    Seekgram, where you log in to Telegram.

    The installer itself is small. On its first start Seekgram downloads the
    search model and its tools (about 5 GB) and shows the progress.

    When piped into iex, configure with environment variables:
      $env:SEEKGRAM_VERSION = 'v0.1.0'     # default: latest
      $env:SEEKGRAM_NO_AUTOSTART = '1'     # don't start at Windows logon
      $env:SEEKGRAM_NO_LAUNCH = '1'        # don't open the app when done

.EXAMPLE
    irm https://nominy.github.io/seekgram/install.ps1 | iex
.EXAMPLE
    & ([scriptblock]::Create((irm https://nominy.github.io/seekgram/install.ps1))) -Uninstall
.EXAMPLE
    & ([scriptblock]::Create((irm https://nominy.github.io/seekgram/install.ps1))) -Uninstall -Purge
.EXAMPLE
    .\install.ps1 -FromSource
#>
param(
    [string] $Version = $(if ($env:SEEKGRAM_VERSION) { $env:SEEKGRAM_VERSION } else { 'latest' }),
    # Install a release zip you already have.
    [string] $Archive,
    # Build this checkout (or a fresh clone) with cargo instead of downloading.
    [switch] $FromSource,
    [switch] $NoAutostart = [bool]$env:SEEKGRAM_NO_AUTOSTART,
    # Install without opening the app afterwards (unattended installs).
    [switch] $NoLaunch = [bool]$env:SEEKGRAM_NO_LAUNCH,
    [switch] $Uninstall,
    # With -Uninstall: also delete the Telegram login, the index and the model.
    [switch] $Purge
)

function Install-Seekgram {
    $ErrorActionPreference = 'Stop'
    $ProgressPreference = 'SilentlyContinue'
    Set-StrictMode -Version Latest
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

    $AppName = 'Seekgram'
    $ExeName = 'seekgram.exe'
    $Repo = if ($env:SEEKGRAM_REPO) { $env:SEEKGRAM_REPO } else { 'Nominy/seekgram' }
    $Asset = 'seekgram-windows-x64.zip'

    $hive = [Microsoft.Win32.Registry]::CurrentUser
    $installDir = Join-Path $env:LOCALAPPDATA "Programs\$AppName"
    # Where the app keeps the Telegram login, the index and the model (src/paths.rs).
    $dataDir = Join-Path $env:LOCALAPPDATA $AppName
    $startMenu = Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs'
    $shortcut = Join-Path $startMenu "$AppName.lnk"
    $exe = Join-Path $installDir $ExeName
    $uninstallKey = "Software\Microsoft\Windows\CurrentVersion\Uninstall\$AppName"
    $runKey = 'Software\Microsoft\Windows\CurrentVersion\Run'
    $base = if ($env:SEEKGRAM_DOWNLOAD_URL) { $env:SEEKGRAM_DOWNLOAD_URL.TrimEnd('/') }
            elseif ($Version -eq 'latest') { "https://github.com/$Repo/releases/latest/download" }
            else { "https://github.com/$Repo/releases/download/$Version" }

    # ASCII only: Windows PowerShell 5.1 decodes piped and BOM-less scripts as ANSI.
    function Say([string] $text) { Write-Host '> ' -ForegroundColor Cyan -NoNewline; Write-Host $text }
    function Note([string] $text) { Write-Host "  $text" -ForegroundColor DarkGray }
    function Warn([string] $text) { Write-Host '! ' -ForegroundColor Yellow -NoNewline; Write-Host $text -ForegroundColor Yellow }

    function Set-Value([string] $path, [string] $name, $value, $kind = [Microsoft.Win32.RegistryValueKind]::String) {
        $key = $hive.CreateSubKey($path)
        try { $key.SetValue($name, $value, $kind) } finally { $key.Dispose() }
    }
    function Test-Value([string] $path, [string] $name) {
        $key = $hive.OpenSubKey($path)
        if (-not $key) { return $false }
        try { return $null -ne $key.GetValue($name) } finally { $key.Dispose() }
    }
    function Remove-Value([string] $path, [string] $name) {
        $key = $hive.OpenSubKey($path, $true)
        if ($key) { try { $key.DeleteValue($name, $false) } finally { $key.Dispose() } }
    }

    # Processes of this install only, so a development copy elsewhere is left alone.
    function Get-Running {
        @(Get-CimInstance Win32_Process -Filter "Name = '$ExeName'" -ErrorAction SilentlyContinue |
            Where-Object { $_.ExecutablePath -and ($_.ExecutablePath -ieq $exe) })
    }
    # The panel is only a view and can simply be closed. The service may be in the
    # middle of writing to the index, so it is asked to finish first and only
    # stopped the hard way if it has not exited after 30 seconds.
    function Stop-Running {
        $running = Get-Running
        if (-not $running) { return }
        $running | Where-Object { "$($_.CommandLine)" -notmatch '\sservice\s*$' } |
            ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
        if (-not (Get-Running)) { return }
        Say "Stopping the running $AppName service"
        $deadline = (Get-Date).AddSeconds(30)
        try {
            $stop = Start-Process -FilePath $exe -ArgumentList 'stop' -WindowStyle Hidden -PassThru
            $stop.WaitForExit(30000) | Out-Null
        } catch { Note "Couldn't ask it to stop ($($_.Exception.Message))." }
        while ((Get-Running) -and ((Get-Date) -lt $deadline)) { Start-Sleep -Milliseconds 500 }
        $left = Get-Running
        if ($left) {
            Note 'It did not stop in 30 seconds; ending the process.'
            $left | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
            Start-Sleep -Seconds 1
        }
    }
    function Set-PathEntry([bool] $present) {
        $current = [Environment]::GetEnvironmentVariable('Path', 'User')
        $parts = @(if ($current) { $current.Split(';') | Where-Object { $_ -and ($_.TrimEnd('\') -ine $installDir) } })
        if ($present) { $parts += $installDir }
        [Environment]::SetEnvironmentVariable('Path', ($parts -join ';'), 'User')
    }
    function Register-Uninstaller([string] $version) {
        $script = Join-Path $installDir 'install.ps1'
        Set-Value $uninstallKey 'DisplayName' $AppName
        Set-Value $uninstallKey 'DisplayVersion' $version
        Set-Value $uninstallKey 'DisplayIcon' "`"$exe`",0"
        Set-Value $uninstallKey 'Publisher' 'Nominy'
        Set-Value $uninstallKey 'URLInfoAbout' "https://github.com/$Repo"
        Set-Value $uninstallKey 'InstallLocation' $installDir
        Set-Value $uninstallKey 'UninstallString' "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$script`" -Uninstall"
        Set-Value $uninstallKey 'NoModify' 1 ([Microsoft.Win32.RegistryValueKind]::DWord)
        Set-Value $uninstallKey 'NoRepair' 1 ([Microsoft.Win32.RegistryValueKind]::DWord)
    }

    # ----------------------------------------------------------- uninstall

    if ($Uninstall) {
        Stop-Running
        Remove-Value $runKey $AppName
        $hive.DeleteSubKeyTree($uninstallKey, $false)
        Set-PathEntry $false
        if (Test-Path $shortcut) { Remove-Item -Force $shortcut }
        if (Test-Path $installDir) { Remove-Item -Recurse -Force $installDir }
        Say "$AppName removed."
        if ($Purge) {
            if (Test-Path $dataDir) { Remove-Item -Recurse -Force $dataDir }
            Note "Deleted $dataDir (Telegram login, index and model)."
            Note 'To also end the login on Telegram''s side: Telegram > Settings > Devices.'
        } elseif (Test-Path $dataDir) {
            Note "Your Telegram login, index and the downloaded model are kept in $dataDir"
            Note 'Delete that folder, or uninstall with -Purge, to remove them too.'
        }
        return
    }
    if ($Purge) { throw '-Purge only goes together with -Uninstall.' }

    # -------------------------------------------------------------- checks

    if (-not [Environment]::Is64BitOperatingSystem) { throw "$AppName needs 64-bit Windows." }
    $gpu = $null
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        try {
            $ErrorActionPreference = 'Continue'
            $gpu = (& nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1)
            if ($LASTEXITCODE -ne 0) { $gpu = $null }
        } catch { $gpu = $null } finally { $ErrorActionPreference = 'Stop' }
    }
    if ($gpu) {
        Note "Graphics card: $("$gpu".Trim())"
    } else {
        Warn 'No NVIDIA graphics card was found (nvidia-smi is missing or reports none).'
        Warn "$AppName needs one with 4 GB of memory or more and a current NVIDIA driver to index and search."
        Warn 'Installing anyway; the app will tell you again. Drivers: https://www.nvidia.com/drivers'
    }

    # ------------------------------------------------------------- payload

    $tmp = Join-Path ([IO.Path]::GetTempPath()) ("seekgram-" + [Guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $tmp | Out-Null
    try {
        $here = if ($ScriptPath) { Split-Path -Parent $ScriptPath } else { $null }
        if ($FromSource) {
            if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
                throw 'Building needs Rust:  winget install Rustlang.Rustup  (then reopen the terminal)'
            }
            $src = if ($here -and (Test-Path (Join-Path $here 'Cargo.toml'))) { $here } else {
                if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw 'Building from source needs git.' }
                $ref = if ($Version -eq 'latest') { 'main' } else { $Version }
                Say "Cloning $Repo ($ref)"
                & git clone --quiet --depth 1 --branch $ref "https://github.com/$Repo.git" (Join-Path $tmp 'src')
                if ($LASTEXITCODE -ne 0) { throw 'git clone failed.' }
                Join-Path $tmp 'src'
            }
            Say 'Building from source (the first build takes several minutes)'
            Push-Location $src
            try {
                & cargo build --release --locked | Out-Host
                if ($LASTEXITCODE -ne 0) { throw "cargo build failed with exit code $LASTEXITCODE" }
            } finally { Pop-Location }
            $binary = Join-Path $src "target\release\$ExeName"
        } else {
            if (-not $Archive) {
                Say "Downloading $AppName ($Version)"
                $Archive = Join-Path $tmp $Asset
                try { Invoke-WebRequest -UseBasicParsing -Uri "$base/$Asset" -OutFile $Archive }
                catch { throw "Couldn't download $base/$Asset ($($_.Exception.Message))." }
                $sums = Join-Path $tmp 'SHA256SUMS'
                Invoke-WebRequest -UseBasicParsing -Uri "$base/SHA256SUMS" -OutFile $sums
                $line = Get-Content $sums | Where-Object { $_ -match "\s\*?$([regex]::Escape($Asset))$" } | Select-Object -First 1
                if (-not $line) { throw "$Asset is not listed in SHA256SUMS." }
                $expected = ($line -split '\s+')[0].ToLowerInvariant()
                $actual = (Get-FileHash -Algorithm SHA256 $Archive).Hash.ToLowerInvariant()
                if ($expected -ne $actual) { throw "Checksum mismatch for $Asset (expected $expected, got $actual)." }
                Note 'checksum ok'
            } elseif (-not (Test-Path $Archive)) {
                throw "There is no file at $Archive."
            }
            $unpacked = Join-Path $tmp 'unpacked'
            Expand-Archive -Path $Archive -DestinationPath $unpacked -Force
            $binary = Join-Path $unpacked $ExeName
            if (-not (Test-Path $binary)) { throw "The archive doesn't contain $ExeName." }
        }

        # ------------------------------------------------------- install

        $upgrade = Test-Path $exe
        Stop-Running
        New-Item -ItemType Directory -Force -Path $installDir | Out-Null
        Copy-Item -Force $binary $exe
        # Keep a copy of this script for "Apps & features" -> Uninstall.
        $selfCopy = Join-Path $installDir 'install.ps1'
        if ($ScriptPath -and ($ScriptPath -ine $selfCopy)) {
            Copy-Item -Force $ScriptPath $selfCopy
        } elseif (-not $ScriptPath) {
            try { Invoke-WebRequest -UseBasicParsing -Uri "$base/install.ps1" -OutFile $selfCopy }
            catch { Note "Couldn't fetch the uninstaller; uninstall later with the one-liner and -Uninstall." }
        }
        # The version resource is in the file itself, so nothing has to be run to read it.
        $installedVersion = "$((Get-Item $exe).VersionInfo.ProductVersion)".Trim()
        if (-not $installedVersion) { $installedVersion = $Version.TrimStart('v') }
        Register-Uninstaller $installedVersion
        Set-PathEntry $true

        New-Item -ItemType Directory -Force -Path $startMenu | Out-Null
        $shell = New-Object -ComObject WScript.Shell
        $link = $shell.CreateShortcut($shortcut)
        $link.TargetPath = $exe
        $link.WorkingDirectory = $installDir
        $link.IconLocation = "$exe,0"
        $link.Description = 'Search your Telegram media'
        $link.Save()

        # Start at logon. On an upgrade the existing choice wins, so switching
        # "start with Windows" off in the app is not undone by updating.
        $autostart = "`"$exe`" service"
        if ($NoAutostart) {
            Remove-Value $runKey $AppName
            Note 'Not starting at Windows logon; indexing and search only work while Seekgram is running.'
        } elseif ((-not $upgrade) -or (Test-Value $runKey $AppName)) {
            Set-Value $runKey $AppName $autostart
        } else {
            Note 'Start at Windows logon stays off, as it was before this update.'
        }

        Say "Installed $AppName $installedVersion in $installDir"
        if (-not $upgrade) {
            Note 'On its first start Seekgram downloads the search model and its tools:'
            Note 'about 5 GB, once. The window shows the progress; you can log in meanwhile.'
        }
        Write-Host ''
        if ($NoLaunch) {
            Write-Host '  Open Seekgram from the Start menu.'
        } else {
            Say "Opening $AppName"
            Start-Process -FilePath $exe -WorkingDirectory $installDir
        }
        if (-not $upgrade) {
            Write-Host '  Next: log in to Telegram in the Seekgram window, then send ' -NoNewline
            Write-Host '?pink dragon' -ForegroundColor White -NoNewline
            Write-Host ' to your Saved Messages.'
        }
        Write-Host ''
    } finally {
        Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
    }
}

# Empty when piped into iex; the script's own path when run as a file.
$ScriptPath = $MyInvocation.MyCommand.Path
Install-Seekgram
