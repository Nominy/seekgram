<#
.SYNOPSIS
    Seekgram installer for Windows.

.DESCRIPTION
    irm https://nominy.github.io/seekgram/install.ps1 | iex

    Downloads Seekgram Setup from the latest release, checks its SHA-256 and runs
    it. Setup installs Seekgram for your user (no admin) into
    %LOCALAPPDATA%\Seekgram.App, adds it to the Start menu and "Apps & features",
    starts it at Windows logon and opens it. Seekgram then keeps itself up to date
    (Settings > "Install updates automatically").

    A copy installed by an earlier version of this script (in
    %LOCALAPPDATA%\Programs\Seekgram) is stopped and removed first. Your Telegram
    login, index and the downloaded model in %LOCALAPPDATA%\Seekgram are kept.

    When piped into iex, configure with environment variables:
      $env:SEEKGRAM_VERSION = 'v0.1.0'     # default: latest
      $env:SEEKGRAM_NO_LAUNCH = '1'        # don't open Seekgram when done

.EXAMPLE
    irm https://nominy.github.io/seekgram/install.ps1 | iex
.EXAMPLE
    & ([scriptblock]::Create((irm https://nominy.github.io/seekgram/install.ps1))) -Uninstall
.EXAMPLE
    & ([scriptblock]::Create((irm https://nominy.github.io/seekgram/install.ps1))) -Uninstall -Purge
#>
param(
    [string] $Version = $(if ($env:SEEKGRAM_VERSION) { $env:SEEKGRAM_VERSION } else { 'latest' }),
    # Run a Seekgram Setup you already have instead of downloading it.
    [string] $Setup,
    # Install without opening Seekgram afterwards (unattended installs).
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
    $Repo = if ($env:SEEKGRAM_REPO) { $env:SEEKGRAM_REPO } else { 'Nominy/seekgram' }
    # Named by `vpk pack --packId Seekgram.App` (.github/workflows/release.yml).
    $Asset = 'Seekgram.App-win-Setup.exe'

    $hive = [Microsoft.Win32.Registry]::CurrentUser
    # Where Setup installs; Velopack deletes this folder on uninstall.
    $appDir = Join-Path $env:LOCALAPPDATA 'Seekgram.App'
    $updateExe = Join-Path $appDir 'Update.exe'
    $appExe = Join-Path $appDir 'current\seekgram.exe'
    # Where the app keeps the Telegram login, the index and the model (src/paths.rs).
    $dataDir = Join-Path $env:LOCALAPPDATA $AppName
    # The copy an earlier version of this script installed.
    $oldDir = Join-Path $env:LOCALAPPDATA "Programs\$AppName"
    $oldExe = Join-Path $oldDir 'seekgram.exe'
    $oldShortcut = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\$AppName.lnk"
    $oldUninstallKey = "Software\Microsoft\Windows\CurrentVersion\Uninstall\$AppName"
    $runKey = 'Software\Microsoft\Windows\CurrentVersion\Run'
    $base = if ($env:SEEKGRAM_DOWNLOAD_URL) { $env:SEEKGRAM_DOWNLOAD_URL.TrimEnd('/') }
            elseif ($Version -eq 'latest') { "https://github.com/$Repo/releases/latest/download" }
            else { "https://github.com/$Repo/releases/download/$Version" }

    # ASCII only: Windows PowerShell 5.1 decodes piped and BOM-less scripts as ANSI.
    function Say([string] $text) { Write-Host '> ' -ForegroundColor Cyan -NoNewline; Write-Host $text }
    function Note([string] $text) { Write-Host "  $text" -ForegroundColor DarkGray }

    # Processes started from one program file, so other copies are left alone.
    function Get-Running([string] $exe) {
        @(Get-CimInstance Win32_Process -Filter "Name = 'seekgram.exe'" -ErrorAction SilentlyContinue |
            Where-Object { $_.ExecutablePath -and ($_.ExecutablePath -ieq $exe) })
    }
    # The panel is only a view and can simply be closed. The service may be in the
    # middle of writing to the index, so it is asked to finish first and only
    # stopped the hard way if it has not exited after 30 seconds.
    function Stop-Running([string] $exe) {
        if (-not (Get-Running $exe)) { return }
        Get-Running $exe | Where-Object { "$($_.CommandLine)" -notmatch '\sservice\s*$' } |
            ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
        if (-not (Get-Running $exe)) { return }
        Say "Stopping the running $AppName service"
        $deadline = (Get-Date).AddSeconds(30)
        try {
            $stop = Start-Process -FilePath $exe -ArgumentList 'stop' -WindowStyle Hidden -PassThru
            $stop.WaitForExit(30000) | Out-Null
        } catch { Note "Couldn't ask it to stop ($($_.Exception.Message))." }
        while ((Get-Running $exe) -and ((Get-Date) -lt $deadline)) { Start-Sleep -Milliseconds 500 }
        $left = Get-Running $exe
        if ($left) {
            Note 'It did not stop in 30 seconds; ending the process.'
            $left | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
            Start-Sleep -Seconds 1
        }
    }

    # Removes the copy an earlier version of this script installed: its program
    # folder, Start menu shortcut, "Apps & features" entry, PATH entry and start at
    # logon. The data folder stays; Setup's copy uses the same one.
    function Remove-OldInstall {
        if (-not (Test-Path $oldDir)) { return }
        Say "Removing the earlier $AppName install in $oldDir"
        Stop-Running $oldExe
        $key = $hive.OpenSubKey($runKey, $true)
        if ($key) {
            try {
                $value = "$($key.GetValue($AppName))"
                if ($value -and ($value -like "*$oldExe*")) { $key.DeleteValue($AppName, $false) }
            } finally { $key.Dispose() }
        }
        $hive.DeleteSubKeyTree($oldUninstallKey, $false)
        $path = [Environment]::GetEnvironmentVariable('Path', 'User')
        if ($path) {
            $parts = @($path.Split(';') | Where-Object { $_ -and ($_.TrimEnd('\') -ine $oldDir) })
            [Environment]::SetEnvironmentVariable('Path', ($parts -join ';'), 'User')
        }
        if (Test-Path $oldShortcut) { Remove-Item -Force $oldShortcut }
        Remove-Item -Recurse -Force $oldDir
        Note 'done; your login, index and model are kept'
    }

    # rmdir removes a junction or link inside the folder without following it, unlike
    # Remove-Item -Recurse in Windows PowerShell 5.1.
    function Remove-Tree([string] $path) {
        if ((Get-Item -LiteralPath $path -Force).PSIsContainer) {
            & cmd.exe /d /c rmdir /s /q "`"$path`""
            if (Test-Path -LiteralPath $path) { throw "Couldn't delete $path." }
        } else {
            Remove-Item -LiteralPath $path -Force
        }
    }

    # Deletes the data folder. If it is a junction (the data was moved to another
    # drive), the folder it points to is emptied and the junction removed, and both
    # are named.
    function Remove-Data {
        if (-not (Test-Path -LiteralPath $dataDir)) { return }
        $item = Get-Item -LiteralPath $dataDir -Force
        if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
            $target = @($item.Target)[0]
            if ($target -and (Test-Path -LiteralPath $target)) {
                Get-ChildItem -LiteralPath $target -Force | ForEach-Object { Remove-Tree $_.FullName }
                Note "Deleted the contents of $target (Telegram login, index and model), where $dataDir pointed."
            }
            [IO.Directory]::Delete($dataDir, $false)
            Note "Removed the link $dataDir."
        } else {
            Remove-Tree $dataDir
            Note "Deleted $dataDir (Telegram login, index and model)."
        }
        Note 'To also end the login on Telegram''s side: Telegram > Settings > Devices.'
    }

    # ----------------------------------------------------------- uninstall

    if ($Uninstall) {
        Remove-OldInstall
        if (Test-Path $updateExe) {
            # Velopack's uninstaller ends Seekgram without asking, so it is asked first.
            Stop-Running $appExe
            Say "Uninstalling $AppName"
            $run = Start-Process -FilePath $updateExe -ArgumentList '--uninstall', '--silent' -PassThru -Wait
            if ($run.ExitCode -ne 0) { throw "The uninstaller failed with exit code $($run.ExitCode)." }
            Say "$AppName removed."
        } else {
            Note "$AppName is not installed in $appDir."
        }
        if ($Purge) {
            Remove-Data
        } elseif (Test-Path $dataDir) {
            Note "Your Telegram login, index and the downloaded model are kept in $dataDir"
            Note 'Delete that folder, or uninstall with -Purge, to remove them too.'
        }
        return
    }
    if ($Purge) { throw '-Purge only goes together with -Uninstall.' }

    # ------------------------------------------------------------- install

    if (-not [Environment]::Is64BitOperatingSystem) { throw "$AppName needs 64-bit Windows." }

    $tmp = Join-Path ([IO.Path]::GetTempPath()) ("seekgram-" + [Guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $tmp | Out-Null
    try {
        if (-not $Setup) {
            Say "Downloading $AppName Setup ($Version)"
            $Setup = Join-Path $tmp $Asset
            try { Invoke-WebRequest -UseBasicParsing -Uri "$base/$Asset" -OutFile $Setup }
            catch { throw "Couldn't download $base/$Asset ($($_.Exception.Message))." }
            $sums = Join-Path $tmp 'SHA256SUMS'
            Invoke-WebRequest -UseBasicParsing -Uri "$base/SHA256SUMS" -OutFile $sums
            $line = Get-Content $sums | Where-Object { $_ -match "\s\*?$([regex]::Escape($Asset))$" } | Select-Object -First 1
            if (-not $line) { throw "$Asset is not listed in SHA256SUMS." }
            $expected = ($line -split '\s+')[0].ToLowerInvariant()
            $actual = (Get-FileHash -Algorithm SHA256 $Setup).Hash.ToLowerInvariant()
            if ($expected -ne $actual) { throw "Checksum mismatch for $Asset (expected $expected, got $actual)." }
            Note 'checksum ok'
        } elseif (-not (Test-Path $Setup)) {
            throw "There is no file at $Setup."
        }

        Remove-OldInstall
        # Reinstalling over a running copy: let it finish what it is doing first.
        Stop-Running $appExe

        Say "Installing $AppName"
        # A silent Setup opens nothing; Seekgram is started below.
        $run = Start-Process -FilePath $Setup -ArgumentList '--silent' -PassThru -Wait
        if ($run.ExitCode -ne 0) { throw "Setup failed with exit code $($run.ExitCode)." }
        if (-not (Test-Path $appExe)) { throw "Setup finished but $appExe is missing." }

        Say "Installed $AppName in $appDir"
        Note 'It updates itself; switch that off in Settings if you prefer.'
        Note 'On its first start Seekgram downloads the search model and its tools:'
        Note 'about 1.6 GB, once. The window shows the progress; you can log in meanwhile.'
        Write-Host ''
        if ($NoLaunch) {
            Write-Host '  Open Seekgram from the Start menu.'
        } else {
            Start-Process -FilePath $appExe -WorkingDirectory (Split-Path -Parent $appExe)
            Write-Host '  Next: log in to Telegram in the Seekgram window, then send ' -NoNewline
            Write-Host '?pink dragon' -ForegroundColor White -NoNewline
            Write-Host ' to your Saved Messages.'
        }
        Write-Host ''
    } finally {
        Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
    }
}

Install-Seekgram
