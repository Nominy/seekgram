# Synthetic speech samples from the SAPI voices that ship with Windows (no download, no private data).
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File scripts/make_speech.ps1
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$out = Join-Path (Split-Path $PSScriptRoot -Parent) 'media\audio'
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$voices = $synth.GetInstalledVoices() | ForEach-Object { $_.VoiceInfo }
$voices | ForEach-Object { Write-Output ("voice: {0} [{1}]" -f $_.Name, $_.Culture) }
$format = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(22050, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)

function Speak($culturePrefix, $file, $text) {
    $voice = $voices | Where-Object { $_.Culture.Name -like "$culturePrefix*" } | Select-Object -First 1
    if (-not $voice) { Write-Output "no $culturePrefix voice; skipping $file"; return }
    $synth.SelectVoice($voice.Name)
    $synth.SetOutputToWaveFile((Join-Path $out $file), $format)
    $synth.Speak($text)
    $synth.SetOutputToNull()
    Write-Output "wrote $file with $($voice.Name)"
}

$en = "The weather forecast for tomorrow says heavy rain in the morning, followed by sunshine in the afternoon. " +
      "Please remember to bring an umbrella, and do not forget that the meeting about the new railway bridge starts at nine o'clock. " +
      "After lunch we will visit the museum of natural history and look at the dinosaur skeletons. " +
      "In the evening there is a concert in the park with a jazz band from New Orleans."
$ru = "Завтра утром ожидается сильный дождь, а после обеда выглянет солнце. " +
      "Не забудьте взять зонт. Совещание о новом железнодорожном мосте начнётся в девять часов. " +
      "После обеда мы пойдём в музей естественной истории и посмотрим на скелеты динозавров. " +
      "Вечером в парке будет концерт джазового оркестра."
Speak 'en' 'a6_speech_en.wav' $en
Speak 'ru' 'a7_speech_ru.wav' $ru
