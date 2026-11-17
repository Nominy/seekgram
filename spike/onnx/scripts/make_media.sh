#!/usr/bin/env bash
# Generates the synthetic test media. Photos come from picsum.photos (Unsplash, CC0-style licence).
# Usage: make_media.sh   (idempotent; existing files are kept)
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
command -v cygpath >/dev/null && ROOT="$(cygpath -m "$ROOT")"   # ffmpeg needs C:/ paths in concat lists
FF="${FFMPEG:-D:/Seekgram/tools/ffmpeg.exe}"
P="$ROOT/media/photos"; V="$ROOT/media/video"; A="$ROOT/media/audio"
mkdir -p "$P" "$V" "$A"
ff() { "$FF" -hide_banner -loglevel error -nostdin -y "$@"; }
need() { [ ! -s "$1" ]; }

# --- photos: 16 public photos, fixed ids, varied aspect ratios -----------------------------
i=0
for spec in 10/1600/1067 20/1200/800 28/1280/960 40/900/1200 64/800/800 96/1920/1080 103/1024/1536 \
            119/1400/933 152/1000/1500 169/2000/1000 177/640/480 200/1500/1000 237/1200/1200 \
            292/1280/853 433/1100/1400 659/2400/900; do
  id=${spec%%/*}; rest=${spec#*/}; w=${rest%/*}; h=${rest#*/}
  f=$(printf "%s/p%02d_id%s_%sx%s.jpg" "$P" $i "$id" "$w" "$h")
  need "$f" && curl -sL --fail "https://picsum.photos/id/$spec.jpg" -o "$f"
  i=$((i+1))
done

# --- extra image edge cases (decoder / resize / colour-mode coverage) -----------------------
need "$P/x16_testsrc2_1280x720.png" && ff -f lavfi -i testsrc2=s=1280x720 -frames:v 1 "$P/x16_testsrc2_1280x720.png"
need "$P/x17_tiny_96x64.png"        && ff -i "$P/p11_id200_1500x1000.jpg" -vf scale=96:64 -frames:v 1 "$P/x17_tiny_96x64.png"
need "$P/x18_gray_1000x1500.jpg"    && ff -i "$P/p08_id152_1000x1500.jpg" -vf format=gray -frames:v 1 -q:v 3 "$P/x18_gray_1000x1500.jpg"
need "$P/x19_alpha_800x800.png"     && ff -i "$P/p04_id64_800x800.jpg" -vf "format=rgba,geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':a='255*gt(hypot(X-400,Y-400),150)'" -frames:v 1 "$P/x19_alpha_800x800.png"
need "$P/x20_webp_1280x853.webp"    && ff -i "$P/p13_id292_1280x853.jpg" -c:v libwebp -quality 80 -frames:v 1 "$P/x20_webp_1280x853.webp"
need "$P/x21_strip_3000x300.png"    && ff -i "$P/p15_id659_2400x900.jpg" -vf "crop=2400:240:0:330,scale=3000:300" -frames:v 1 "$P/x21_strip_3000x300.png"
# x22_exif6_*.jpg (EXIF orientation 6) is written by scripts/make_exif.py because ffmpeg cannot set the tag.

# --- audio ---------------------------------------------------------------------------------
need "$A/a0_sweep_30s.wav"  && ff -f lavfi -i "aevalsrc='0.5*sin(2*PI*(100*t+(7900/60)*t*t))':s=16000:d=30" -c:a pcm_s16le "$A/a0_sweep_30s.wav"
need "$A/a1_pink_30s.flac"  && ff -f lavfi -i "anoisesrc=color=pink:seed=7:r=44100:d=30:a=0.4" -ac 2 "$A/a1_pink_30s.flac"
# Music-like: a I-vi-IV-V progression of decaying triads with a bass line, 44.1 kHz stereo MP3, 45 s.
need "$A/a2_chords_45s.mp3" && ff -f lavfi -i "aevalsrc='
 0.22*exp(-3*mod(t,0.5))*( sin(2*PI*t*(261.63*eq(mod(floor(t/2),4),0)+220.00*eq(mod(floor(t/2),4),1)+174.61*eq(mod(floor(t/2),4),2)+196.00*eq(mod(floor(t/2),4),3)))
  + sin(2*PI*t*(329.63*eq(mod(floor(t/2),4),0)+261.63*eq(mod(floor(t/2),4),1)+220.00*eq(mod(floor(t/2),4),2)+246.94*eq(mod(floor(t/2),4),3)))
  + sin(2*PI*t*(392.00*eq(mod(floor(t/2),4),0)+329.63*eq(mod(floor(t/2),4),1)+261.63*eq(mod(floor(t/2),4),2)+293.66*eq(mod(floor(t/2),4),3))))
 + 0.25*sin(2*PI*t*(65.41*eq(mod(floor(t/2),4),0)+55.00*eq(mod(floor(t/2),4),1)+43.65*eq(mod(floor(t/2),4),2)+49.00*eq(mod(floor(t/2),4),3)))
 |0.2*exp(-6*mod(t,0.25))*sin(2*PI*t*1046.5)+0.2*exp(-3*mod(t,0.5))*sin(2*PI*t*(261.63*eq(mod(floor(t/2),4),0)+220.00*eq(mod(floor(t/2),4),1)+174.61*eq(mod(floor(t/2),4),2)+196.00*eq(mod(floor(t/2),4),3)))':s=44100:d=45" -c:a libmp3lame -q:a 4 "$A/a2_chords_45s.mp3"
# Speech-like without TTS: glottal pulse train (pitch wobble) through three moving formant resonators, syllable-gated.
need "$A/a3_formant_20s.wav" && ff -f lavfi -i "aevalsrc='0.6*(0.5+0.5*sin(2*PI*3.1*t))*gt(sin(2*PI*0.37*t)+0.8,0)*(
  exp(-60*mod(t*(118+14*sin(2*PI*0.7*t)),1)/118*118/100)*sin(2*PI*(600+250*sin(2*PI*1.3*t))*mod(t*(118+14*sin(2*PI*0.7*t)),1)/118)
 +0.6*exp(-80*mod(t*(118+14*sin(2*PI*0.7*t)),1)/118*118/100)*sin(2*PI*(1500+500*sin(2*PI*0.9*t+1))*mod(t*(118+14*sin(2*PI*0.7*t)),1)/118)
 +0.3*exp(-100*mod(t*(118+14*sin(2*PI*0.7*t)),1)/118*118/100)*sin(2*PI*2600*mod(t*(118+14*sin(2*PI*0.7*t)),1)/118))':s=16000:d=20" -c:a pcm_s16le "$A/a3_formant_20s.wav"
need "$A/a4_short_3.7s.ogg"  && ff -f lavfi -i "sine=f=440:r=48000:d=3.7" -af "tremolo=f=6:d=0.7" -c:a libopus -b:a 48k "$A/a4_short_3.7s.ogg"
need "$A/a5_quiet_12s.wav"   && ff -f lavfi -i "anoisesrc=color=white:seed=3:r=16000:d=12:a=0.002" -c:a pcm_s16le "$A/a5_quiet_12s.wav"
# a6_speech_en.wav / a7_speech_ru.wav are real synthetic speech from the Windows built-in SAPI voices: scripts/make_speech.ps1.

# --- video ---------------------------------------------------------------------------------
X264=(-c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p)
need "$V/v0_testsrc2_30s.mp4"   && ff -f lavfi -i testsrc2=s=640x360:r=30:d=30 -f lavfi -i "sine=f=330:r=44100:d=30" "${X264[@]}" -c:a aac -shortest "$V/v0_testsrc2_30s.mp4"
need "$V/v1_mandelbrot_30s.mp4" && ff -f lavfi -i mandelbrot=s=854x480:r=25 -t 30 "${X264[@]}" "$V/v1_mandelbrot_30s.mp4"
# slideshow <out> <seconds per photo> <WxH> <fps> <encoder args...> -- <photos...>
slideshow() {
  local out="$1" secs="$2" size="$3" fps="$4"; shift 4
  local enc=(); while [ "$1" != "--" ]; do enc+=("$1"); shift; done; shift
  local w=${size%x*} h=${size#*x} inputs=() graph="" n=0
  for f in "$@"; do
    inputs+=(-loop 1 -t "$secs" -i "$f")
    graph+="[$n:v]scale=$w:$h:force_original_aspect_ratio=decrease,pad=$w:$h:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=$fps,format=yuv420p[s$n];"
    n=$((n+1))
  done
  for k in $(seq 0 $((n-1))); do graph+="[s$k]"; done
  ff "${inputs[@]}" -filter_complex "${graph}concat=n=$n:v=1:a=0[v]" -map "[v]" "${enc[@]}" "$out"
}
# Slideshow: 12 photos x 6 s = 72 s, 1280x720, 30 fps, silent. Chunks 0-30, 30-60 and 60-72 are used.
need "$V/v2_slideshow_72s.mp4" && slideshow "$V/v2_slideshow_72s.mp4" 6 1280x720 30 "${X264[@]}" -- "$P"/p0[0-9]_*.jpg "$P"/p1[01]_*.jpg
# Portrait Ken Burns pan over one photo, NTSC frame rate, with a music track: 24 s.
need "$V/v3_kenburns_portrait_24s.mp4" && ff -loop 1 -i "$P/p06_id103_1024x1536.jpg" -i "$A/a2_chords_45s.mp3" -vf "zoompan=z='min(zoom+0.0008,1.5)':x='iw/2-(iw/zoom/2)':y='(ih-ih/zoom)*on/720':d=720:s=720x1280:fps=30000/1001" -t 24 "${X264[@]}" -c:a aac -shortest "$V/v3_kenburns_portrait_24s.mp4"
# 58 s clip (the worker's maximum; exercises the processor's 32-frame uniform subsampling), VP9/WebM.
need "$V/v4_slides_58s.webm" && slideshow "$V/v4_slides_58s.webm" 2 640x480 24 -c:v libvpx-vp9 -crf 34 -b:v 0 -deadline realtime -cpu-used 8 -- \
  "$P"/p1[2-5]_*.jpg "$P"/p0[0-9]_*.jpg "$P"/p1[01]_*.jpg "$P"/p0[0-9]_*.jpg "$P"/p1[0-2]_*.jpg
need "$V/v5_short_0.4s.mp4"    && ff -f lavfi -i "testsrc2=s=320x240:r=25:d=0.4" "${X264[@]}" "$V/v5_short_0.4s.mp4"
need "$V/v6_life_10s.mkv"      && ff -f lavfi -i "life=s=480x480:r=15:mold=10:ratio=0.1:death_color=#102030:life_color=#f0e080,scale=960:960:flags=neighbor" -t 10 "${X264[@]}" "$V/v6_life_10s.mkv"
echo media-ok
