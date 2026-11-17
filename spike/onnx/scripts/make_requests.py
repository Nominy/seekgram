"""Writes media/requests.jsonl (the fixed fidelity test set) with absolute paths for this checkout.
Usage: <any python> scripts/make_requests.py"""
import json
import os
import sys

sys.dont_write_bytecode = True
root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "media"))

# (query, index of the photo it describes in the sorted p*.jpg list)
TEXTS = [
    ("a bear looking at the camera", 14),
    ("чёрный щенок на деревянном полу", 12),
    ("фиолетовый цветок крупным планом", 8),
    ("игровой контроллер на столе", 5),
    ("woman in sunglasses holding flowers", 4),
    ("турист с рюкзаком в горах", 10),
    ("highland cow on a beach", 11),
    ("свежие овощи и лук на разделочной доске", 13),
    ("Laptop auf einem weißen Schreibtisch", 7),
    ("un río en un bosque verde", 2),
    ("des teckels qui reniflent l'herbe", 9),
    ("森の向こうに見える海", 0),
    ("кошачий нос крупным планом", 3),
    ("feet in sneakers resting in a field", 6),
    ("хаски в шлейке", 15),
    ("design books, pencils and a phone on a desk", 1),
    ("кот", None),
    ("The quick brown fox jumps over the lazy dog. " * 12, None),
    ("фото заката 2019 года, море, пляж 🌅 #отпуск", None),
    ("مدينة قديمة في الليل", None),
]
AUDIO = [
    ("a0_sweep_30s.wav", 0, 30), ("a1_pink_30s.flac", 0, 30), ("a2_chords_45s.mp3", 0, 30),
    ("a2_chords_45s.mp3", 15, 45), ("a3_formant_20s.wav", 0, 20), ("a4_short_3.7s.ogg", None, None),
    ("a5_quiet_12s.wav", 0, 12), ("a6_speech_en.wav", None, None), ("a7_speech_ru.wav", 0, 30),
    ("a1_pink_30s.flac", 10, 25.5),
]
VIDEO = [
    ("v0_testsrc2_30s.mp4", None, None), ("v1_mandelbrot_30s.mp4", None, None), ("v2_slideshow_72s.mp4", 0, 30),
    ("v2_slideshow_72s.mp4", 30, 60), ("v2_slideshow_72s.mp4", 60, 72), ("v3_kenburns_portrait_24s.mp4", None, None),
    ("v4_slides_58s.webm", 0, 58), ("v5_short_0.4s.mp4", None, None), ("v6_life_10s.mkv", None, None),
    ("v1_mandelbrot_30s.mp4", 12.5, 27.25),
]

requests = []
for i, (text, photo) in enumerate(TEXTS):
    requests.append({"id": f"t{i:02d}", "kind": "text", "input": text, "_photo": photo})
photos = sorted(os.listdir(os.path.join(root, "photos")))
for name in photos:
    requests.append({"id": "i_" + name.split("_")[0], "kind": "image", "input": os.path.join(root, "photos", name)})


def bounded(prefix, kind, folder, items):
    for i, (name, start, end) in enumerate(items):
        request = {"id": f"{prefix}{i:02d}", "kind": kind, "input": os.path.join(root, folder, name)}
        if start is not None:
            request["start_seconds"] = start
        if end is not None:
            request["end_seconds"] = end
        requests.append(request)


bounded("a", "audio", "audio", AUDIO)
bounded("v", "video", "video", VIDEO)
with open(os.path.join(root, "requests.jsonl"), "w", encoding="utf-8", newline="\n") as handle:
    for request in requests:
        handle.write(json.dumps(request, ensure_ascii=False) + "\n")
print(len(requests), "requests")
