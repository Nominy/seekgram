"""Writes media/photos/x22_exif6_1200x900.jpg: a photo stored rotated with EXIF Orientation=6, so a
viewer (and the worker's ImageOps.exif_transpose) shows it upright as 900x1200.
Usage: <any python with Pillow> scripts/make_exif.py"""
import os
import sys

sys.dont_write_bytecode = True
from PIL import Image

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "media", "photos")
source = Image.open(os.path.join(root, "p03_id40_900x1200.jpg")).convert("RGB")
stored = source.transpose(Image.Transpose.ROTATE_90)  # orientation 6 = rotate 90 CW to display
exif = Image.Exif()
exif[0x0112] = 6
stored.save(os.path.join(root, "x22_exif6_1200x900.jpg"), quality=90, exif=exif)
print(stored.size)
