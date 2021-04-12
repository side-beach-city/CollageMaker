from pathlib import Path
import time
import shutil

import yaml
import cv2
import numpy as np

class Point:
  def __init__(self, x=0, y=0, xy=None, yx=None) -> None:
    if not yx is None: xy = list(reversed(yx))
    if not xy is None:
      self.x = int(xy[0])
      self.y = int(xy[1])
    else:
      self.x = int(x)
      self.y = int(y)
  def __repr__(self):
    return f"X:{self.x} Y:{self.y}"

IM = Point(640, 480)

# setting
with open("config.yml", "r", encoding="UTF-8") as f:
  config = yaml.safe_load(f)

p = Path(config["dir"])
fl = []
for f in config["filepattern"]:
  fl += list(p.glob(f))
fl = filter(lambda f: not f.name.endswith(config["exclude"]), fl)

# prepare
work = Path("work")
if work.exists():
  shutil.rmtree("work")
time.sleep(0.1)
work.mkdir()

# Convert to 4:3 image
for i, f in enumerate(fl):
  img = None
  img2 = None
  canvas = None
  npa = np.fromfile(str(f), dtype=np.uint8)
  img = cv2.imdecode(npa, cv2.IMREAD_COLOR)
  try:
    sbase = Point(yx=img.shape[:2])
    # Clipping
    if sbase.y > sbase.x:
      scanvas = Point(sbase.x, sbase.x / 4 * 3)
    elif sbase.x > sbase.y:
      scanvas = Point(sbase.y / 3 * 4, sbase.y)
    else:
      scanvas = Point(sbase.x / 3 * 4, sbase.y)
    simage = Point(min(scanvas.x, sbase.x), min(scanvas.y, sbase.y))
    ctl = Point(scanvas.x / 2 - simage.x / 2, scanvas.y / 2 - simage.y / 2)
    stl = Point(sbase.x / 2 - simage.x / 2, sbase.y / 2 - simage.y / 2)
    canvas = np.ones((scanvas.y, scanvas.x, 3), np.uint8) * 255
    canvas[ctl.y: ctl.y + simage.y, ctl.x: ctl.x + simage.x] = \
      img[stl.y: stl.y + simage.y, stl.x: stl.x + simage.x]
    img2 = cv2.resize(canvas, dsize=(IM.x, IM.y), interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite(f"work/{i + 1}.png", img2)
  finally:
    del npa
    del img
    del img2
    del canvas

# Make collage

