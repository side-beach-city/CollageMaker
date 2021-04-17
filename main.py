from pathlib import Path
import time
import shutil
import math

import yaml
import cv2
from tqdm import tqdm
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
fl = list(filter(lambda f: not f.name.endswith(config["exclude"]), fl))
centerlogo = p / config["centerlogo"]

# prepare
work = Path("work")
if work.exists():
  shutil.rmtree("work")
time.sleep(0.1)
work.mkdir()

## Convert to 4:3 image
print(f"> Convert to 4:3 image")
for i, f in enumerate(tqdm(sorted(fl, key=lambda p: p.stem))):
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
    cv2.imwrite(f"work/{i + 1:02}.png", img2)
  finally:
    del npa
    del img
    del img2
    del canvas
print(f"Finished")

## Make collage
print(">> Make collage")
work = Path("work")
fl = list(work.glob("*.png"))
npa = np.fromfile(str(f), dtype=np.uint8)
center = cv2.imdecode(npa, cv2.IMREAD_COLOR)
canvas = None
scenter = Point(yx=center.shape[:2])
try:
  # Calculation size
  c = math.ceil(len(fl) / 4)
  y = scenter.y / c
  soneimg = Point(y / 3 * 4, y) # Size of one image(4:3)
  y = y * c
  scanvas = Point(y / 9 * 16, y) # Overall image size(16:9)
  canvas = np.ones((scanvas.y, scanvas.x, 3), np.uint8) * 255
  # Placement of individual images
  print("> Placement of individual images")
  xlist = (0, soneimg.x, scanvas.x - soneimg.x * 2, scanvas.x - soneimg.x)
  for i, f in enumerate(tqdm(list(fl))):
    inpa = np.fromfile(str(f), dtype=np.uint8)
    img = cv2.imdecode(inpa, cv2.IMREAD_COLOR)
    img2 = cv2.resize(img, dsize=(soneimg.x, soneimg.y), interpolation=cv2.INTER_LANCZOS4)
    try:
      poneimg = Point(xlist[i % 4], soneimg.y * (i // 4))
      canvas[poneimg.y: poneimg.y + soneimg.y, poneimg.x:poneimg.x + soneimg.x] = img2[0:soneimg.y, 0:soneimg.x]
    finally:
      del inpa
      del img
      del img2
  print("Finished")
  # Placement of center logo
  print("> Placement of center logo")
  inpa = np.fromfile(str(centerlogo), dtype=np.uint8)
  img = cv2.imdecode(inpa, cv2.IMREAD_COLOR)
  img2 = cv2.resize(img, dsize=(scanvas.y, scanvas.y), interpolation=cv2.INTER_LANCZOS4)
  try:
    soneimg = Point(yx=img2.shape[:2])
    poneimg = Point(scanvas.x / 2 - soneimg.x / 2, scanvas.y / 2 - soneimg.y / 2)
    canvas[poneimg.y: poneimg.y + soneimg.y, poneimg.x:poneimg.x + soneimg.x] = img2[0:soneimg.y, 0:soneimg.x]
    print("Finished")
  finally:
    del inpa
    del img
  cv2.imwrite(f"out.png", canvas)
  print("All Finished!!")
finally:
  del npa
  del center
