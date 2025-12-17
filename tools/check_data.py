import glob
import numpy as np
from PIL import Image

paths = glob.glob(r"data\train\*\*.png")
if not paths:
    raise SystemExit("No PNGs found in data\\train\\*\\*.png")

p = paths[0]
img = np.array(Image.open(p).convert("L"))

print("Example file:", p)
print("dtype:", img.dtype)
print("min:", img.min(), "max:", img.max(), "mean:", float(img.mean()))
print("unique (first 20):", np.unique(img)[:20])
