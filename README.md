# hiktemp

Minimal radiometric temperature extraction from Hikvision thermal cameras via ISAPI.  
No SDK required. Only depends on `requests` and `numpy`.

## Install

```bash
pip install hiktemp
```

Or from source:

```bash
git clone https://github.com/yourname/hiktemp
cd hiktemp
pip install -e .
```

## Usage

```python
from hiktemp import hiktemp

# pull one frame
frame = hiktemp("http://192.168.1.1", "admin", "password")

frame.matrix          # np.ndarray (H, W) float32 — per-pixel °C
frame.jpeg            # bytes — raw JPEG from camera (AGC, display only)
frame.meta            # dict  — camera descriptor
frame.min             # float — global min °C
frame.max             # float — global max °C
frame.mean            # float — global mean °C
frame.std             # float — global std °C
frame.hotspot()       # (row, col) — global hotspot
frame.coldspot()      # (row, col) — global coldspot
frame.to_bgr()        # np.ndarray (H,W,3) uint8 — cv2.imshow() ready

# band filter — lo/hi passed at call time
frame.hotspot(lo=28, hi=29)       # hotspot within band
frame.coldspot(lo=28, hi=29)      # coldspot within band
frame.masked(lo=28, hi=29)        # matrix, out-of-band pixels = NaN
frame.alpha(lo=28, hi=29)         # float32 (H,W) quintic smoothstep mask
frame.to_rgba(lo=28, hi=29)       # (H,W,4) uint8 with alpha mask applied

# reuse session for continuous polling
import requests
from requests.auth import HTTPDigestAuth

session = requests.Session()
session.auth = HTTPDigestAuth("admin", "password")

while True:
    frame = hiktemp("http://192.168.1.1", "admin", "password", session=session)
    bgr   = frame.to_bgr()
    # cv2.imshow("thermal", bgr)
```

## Requirements

- Python >= 3.10
- `requests >= 2.28`
- `numpy >= 1.23`

opencv-python is **optional** — `to_bgr()` and `to_rgba()` return plain numpy arrays.

## Tested on

- DS-2TD2138-10/QY (384×288, float32 °C, channel 1)
