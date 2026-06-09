# hiktemp

Minimal radiometric temperature extraction from Hikvision thermal cameras via ISAPI.  
No SDK required. Only depends on `requests` and `numpy`.

## Install

```bash
pip install hiktemp
```

Or from source:

```bash
git clone https://github.com/xynogen/hiktemp
cd hiktemp
pip install -e .
```

## Usage

```python
from hiktemp import hiktemp

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

# band filter — lo/hi passed at call time
frame.hotspot(lo=28, hi=29)    # hotspot within band
frame.coldspot(lo=28, hi=29)   # coldspot within band
frame.masked(lo=28, hi=29)     # matrix with out-of-band pixels = NaN
frame.alpha(lo=28, hi=29)      # float32 (H,W) quintic smoothstep mask 0–1
```

Visualization is left to the caller:

```python
import matplotlib.pyplot as plt

plt.imshow(frame.matrix, cmap="inferno")
plt.colorbar(label="°C")
plt.show()

# with band mask
import numpy as np
import matplotlib.pyplot as plt

rgba = plt.get_cmap("inferno")(frame.alpha(lo=28, hi=29))
plt.imshow(rgba)
plt.show()
```

Reuse session for continuous polling:

```python
import requests
from requests.auth import HTTPDigestAuth

session = requests.Session()
session.auth = HTTPDigestAuth("admin", "password")

while True:
    frame = hiktemp("http://192.168.1.1", "admin", "password", session=session)
    print(frame.hotspot())
```

## Requirements

- Python >= 3.10
- `requests >= 2.28`
- `numpy >= 1.23`

## Tested on

- DS-2TD2138-10/QY (384×288, float32 °C, channel 1)
