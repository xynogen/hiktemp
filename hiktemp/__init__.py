"""
hiktemp
~~~~~~~
Minimal radiometric temperature extraction from Hikvision thermal cameras.

    >>> from hiktemp import hiktemp
    >>> frame = hiktemp("http://192.168.1.1", "admin", "password")
    >>> frame.matrix              # np.ndarray (H,W) float32 °C
    >>> frame.hotspot()           # (row, col) global
    >>> frame.hotspot(lo=28, hi=29)  # (row, col) within band
    >>> frame.masked(lo=28, hi=29)   # NaN outside band
    >>> frame.to_bgr()            # (H,W,3) uint8 — cv2.imshow() ready
    >>> frame.to_rgba(lo=28, hi=29)  # (H,W,4) uint8 with alpha mask

Only requires: requests, numpy.
opencv is optional (to_bgr / to_rgba return plain numpy arrays).
"""

from __future__ import annotations

import requests

from ._fetch import fetch
from ._frame import ThermalFrame

__version__ = "0.1.0"
__all__ = ["hiktemp", "ThermalFrame"]


def hiktemp(
    url: str,
    username: str,
    password: str,
    *,
    channel: int = 1,
    timeout: float = 10.0,
    session: requests.Session | None = None,
) -> ThermalFrame:
    """
    Pull one radiometric frame from a Hikvision thermal camera.

    Parameters
    ----------
    url : str
        Base URL of the camera, e.g. ``"http://192.168.1.1"``.
    username : str
        Digest-auth username.
    password : str
        Digest-auth password.
    channel : int
        ISAPI thermal channel, default 1.
    timeout : float
        HTTP request timeout in seconds.
    session : requests.Session, optional
        Reuse an existing session (avoids repeated digest handshakes).

    Returns
    -------
    ThermalFrame
        ``.matrix``              — raw float32 °C array (H, W)
        ``.jpeg``                — raw JPEG bytes (AGC, display only)
        ``.meta``                — camera descriptor dict
        ``.min/max/mean/std``    — scalar stats over full matrix
        ``.hotspot(lo, hi)``     — (row, col) of peak temp, optional band
        ``.coldspot(lo, hi)``    — (row, col) of min temp, optional band
        ``.masked(lo, hi)``      — matrix with out-of-band pixels = NaN
        ``.alpha(lo, hi)``       — float32 (H,W) quintic smoothstep mask
        ``.to_bgr()``            — (H,W,3) uint8, cv2.imshow()-compatible
        ``.to_rgba(lo, hi)``     — (H,W,4) uint8, with band alpha applied
    """
    meta, jpeg, matrix = fetch(
        url,
        username,
        password,
        channel=channel,
        timeout=timeout,
        session=session,
    )
    return ThermalFrame(matrix, jpeg, meta)
