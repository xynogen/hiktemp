"""
hiktemp
~~~~~~~
Minimal radiometric temperature extraction from Hikvision thermal cameras.

    >>> from hiktemp import hiktemp
    >>> from hiktemp import hiktemp
    >>> frame = hiktemp("http://192.168.1.1", "admin", "password")
    >>> frame.matrix              # np.ndarray (H,W) float32 °C
    >>> frame.hotspot()           # (row, col) global
    >>> frame.hotspot(lo=28, hi=29)  # (row, col) within band
    >>> frame.masked(lo=28, hi=29)   # NaN outside band

Only requires: requests, numpy.
Visualization (colormap, bgr, rgba) is left to the caller via matplotlib/cv2.
"""

from __future__ import annotations

import requests

from ._fetch import fetch
from ._frame import ThermalFrame

__version__ = "0.1.0"
__all__ = ["hiktemp", "ThermalFrame"]


def hiktemp(
    url: str,
    username: str | None = None,
    password: str | None = None,
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
    username : str, optional
        Digest-auth username. Required when session is not provided.
    password : str, optional
        Digest-auth password. Required when session is not provided.
    channel : int
        ISAPI thermal channel, default 1.
    timeout : float
        HTTP request timeout in seconds.
    session : requests.Session, optional
        Reuse an existing pre-authenticated session.
        When provided, username and password are not needed.

    Returns
    -------
    ThermalFrame
        ``.matrix``           — raw float32 °C array (H, W)
        ``.jpeg``             — raw JPEG bytes (AGC, display only)
        ``.meta``             — camera descriptor dict
        ``.min/max/mean/std`` — scalar stats over full matrix
        ``.hotspot(lo, hi)``  — (row, col) of peak temp, optional band
        ``.coldspot(lo, hi)`` — (row, col) of min temp, optional band
        ``.masked(lo, hi)``   — matrix with out-of-band pixels = NaN
        ``.alpha(lo, hi)``    — float32 (H,W) quintic smoothstep mask
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
