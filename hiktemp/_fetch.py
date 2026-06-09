"""
hiktemp._fetch
~~~~~~~~~~~~~~
HTTP layer: digest auth, multipart parse, float32 blob decode.
Only depends on `requests` and `numpy`.
"""

from __future__ import annotations

import json
import requests
from requests.auth import HTTPDigestAuth

import numpy as np

_ENDPOINT = "/ISAPI/Thermal/channels/{ch}/thermometry/jpegPicWithAppendData?format=json"
_BOUNDARY = b"--boundary"


def _body(part: bytes) -> bytes:
    """Strip multipart headers, return body bytes."""
    i = part.find(b"\r\n\r\n")
    return part[i + 4:]


def fetch(
    url: str,
    username: str,
    password: str,
    channel: int = 1,
    timeout: float = 10.0,
    session: requests.Session | None = None,
) -> tuple[dict, bytes, np.ndarray]:
    """
    Pull one thermal frame from a Hikvision ISAPI endpoint.

    Returns
    -------
    meta : dict
        Parsed JSON descriptor from Part 1.
    jpeg : bytes
        Raw JPEG bytes from Part 2.
    matrix : np.ndarray  shape (H, W)  dtype float32
        Per-pixel temperature in °C from Part 3.
    """
    own_session = session is None
    if own_session:
        session = requests.Session()
        session.auth = HTTPDigestAuth(username, password)

    try:
        resp = session.get(
            url.rstrip("/") + _ENDPOINT.format(ch=channel),
            stream=True,
            timeout=timeout,
        )
        resp.raise_for_status()
    finally:
        if own_session:
            session.close()

    raw   = resp.content
    parts = raw.split(_BOUNDARY)

    # Part 1 — JSON metadata
    meta_body = _body(parts[1])
    meta_body = meta_body[: meta_body.find(b"--")]
    meta      = json.loads(meta_body)["JpegPictureWithAppendData"]

    W: int = meta["jpegPicWidth"]
    H: int = meta["jpegPicHeight"]

    # Part 2 — thermal JPEG
    jpeg = _body(parts[2])
    jpeg = jpeg[: jpeg.find(_BOUNDARY)]

    # Part 3 — float32 temperature blob
    blob   = _body(parts[3])[: W * H * 4]
    matrix = np.frombuffer(blob, dtype="<f4").reshape(H, W).copy()

    return meta, jpeg, matrix
