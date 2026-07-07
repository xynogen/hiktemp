"""
hiktemp._fetch
~~~~~~~~~~~~~~
HTTP layer: digest auth, multipart parse, float32 blob decode.
Only depends on `requests` and `numpy`.
"""

from __future__ import annotations

import json

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from requests.auth import HTTPDigestAuth
from urllib3.util.retry import Retry

_ENDPOINT = "/ISAPI/Thermal/channels/{ch}/thermometry/jpegPicWithAppendData?format=json"
_BOUNDARY = b"--boundary"
_RETRY = Retry(
    total=3,
    backoff_factor=0.3,
    status_forcelist=[500, 502, 503, 504],
    allowed_methods=["GET"],
)


def _body(part: bytes) -> bytes:
    """Strip multipart headers, return body bytes."""
    i = part.find(b"\r\n\r\n")
    return part[i + 4 :]


def fetch(
    url: str,
    username: str | None = None,
    password: str | None = None,
    channel: int = 1,
    timeout: float = 10.0,
    session: requests.Session | None = None,
    retries: Retry | int | None = None,
) -> tuple[dict, bytes, np.ndarray]:
    """
    Pull one thermal frame from a Hikvision ISAPI endpoint.

    Parameters
    ----------
    url : str
        Base camera URL, e.g. ``"http://192.168.1.1"``.
    username, password : str, optional
        Digest-auth credentials.  Required when *session* is not provided.
    channel : int
        ISAPI thermal channel (default 1).
    timeout : float
        HTTP request timeout in seconds.
    session : requests.Session, optional
        Reuse an existing pre-authenticated session.
    retries : Retry | int | None
        Retry policy for transient errors (HTTP 500/502/503/504 and
        connection failures).  Pass an ``int`` for a simple retry count
        with default back-off, a :class:`urllib3.util.retry.Retry`
        instance for full control, or ``None`` (default) to use the
        built-in policy (3 retries, 0.3 s exponential back-off).
        Set to ``0`` to disable retries.

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
        if not username or not password:
            raise ValueError("username and password are required when session is not provided")
        session = requests.Session()
        session.auth = HTTPDigestAuth(username, password)

        retry = (
            _RETRY
            if retries is None
            else Retry(total=retries, backoff_factor=0.3)
            if isinstance(retries, int)
            else retries
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

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

    raw = resp.content
    parts = raw.split(_BOUNDARY)

    if len(parts) < 4:
        raise ValueError(f"Malformed multipart response: expected >=4 parts, got {len(parts)}")

    # Part 1 — JSON metadata
    meta_body = _body(parts[1])
    meta_body = meta_body[: meta_body.find(b"--")]
    try:
        meta = json.loads(meta_body)["JpegPictureWithAppendData"]
    except (json.JSONDecodeError, KeyError) as exc:
        raise ValueError("Failed to parse camera metadata from response") from exc

    if "jpegPicWidth" not in meta or "jpegPicHeight" not in meta:
        raise ValueError("Camera metadata missing jpegPicWidth/jpegPicHeight")

    w: int = meta["jpegPicWidth"]
    h: int = meta["jpegPicHeight"]

    # Part 2 — thermal JPEG
    jpeg = _body(parts[2])
    jpeg = jpeg[: jpeg.find(_BOUNDARY)]

    # Part 3 — float32 temperature blob
    expected = w * h * 4
    blob = _body(parts[3])[:expected]
    if len(blob) < expected:
        raise ValueError(f"Temperature blob truncated: expected {expected} bytes, got {len(blob)}")
    matrix = np.frombuffer(blob, dtype="<f4").reshape(h, w).copy()

    return meta, jpeg, matrix
