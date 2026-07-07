"""
Smoke tests for hiktemp.
Uses synthetic in-memory fixtures — no file, no live camera needed.
"""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from hiktemp._frame import ThermalFrame

# ── synthetic fixtures ────────────────────────────────────────────────────────

H, W = 288, 384


def _make_matrix() -> np.ndarray:
    """Synthetic 288x384 float32 temperature matrix with a known hotspot."""
    rng = np.random.default_rng(42)
    m = rng.uniform(27.0, 30.0, (H, W)).astype(np.float32)
    m[100, 200] = 35.0  # known hotspot
    m[50, 50] = 20.0  # known coldspot
    return m


def _make_raw_response(matrix: np.ndarray) -> bytes:
    """Build a minimal multipart response matching the camera wire format."""
    meta = {
        "JpegPictureWithAppendData": {
            "jpegPicWidth": W,
            "jpegPicHeight": H,
            "temperatureDataLength": 4,
            "p2pDataLen": W * H * 4,
        }
    }
    # _fetch.py strips up to b"--" inside Part 1 body, so append it as separator
    meta_bytes = json.dumps(meta).encode() + b"--"
    jpeg_bytes = b"\xff\xd8\xff\xd9"  # minimal valid JPEG stub
    blob_bytes = matrix.tobytes()  # float32 little-endian

    def part(body: bytes) -> bytes:
        return b"\r\nContent-Type: application/octet-stream\r\n\r\n" + body

    return (
        b"--boundary"
        + part(meta_bytes)
        + b"--boundary"
        + part(jpeg_bytes)
        + b"--boundary"
        + part(blob_bytes)
        + b"--boundary--"
    )


@pytest.fixture(scope="session")
def matrix() -> np.ndarray:
    return _make_matrix()


@pytest.fixture(scope="session")
def raw_response(matrix) -> bytes:
    return _make_raw_response(matrix)


@pytest.fixture(scope="session")
def frame(matrix) -> ThermalFrame:
    return ThermalFrame(matrix, b"\xff\xd8\xff\xd9", {"synthetic": True})


@pytest.fixture(scope="session")
def frame_banded(matrix) -> ThermalFrame:
    return ThermalFrame(matrix, b"\xff\xd8\xff\xd9", {"synthetic": True})


# ── ThermalFrame basic ────────────────────────────────────────────────────────


def test_matrix_shape(frame):
    assert frame.matrix.shape == (H, W)
    assert frame.matrix.dtype == np.float32


def test_stats(frame):
    assert frame.min <= frame.mean <= frame.max
    assert frame.std >= 0


def test_hotspot_in_bounds(frame):
    r, c = frame.hotspot()
    assert 0 <= r < H and 0 <= c < W
    assert frame.matrix[r, c] == pytest.approx(frame.matrix.max())


def test_hotspot_is_known(frame):
    r, c = frame.hotspot()
    assert (r, c) == (100, 200)


def test_coldspot_in_bounds(frame):
    r, c = frame.coldspot()
    assert 0 <= r < H and 0 <= c < W
    assert frame.matrix[r, c] == pytest.approx(frame.matrix.min())


def test_coldspot_is_known(frame):
    r, c = frame.coldspot()
    assert (r, c) == (50, 50)


def test_repr(frame):
    assert "ThermalFrame" in repr(frame)
    assert str(W) in repr(frame)


# ── band masking ──────────────────────────────────────────────────────────────


def test_alpha_shape(frame_banded):
    a = frame_banded.alpha(lo=28.0, hi=29.0)
    assert a.shape == (H, W)
    assert a.dtype == np.float32


def test_alpha_range(frame_banded):
    a = frame_banded.alpha(lo=28.0, hi=29.0)
    assert a.min() >= 0.0
    assert a.max() <= 1.0 + 1e-6  # float32 rounding


def test_masked_outside_is_nan(frame_banded):
    m = frame_banded.masked(lo=28.0, hi=29.0)
    # hotspot (35.0) and coldspot (20.0) are far outside band — must be NaN
    assert np.isnan(m[100, 200])
    assert np.isnan(m[50, 50])


def test_masked_inside_not_nan(frame_banded):
    m = frame_banded.masked(lo=28.0, hi=29.0)
    # matrix is uniform ~28-30, find a pixel solidly inside band
    mid = (frame_banded.matrix >= 28.2) & (frame_banded.matrix <= 28.8)
    if mid.any():
        assert not np.any(np.isnan(m[mid]))


def test_hotspot_in_band(frame_banded):
    r, c = frame_banded.hotspot(lo=28.0, hi=29.0)
    assert 28.0 - 0.1 <= frame_banded.matrix[r, c] <= 29.0 + 0.1


def test_coldspot_in_band(frame_banded):
    r, c = frame_banded.coldspot(lo=28.0, hi=29.0)
    assert 28.0 - 0.1 <= frame_banded.matrix[r, c] <= 29.0 + 0.1


# ── hiktemp() integration (mocked HTTP) ──────────────────────────────────────


def test_hiktemp_integration(raw_response):
    mock_resp = MagicMock()
    mock_resp.content = raw_response
    mock_resp.raise_for_status = MagicMock()

    from hiktemp._fetch import fetch

    instance = MagicMock()
    instance.get.return_value = mock_resp

    with patch("hiktemp._fetch.requests.Session", return_value=instance):
        _meta, _jpeg, mat = fetch("http://fake", "admin", "password", session=instance)

    assert mat.shape == (H, W)
    assert mat.dtype == np.float32
    assert mat[100, 200] == pytest.approx(35.0, abs=1e-3)


# ── fetch error paths ─────────────────────────────────────────────────────────


def test_fetch_missing_credentials():
    from hiktemp._fetch import fetch

    with pytest.raises(ValueError, match="username and password are required"):
        fetch("http://fake")

    with pytest.raises(ValueError, match="username and password are required"):
        fetch("http://fake", username="admin")

    with pytest.raises(ValueError, match="username and password are required"):
        fetch("http://fake", password="pass")


def test_fetch_http_error():
    """HTTP 401/500 should propagate via raise_for_status."""
    from requests.exceptions import HTTPError

    from hiktemp._fetch import fetch

    instance = MagicMock()
    resp = MagicMock()
    resp.raise_for_status.side_effect = HTTPError("401 Unauthorized")
    instance.get.return_value = resp

    with pytest.raises(HTTPError, match="401"):
        fetch("http://fake", session=instance)


def test_fetch_malformed_multipart_too_few_parts():
    """Response with <4 boundary-delimited parts should raise ValueError."""
    from hiktemp._fetch import fetch

    instance = MagicMock()
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    # Only 2 parts (preamble + one body) — not enough
    resp.content = b"--boundary\r\nsome data--boundary--"
    instance.get.return_value = resp

    with pytest.raises(ValueError, match="Malformed multipart response"):
        fetch("http://fake", session=instance)


def test_fetch_malformed_json():
    """Garbage JSON in Part 1 should raise ValueError."""
    from hiktemp._fetch import fetch

    def part(body: bytes) -> bytes:
        return b"\r\nContent-Type: application/octet-stream\r\n\r\n" + body

    raw = (
        b"--boundary"
        + part(b"NOT VALID JSON--")
        + b"--boundary"
        + part(b"jpeg")
        + b"--boundary"
        + part(b"blob")
        + b"--boundary--"
    )
    instance = MagicMock()
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.content = raw
    instance.get.return_value = resp

    with pytest.raises(ValueError, match="Failed to parse camera metadata"):
        fetch("http://fake", session=instance)


def test_fetch_missing_metadata_keys():
    """JSON missing jpegPicWidth/jpegPicHeight should raise ValueError."""
    from hiktemp._fetch import fetch

    meta = {"JpegPictureWithAppendData": {"other": 1}}

    def part(body: bytes) -> bytes:
        return b"\r\nContent-Type: application/octet-stream\r\n\r\n" + body

    raw = (
        b"--boundary"
        + part(json.dumps(meta).encode() + b"--")
        + b"--boundary"
        + part(b"jpeg")
        + b"--boundary"
        + part(b"blob")
        + b"--boundary--"
    )
    instance = MagicMock()
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.content = raw
    instance.get.return_value = resp

    with pytest.raises(ValueError, match="missing jpegPicWidth/jpegPicHeight"):
        fetch("http://fake", session=instance)


def test_fetch_truncated_blob():
    """Blob shorter than W*H*4 should raise ValueError."""
    from hiktemp._fetch import fetch

    meta = {
        "JpegPictureWithAppendData": {
            "jpegPicWidth": 384,
            "jpegPicHeight": 288,
        }
    }

    def part(body: bytes) -> bytes:
        return b"\r\nContent-Type: application/octet-stream\r\n\r\n" + body

    raw = (
        b"--boundary"
        + part(json.dumps(meta).encode() + b"--")
        + b"--boundary"
        + part(b"\xff\xd8\xff\xd9")
        + b"--boundary"
        + part(b"\x00" * 100)  # way too short for 384*288*4
        + b"--boundary--"
    )
    instance = MagicMock()
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.content = raw
    instance.get.return_value = resp

    with pytest.raises(ValueError, match="Temperature blob truncated"):
        fetch("http://fake", session=instance)


# ── ThermalFrame edge cases ───────────────────────────────────────────────────


def test_single_pixel_matrix():
    """1x1 matrix should work for all operations."""
    m = np.array([[25.0]], dtype=np.float32)
    f = ThermalFrame(m, b"", {})

    assert f.min == pytest.approx(25.0)
    assert f.max == pytest.approx(25.0)
    assert f.mean == pytest.approx(25.0)
    assert f.std == pytest.approx(0.0)
    assert f.hotspot() == (0, 0)
    assert f.coldspot() == (0, 0)
    assert "ThermalFrame" in repr(f)


def test_band_lo_equals_hi():
    """lo == hi should produce a very narrow band (only fade region)."""
    m = np.array([[24.0, 25.0, 26.0]], dtype=np.float32)
    f = ThermalFrame(m, b"", {})

    a = f.alpha(lo=25.0, hi=25.0)
    assert a.shape == (1, 3)
    assert a.dtype == np.float32
    # center pixel (25.0) should have highest alpha
    assert a[0, 1] >= a[0, 0]
    assert a[0, 1] >= a[0, 2]


def test_band_covers_full_range():
    """Band spanning full matrix range should keep all pixels."""
    m = _make_matrix()
    f = ThermalFrame(m, b"", {})

    lo, hi = float(m.min()), float(m.max())
    masked = f.masked(lo=lo, hi=hi)
    # no pixel should be NaN when band covers everything
    assert not np.any(np.isnan(masked))

    a = f.alpha(lo=lo, hi=hi)
    # all pixels should have alpha close to 1.0
    assert a.min() > 0.9


def test_band_excludes_everything():
    """Band far outside data range should NaN everything."""
    m = _make_matrix()  # ~20-35 range
    f = ThermalFrame(m, b"", {})

    masked = f.masked(lo=100.0, hi=200.0)
    assert np.all(np.isnan(masked))


def test_matrix_with_nan_input():
    """NaN in input matrix should not crash stats or spot-finding."""
    m = np.array([[25.0, np.nan, 30.0], [28.0, 27.0, 26.0]], dtype=np.float32)
    f = ThermalFrame(m, b"", {})

    # numpy propagates NaN in min/max — just ensure no crash
    assert isinstance(f.min, float)
    assert isinstance(f.max, float)
    assert isinstance(f.mean, float)
    assert isinstance(f.std, float)

    # hotspot/coldspot should return valid indices (not crash)
    r, c = f.hotspot()
    assert 0 <= r < 2 and 0 <= c < 3

    r, c = f.coldspot()
    assert 0 <= r < 2 and 0 <= c < 3


def test_uniform_matrix():
    """All-same-value matrix — hotspot/coldspot should return valid coords."""
    m = np.full((10, 10), 25.0, dtype=np.float32)
    f = ThermalFrame(m, b"", {})

    assert f.min == pytest.approx(25.0)
    assert f.max == pytest.approx(25.0)
    assert f.std == pytest.approx(0.0)

    r, c = f.hotspot()
    assert 0 <= r < 10 and 0 <= c < 10

    r, c = f.coldspot()
    assert 0 <= r < 10 and 0 <= c < 10


def test_hotspot_coldspot_with_only_lo_or_hi(frame):
    """Passing only lo or only hi (not both) should use full matrix."""
    r1, c1 = frame.hotspot(lo=28.0)
    r2, c2 = frame.hotspot()
    assert (r1, c1) == (r2, c2)

    r1, c1 = frame.coldspot(hi=29.0)
    r2, c2 = frame.coldspot()
    assert (r1, c1) == (r2, c2)


def test_stats_are_cached():
    """Accessing stats multiple times should reuse the cached dict."""
    m = _make_matrix()
    f = ThermalFrame(m, b"", {})

    # First access triggers computation
    _ = f.min
    stats_ref = f._stats
    assert stats_ref is not None

    # Subsequent accesses return the same cached dict
    _ = f.max
    _ = f.mean
    _ = f.std
    assert f._stats is stats_ref


def test_fetch_retries_disabled(raw_response):
    """retries=0 should still work (no retry adapter)."""
    from hiktemp._fetch import fetch

    mock_resp = MagicMock()
    mock_resp.content = raw_response
    mock_resp.raise_for_status = MagicMock()

    instance = MagicMock()
    instance.get.return_value = mock_resp

    with patch("hiktemp._fetch.requests.Session", return_value=instance):
        _meta, _jpeg, mat = fetch(
            "http://fake",
            "admin",
            "password",
            retries=0,
        )

    assert mat.shape == (H, W)
