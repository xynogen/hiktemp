"""
Smoke tests for hiktemp.
Uses synthetic in-memory fixtures — no file, no live camera needed.
"""

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from hiktemp._frame import ThermalFrame


# ── synthetic fixtures ────────────────────────────────────────────────────────

H, W = 288, 384


def _make_matrix() -> np.ndarray:
    """Synthetic 288×384 float32 temperature matrix with a known hotspot."""
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
        meta, jpeg, mat = fetch("http://fake", "admin", "password", session=instance)

    assert mat.shape == (H, W)
    assert mat.dtype == np.float32
    assert mat[100, 200] == pytest.approx(35.0, abs=1e-3)
