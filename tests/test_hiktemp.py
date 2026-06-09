"""
Smoke tests for hiktemp.
Uses debug_multipart.raw so no live camera needed.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from hiktemp._fetch import _body
from hiktemp._frame import ThermalFrame


# ── fixtures ──────────────────────────────────────────────────────────────────

RAW = os.path.join(os.path.dirname(__file__), "sample_frame.raw")
assert os.path.exists(RAW), f"test fixture missing: {RAW}"


@pytest.fixture
def raw_data():
    with open(RAW, "rb") as f:
        return f.read()


@pytest.fixture
def frame(raw_data):
    """Build a ThermalFrame directly from cached raw file."""
    parts = raw_data.split(b"--boundary")
    meta_body = _body(parts[1])
    meta = json.loads(meta_body.split(b"--")[0])["JpegPictureWithAppendData"]
    W, H = meta["jpegPicWidth"], meta["jpegPicHeight"]
    jpeg = _body(parts[2])
    jpeg = jpeg[: jpeg.find(b"--boundary")]
    blob = _body(parts[3])[: W * H * 4]
    matrix = np.frombuffer(blob, dtype="<f4").reshape(H, W).copy()
    return ThermalFrame(matrix, jpeg, meta)


@pytest.fixture
def frame_banded(raw_data):
    parts = raw_data.split(b"--boundary")
    meta = json.loads(_body(parts[1]).split(b"--")[0])["JpegPictureWithAppendData"]
    W, H = meta["jpegPicWidth"], meta["jpegPicHeight"]
    jpeg = _body(parts[2])
    jpeg = jpeg[: jpeg.find(b"--boundary")]
    blob = _body(parts[3])[: W * H * 4]
    matrix = np.frombuffer(blob, dtype="<f4").reshape(H, W).copy()
    return ThermalFrame(matrix, jpeg, meta)


# ── ThermalFrame basic ────────────────────────────────────────────────────────


def test_matrix_shape(frame):
    assert frame.matrix.shape == (288, 384)
    assert frame.matrix.dtype == np.float32


def test_stats(frame):
    assert frame.min <= frame.mean <= frame.max
    assert frame.std >= 0


def test_hotspot_in_bounds(frame):
    r, c = frame.hotspot()
    H, W = frame.matrix.shape
    assert 0 <= r < H and 0 <= c < W
    assert frame.matrix[r, c] == pytest.approx(frame.matrix.max())


def test_coldspot_in_bounds(frame):
    r, c = frame.coldspot()
    assert frame.matrix[r, c] == pytest.approx(frame.matrix.min())


def test_repr(frame):
    assert "ThermalFrame" in repr(frame)
    assert "384" in repr(frame)


# ── band masking ──────────────────────────────────────────────────────────────


def test_alpha_shape(frame_banded):
    a = frame_banded.alpha(lo=28.0, hi=29.0)
    assert a.shape == frame_banded.matrix.shape
    assert a.dtype == np.float32


def test_alpha_range(frame_banded):
    a = frame_banded.alpha(lo=28.0, hi=29.0)
    assert a.min() >= 0.0
    assert a.max() <= 1.0


def test_masked_outside_is_nan(frame_banded):
    m = frame_banded.masked(lo=28.0, hi=29.0)
    far_outside = frame_banded.matrix < 27.5
    if far_outside.any():
        assert np.all(np.isnan(m[far_outside]))


def test_masked_inside_not_nan(frame_banded):
    m = frame_banded.masked(lo=28.0, hi=29.0)
    mid = (frame_banded.matrix >= 28.2) & (frame_banded.matrix <= 28.8)
    if mid.any():
        assert not np.any(np.isnan(m[mid]))


def test_no_band_full_matrix(frame):
    # no band — hotspot/coldspot use full matrix
    r, c = frame.hotspot()
    assert frame.matrix[r, c] == pytest.approx(frame.matrix.max())


def test_hotspot_in_band(frame_banded):
    r, c = frame_banded.hotspot(lo=28.0, hi=29.0)
    assert 28.0 - 0.1 <= frame_banded.matrix[r, c] <= 29.0 + 0.1


def test_coldspot_in_band(frame_banded):
    r, c = frame_banded.coldspot(lo=28.0, hi=29.0)
    assert 28.0 - 0.1 <= frame_banded.matrix[r, c] <= 29.0 + 0.1


# ── opencv-compat output ──────────────────────────────────────────────────────


def test_to_bgr_shape(frame):
    bgr = frame.to_bgr()
    assert bgr.shape == (288, 384, 3)
    assert bgr.dtype == np.uint8


def test_to_rgba_shape(frame):
    rgba = frame.to_rgba(lo=27.0, hi=35.0)
    assert rgba.shape == (288, 384, 4)
    assert rgba.dtype == np.uint8


def test_to_rgba_alpha_banded(frame_banded):
    rgba = frame_banded.to_rgba(lo=28.0, hi=29.0)
    assert rgba[:, :, 3].min() < 255


# ── hiktemp() integration (mocked HTTP) ──────────────────────────────────────


def test_hiktemp_integration(raw_data):
    mock_resp = MagicMock()
    mock_resp.content = raw_data
    mock_resp.raise_for_status = MagicMock()

    with patch("hiktemp._fetch.requests.Session") as MockSession:
        MockSession.return_value.__enter__ = MagicMock()
        instance = MockSession.return_value
        instance.get.return_value = mock_resp

        # call fetch directly with the mock session
        from hiktemp._fetch import fetch

        with patch("hiktemp._fetch.requests.Session", return_value=instance):
            meta, jpeg, matrix = fetch("http://fake", "admin", "pass", session=instance)

    assert matrix.shape == (288, 384)
    assert matrix.dtype == np.float32
    assert matrix.min() > -21
    assert matrix.max() < 151
