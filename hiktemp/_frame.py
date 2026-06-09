"""
hiktemp._frame
~~~~~~~~~~~~~~
ThermalFrame — thin wrapper around the float32 temperature matrix.
No dependencies beyond numpy.
"""

from __future__ import annotations

import numpy as np


def _band_alpha(matrix: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """
    Quintic smoothstep alpha mask for band [lo, hi].
    1.0 inside band, fades to 0.0 outside over a 5% margin.
    Pure polynomial — no exp, no overflow.
    """
    fade = max((hi - lo) * 0.05, 0.1)

    def _quintic(t: np.ndarray) -> np.ndarray:
        t = np.clip(t, 0.0, 1.0)
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

    rise = _quintic((matrix - (lo - fade)) / fade)
    fall = _quintic(((hi + fade) - matrix) / fade)
    return (rise * fall).astype(np.float32)


class ThermalFrame:
    """
    A single radiometric thermal frame.  Stores only raw sensor data.
    Band filtering is applied at query time via method arguments.

    Parameters
    ----------
    matrix : np.ndarray (H, W) float32
        Per-pixel temperature in °C.
    jpeg : bytes
        Raw JPEG from the camera (AGC, display only).
    meta : dict
        JSON descriptor returned by the camera.
    """

    __slots__ = ("matrix", "jpeg", "meta")

    def __init__(self, matrix: np.ndarray, jpeg: bytes, meta: dict) -> None:
        self.matrix = matrix
        self.jpeg = jpeg
        self.meta = meta

    # ── stats ──────────────────────────────────────────────────────────────────

    @property
    def min(self) -> float:
        return float(self.matrix.min())

    @property
    def max(self) -> float:
        return float(self.matrix.max())

    @property
    def mean(self) -> float:
        return float(self.matrix.mean())

    @property
    def std(self) -> float:
        return float(self.matrix.std())

    def hotspot(
        self,
        lo: float | None = None,
        hi: float | None = None,
    ) -> tuple[int, int]:
        """
        (row, col) of maximum temperature pixel.
        With lo/hi → restricted to pixels inside band.
        """
        if lo is not None and hi is not None:
            m = self.matrix.copy()
            m[self.alpha(lo, hi) < 0.01] = -np.inf
        else:
            m = self.matrix
        r, c = np.unravel_index(int(np.argmax(m)), m.shape)
        return int(r), int(c)

    def coldspot(
        self,
        lo: float | None = None,
        hi: float | None = None,
    ) -> tuple[int, int]:
        """
        (row, col) of minimum temperature pixel.
        With lo/hi → restricted to pixels inside band.
        """
        if lo is not None and hi is not None:
            m = self.matrix.copy()
            m[self.alpha(lo, hi) < 0.01] = np.inf
        else:
            m = self.matrix
        r, c = np.unravel_index(int(np.argmin(m)), m.shape)
        return int(r), int(c)

    # ── band ───────────────────────────────────────────────────────────────────

    def alpha(self, lo: float, hi: float) -> np.ndarray:
        """Float32 (H, W) quintic smoothstep mask — 1.0 inside [lo, hi], 0.0 outside."""
        return _band_alpha(self.matrix, lo, hi)

    def masked(self, lo: float, hi: float) -> np.ndarray:
        """Float32 (H, W) matrix with out-of-band pixels set to NaN."""
        out = self.matrix.copy()
        out[self.alpha(lo, hi) < 0.01] = np.nan
        return out

    # ── dunder ─────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        H, W = self.matrix.shape
        return f"<ThermalFrame {W}×{H} min={self.min:.2f} max={self.max:.2f} °C>"
