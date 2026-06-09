"""
hiktemp._frame
~~~~~~~~~~~~~~
ThermalFrame — thin wrapper around the float32 temperature matrix.
No dependencies beyond numpy; opencv is optional.
"""

from __future__ import annotations

from typing import Literal
import numpy as np


# ── colormap (pure numpy, no matplotlib) ─────────────────────────────────────

def _inferno_lut() -> np.ndarray:
    """256×3 uint8 inferno colormap, interpolated from reference knots."""
    knots = np.array([
        [0.001462, 0.000466, 0.013866],
        [0.087411, 0.044556, 0.224813],
        [0.258234, 0.038571, 0.406485],
        [0.416331, 0.090203, 0.432943],
        [0.578304, 0.148039, 0.404411],
        [0.735683, 0.215906, 0.330245],
        [0.865006, 0.316822, 0.226055],
        [0.961293, 0.460947, 0.106217],
        [0.994738, 0.624750, 0.039886],
        [0.967972, 0.796225, 0.156918],
        [0.988362, 0.998364, 0.644924],
    ], dtype=np.float32)
    xs  = np.linspace(0, 1, len(knots))
    out = np.linspace(0, 1, 256)
    lut = np.stack([np.interp(out, xs, knots[:, c]) for c in range(3)], axis=1)
    return (lut * 255).astype(np.uint8)


_LUT_INFERNO = _inferno_lut()


def _apply_lut(norm: np.ndarray, lut: np.ndarray) -> np.ndarray:
    idx = (np.clip(norm, 0.0, 1.0) * 255).astype(np.uint8)
    return lut[idx]


def _band_alpha(
    matrix: np.ndarray,
    lo: float,
    hi: float,
) -> np.ndarray:
    """
    Quintic smoothstep alpha mask for band [lo, hi].
    Fade zone is *outside* the band: full-on inside, fades to 0 outside.
    Pure polynomial — no exp, no overflow.
    """
    fade = max((hi - lo) * 0.05, 0.1)   # 5 % of band width, min 0.1 °C

    def _quintic(t: np.ndarray) -> np.ndarray:
        t = np.clip(t, 0.0, 1.0)
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

    rise = _quintic((matrix - (lo - fade)) / fade)
    fall = _quintic(((hi + fade) - matrix) / fade)
    return (rise * fall).astype(np.float32)


# ── ThermalFrame ─────────────────────────────────────────────────────────────

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

    def __init__(
        self,
        matrix: np.ndarray,
        jpeg: bytes,
        meta: dict,
    ) -> None:
        self.matrix = matrix
        self.jpeg   = jpeg
        self.meta   = meta

    # ── stats ─────────────────────────────────────────────────────────────────

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

        Without lo/hi → global hotspot across full matrix.
        With lo/hi    → hotspot restricted to pixels inside [lo, hi] band.
        """
        if lo is not None and hi is not None:
            m = self.matrix.copy().astype(np.float32)
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

        Without lo/hi → global coldspot across full matrix.
        With lo/hi    → coldspot restricted to pixels inside [lo, hi] band.
        """
        if lo is not None and hi is not None:
            m = self.matrix.copy().astype(np.float32)
            m[self.alpha(lo, hi) < 0.01] = np.inf
        else:
            m = self.matrix
        r, c = np.unravel_index(int(np.argmin(m)), m.shape)
        return int(r), int(c)

    # ── band methods (lo/hi passed at call time) ──────────────────────────────

    def alpha(self, lo: float, hi: float) -> np.ndarray:
        """
        Float32 (H, W) quintic smoothstep mask for band [lo, hi].
        1.0 inside band, fades to 0.0 outside.
        """
        return _band_alpha(self.matrix, lo, hi)

    def masked(self, lo: float, hi: float) -> np.ndarray:
        """
        Float32 (H, W) matrix with out-of-band pixels set to NaN.
        """
        out = self.matrix.copy()
        out[self.alpha(lo, hi) < 0.01] = np.nan
        return out

    # ── opencv-compatible output ──────────────────────────────────────────────

    def to_bgr(
        self,
        cmap: Literal["inferno"] = "inferno",
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> np.ndarray:
        """
        (H, W, 3) uint8 BGR — drop-in for cv2.imshow().
        vmin/vmax default to matrix min/max.
        """
        lut  = _LUT_INFERNO
        lo   = vmin if vmin is not None else self.min
        hi   = vmax if vmax is not None else self.max
        norm = (self.matrix - lo) / max(hi - lo, 1e-6)
        rgb  = _apply_lut(norm, lut)
        return rgb[:, :, ::-1].copy()   # RGB → BGR

    def to_rgba(
        self,
        lo: float,
        hi: float,
        cmap: Literal["inferno"] = "inferno",
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> np.ndarray:
        """
        (H, W, 4) uint8 RGBA with band alpha applied.
        lo, hi : temperature band for the alpha mask.
        vmin, vmax : colormap normalisation range (default = lo, hi).
        """
        lut   = _LUT_INFERNO
        _vmin = vmin if vmin is not None else lo
        _vmax = vmax if vmax is not None else hi
        norm  = (self.matrix - _vmin) / max(_vmax - _vmin, 1e-6)
        rgb   = _apply_lut(norm, lut)
        a     = (self.alpha(lo, hi) * 255).astype(np.uint8)
        return np.dstack([rgb, a])

    # ── dunder ────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        H, W = self.matrix.shape
        return (
            f"<ThermalFrame {W}×{H} "
            f"min={self.min:.2f} max={self.max:.2f} °C>"
        )
