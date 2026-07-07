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
    Compute a quintic smoothstep alpha mask for temperature band ``[lo, hi]``.

    Returns a float32 array with the same shape as *matrix* where each pixel
    is 1.0 when its temperature is fully inside the band, 0.0 when fully
    outside, and transitions smoothly over a 5 % fade margin (minimum 0.1 °C)
    at each boundary.

    The smoothstep polynomial ``t^3(6t^2 - 15t + 10)`` is used instead of a
    sigmoid/gaussian because it is a pure polynomial -- no ``exp``, no risk
    of float overflow, and C1-continuous at both ends.

    Parameters
    ----------
    matrix : np.ndarray
        (H, W) float32 temperature matrix in °C.
    lo, hi : float
        Lower and upper bounds of the desired temperature band.

    Returns
    -------
    np.ndarray
        (H, W) float32 in [0, 1].
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

    Scalar statistics (``.min``, ``.max``, ``.mean``, ``.std``) are lazily
    computed on first access and cached for the lifetime of the frame.

    Parameters
    ----------
    matrix : np.ndarray (H, W) float32
        Per-pixel temperature in °C.
    jpeg : bytes
        Raw JPEG from the camera (AGC, display only).
    meta : dict
        JSON descriptor returned by the camera.
    """

    __slots__ = ("_stats", "jpeg", "matrix", "meta")

    def __init__(self, matrix: np.ndarray, jpeg: bytes, meta: dict) -> None:
        self.matrix = matrix
        self.jpeg = jpeg
        self.meta = meta
        self._stats: dict[str, float] | None = None

    # ── stats ──────────────────────────────────────────────────────────────────

    def _ensure_stats(self) -> dict[str, float]:
        """Compute and cache scalar stats on first access."""
        if self._stats is None:
            self._stats = {
                "min": float(self.matrix.min()),
                "max": float(self.matrix.max()),
                "mean": float(self.matrix.mean()),
                "std": float(self.matrix.std()),
            }
        return self._stats

    @property
    def min(self) -> float:
        """Minimum temperature in °C across the full matrix (cached)."""
        return self._ensure_stats()["min"]

    @property
    def max(self) -> float:
        """Maximum temperature in °C across the full matrix (cached)."""
        return self._ensure_stats()["max"]

    @property
    def mean(self) -> float:
        """Mean temperature in °C across the full matrix (cached)."""
        return self._ensure_stats()["mean"]

    @property
    def std(self) -> float:
        """Standard deviation of temperature in °C across the full matrix (cached)."""
        return self._ensure_stats()["std"]

    # ── spot finding ───────────────────────────────────────────────────────────

    def hotspot(
        self,
        lo: float | None = None,
        hi: float | None = None,
    ) -> tuple[int, int]:
        """
        Return ``(row, col)`` of the maximum temperature pixel.

        When *lo* and *hi* are both provided the search is restricted to
        pixels inside the temperature band: out-of-band pixels are replaced
        with ``-inf`` so they cannot win the argmax.

        Parameters
        ----------
        lo, hi : float, optional
            Lower/upper bounds of the temperature band in °C.
            Both must be given to activate band filtering.

        Returns
        -------
        tuple[int, int]
            ``(row, col)`` index into ``.matrix``.
        """
        if lo is not None and hi is not None:
            mask = self.alpha(lo, hi)
            m = self.matrix.copy()
            m[mask < 0.01] = -np.inf
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
        Return ``(row, col)`` of the minimum temperature pixel.

        When *lo* and *hi* are both provided the search is restricted to
        pixels inside the temperature band: out-of-band pixels are replaced
        with ``+inf`` so they cannot win the argmin.

        Parameters
        ----------
        lo, hi : float, optional
            Lower/upper bounds of the temperature band in °C.
            Both must be given to activate band filtering.

        Returns
        -------
        tuple[int, int]
            ``(row, col)`` index into ``.matrix``.
        """
        if lo is not None and hi is not None:
            mask = self.alpha(lo, hi)
            m = self.matrix.copy()
            m[mask < 0.01] = np.inf
        else:
            m = self.matrix
        r, c = np.unravel_index(int(np.argmin(m)), m.shape)
        return int(r), int(c)

    # ── band ───────────────────────────────────────────────────────────────────

    def alpha(self, lo: float, hi: float) -> np.ndarray:
        """
        Compute a quintic smoothstep alpha mask for temperature band ``[lo, hi]``.

        Returns a float32 ``(H, W)`` array where 1.0 = fully inside the band,
        0.0 = fully outside, with a smooth polynomial fade at the edges.
        See :func:`_band_alpha` for the math.

        Parameters
        ----------
        lo, hi : float
            Lower/upper bounds of the temperature band in °C.

        Returns
        -------
        np.ndarray
            ``(H, W)`` float32 in ``[0, 1]``.
        """
        return _band_alpha(self.matrix, lo, hi)

    def masked(self, lo: float, hi: float) -> np.ndarray:
        """
        Return a copy of the temperature matrix with out-of-band pixels set to NaN.

        Pixels whose :meth:`alpha` value falls below 0.01 are considered
        outside the band.

        Parameters
        ----------
        lo, hi : float
            Lower/upper bounds of the temperature band in °C.

        Returns
        -------
        np.ndarray
            ``(H, W)`` float32 — same as ``.matrix`` but with NaN outside the band.
        """
        out = self.matrix.copy()
        out[self.alpha(lo, hi) < 0.01] = np.nan
        return out

    # ── dunder ─────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        h, w = self.matrix.shape
        return f"<ThermalFrame {w}x{h} min={self.min:.2f} max={self.max:.2f} °C>"
