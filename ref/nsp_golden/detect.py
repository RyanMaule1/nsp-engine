"""FR-SPK-1..5: adaptive-threshold crossing detector (the C++ engine's contract).

Rules, on the filtered signal y (float32 golden trace, math in float64):

* Noise refresh every R = 3000 samples, at s = 0, R, 2R, ...:
  noise = sorted(|y[s-W:s]|)[(W-1)//2] / 0.6745, with W = 30000. The lower median
  is used, never the mean of the two middle values: use std::nth_element.
  thr = -k * noise holds for samples [s, s+R). There is no threshold
  (so no detection) before s = W.
* Event at sample n iff  y[n] < thr  and  y[n-1] >= thr  (both compared against the
  threshold in force at n)  and  n - last_event >= 30  (only emitted events
  restart the refractory period).
* amplitude = sat16(rint(y[n])), threshold = sat16(rint(thr)), snippet =
  sat16(rint(y[n-8 : n+24])) if it fits in the recording. rint is round-half-even
  (std::nearbyint).
"""

from __future__ import annotations

import numpy as np

from . import K, MAD_SCALE, NOISE_REFRESH, NOISE_WINDOW, REFRACTORY, SNIPPET_POST, SNIPPET_PRE


def sat16(v):
    return np.clip(np.rint(v), -32768, 32767).astype(np.int16)


def noise_levels(y: np.ndarray, window: int = NOISE_WINDOW, refresh: int = NOISE_REFRESH) -> np.ndarray:
    """(n_blocks, n_ch) noise per refresh block; NaN where the window isn't full yet."""
    n_blocks = -(-y.shape[0] // refresh)
    out = np.full((n_blocks, y.shape[1]), np.nan)
    k = (window - 1) // 2
    for b in range(window // refresh, n_blocks):
        s = b * refresh
        win = np.abs(y[s - window:s].astype(np.float64)).T          # contiguous per channel
        out[b] = np.partition(win, k, axis=1)[:, k] / MAD_SCALE
    return out


def detect(y: np.ndarray, k: float = K, refresh: int = NOISE_REFRESH, snippets: bool = False):
    """y: (n_samples, n_ch). Returns (sample, channel, amplitude, threshold, snippet|None, noise)."""
    noise = noise_levels(y, refresh=refresh)
    thr_blocks = -k * noise
    thr = np.repeat(thr_blocks, refresh, axis=0)[:len(y)]
    yy = y.astype(np.float64)
    with np.errstate(invalid="ignore"):
        cand = np.zeros(y.shape, bool)
        cand[1:] = (yy[1:] < thr[1:]) & (yy[:-1] >= thr[1:])
    n_idx, ch = np.nonzero(cand)
    order = np.lexsort((n_idx, ch))                 # per channel, in time
    n_idx, ch = n_idx[order], ch[order]
    keep = np.zeros(len(n_idx), bool)
    last_c, last_n = -1, 0
    for i, (n, c) in enumerate(zip(n_idx, ch)):     # refractory: sequential by definition
        if c != last_c or n - last_n >= REFRACTORY:
            keep[i] = True
            last_c, last_n = c, n
    n_idx, ch = n_idx[keep], ch[keep]
    order = np.lexsort((ch, n_idx))                 # output order: sample, then channel
    n_idx, ch = n_idx[order], ch[order]
    amp = sat16(yy[n_idx, ch])
    thr_ev = sat16(thr[n_idx, ch])
    snip = None
    if snippets:
        ok = (n_idx >= SNIPPET_PRE) & (n_idx + SNIPPET_POST <= len(y))
        snip = np.zeros((len(n_idx), SNIPPET_PRE + SNIPPET_POST), np.int16)
        offs = np.arange(-SNIPPET_PRE, SNIPPET_POST)
        snip[ok] = sat16(yy[n_idx[ok, None] + offs, ch[ok, None]])
        snip = (snip, ok)
    return n_idx, ch, amp, thr_ev, snip, noise
