"""Golden reference for NSP-1 spike detection, built on SpikeInterface.

SpikeInterface does the I/O, probe/channel mapping, filtering, chunked parallel
execution and spike-train matching. The only hand-written signal processing is
the spec's adaptive threshold-crossing rule (``detect.py``); SpikeInterface's
own detectors use a fixed whole-recording noise level and peak timestamps,
which a causal real-time engine cannot reproduce.
"""

# Spec constants (SNP-Engine-Spec.pdf section in brackets)
FS = 30_000                      # [S3]
BAND = (300.0, 6000.0)           # [FR-FLT-1]
FILTER_ORDER = 2                 # scipy order 2 bandpass == 4th order, 2 biquads [FR-FLT-1]
NOISE_WINDOW = FS                # last 1 s [FR-SPK-1]
NOISE_REFRESH = FS // 10         # every 100 ms [FR-SPK-1]
MAD_SCALE = 0.6745               # [FR-SPK-1]
K = 4.5                          # [FR-SPK-2]
REFRACTORY = FS // 1000          # 1 ms [FR-SPK-3]
SNIPPET_PRE, SNIPPET_POST = 8, 24  # [FR-SPK-5]
TILE_OFFSET = 10 * FS            # copy k of the probe is delayed k * 10 s [S3]
