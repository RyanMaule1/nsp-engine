# nsp-golden: spike-detection golden reference

This package produces golden reference outputs for the NSP-1 C++ engine's
filter and spike detector, and pytest checks that compare the engine against
them. SpikeInterface handles I/O, the probe and channel map, filtering,
chunked parallel execution, Kilosort loading and spike-train matching. The
only hand-written signal processing is the spec's adaptive-threshold crossing
rule, in `nsp_golden/detect.py` (about 50 lines).

## Setup

```bash
cd ref
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e .
mkdir -p ../data && unzip -n data.zip rawDataSample.bin params.py channel_map.npy \
  channel_positions.npy Hopkins_20160722_g0_t0.imec.ap.meta spike_times.npy \
  spike_clusters.npy cluster_groups.csv -d ../data   # (run from the repo root)

.venv/bin/nsp-golden --data ../data --out golden/374ch                   # ~2 min
.venv/bin/nsp-golden --data ../data --out golden/1024ch --channels 1024  # S3 channel model
.venv/bin/python -m pytest                                               # self-tests
```

## Testing the engine

Have the engine write two files into an output directory:

* `events.nspe`: spike events in the spec 7.2 format.
* `filtered.f32`: the filtered signal as headerless float32 little-endian,
  sample-major, with the same channels in the same order as the golden. For the
  fixed-point path, convert to ADC units first.

Then run:

```bash
.venv/bin/python -m pytest tests/test_engine.py --engine-out ../out [--fixed] [--golden golden/1024ch]
```

| Test | Pass criterion |
|---|---|
| T-12 event format | every field of every record is valid, records are sorted, and refractory and snippet are consistent |
| T-5 / PR-10 | events match on the same channel within ±1 sample, ≥ 99.9 % both ways (float) or ≥ 99.5 % (`--fixed`) |
| T-3 | `max|e| / max|golden| ≤ 1e-5` per channel (float), or `max|e| ≤ --lsb-tol` ADC units (fixed) |

## The rules the engine must match

**Input.** The input is the 374 sites listed in `channel_map.npy`, in that
(probe) order. The 10 reference sites (rows 36, 75, …, 379) and the sync row
are dropped. The samples are raw int16 values. For N > 374, channel `c` is
channel `c mod 374`, delayed by `floor(c / 374) × 10 s`. The delay wraps
around the 60 s recording.

**Filter.** The filter is
`butter(2, [300, 6000], 'bandpass', fs=30000, output='sos')`: 2 biquads,
4th order overall. It is causal (`direction="forward"`), starts from zero
state, uses float64 math and stores its output as float32. The exact
coefficients are in `manifest.json` under `filter_sos`.

**Noise.** The noise estimate is refreshed at `s = 0, 3000, 6000, …`:
`noise = sorted(|y[s-30000:s]|)[14999] / 0.6745`. That is the lower median,
so use `std::nth_element`. The threshold `thr = -4.5 × noise` is then used for
samples `[s, s+3000)`. There is no threshold before sample 30000.

**Detection.** An event fires at sample `n` if `y[n] < thr` and
`y[n-1] >= thr`, and `n - last_event >= 30` on that channel. Only emitted
events restart the refractory period.

**Record fields.** `amplitude` is `rint(y[n])` and `threshold` is `rint(thr)`.
`rint` rounds half to even, the same as `std::nearbyint`. The optional
snippet is `rint(y[n-8 : n+24])`.

## Golden outputs (`golden/<config>/`)

| File | Contents |
|---|---|
| `filtered/traces_cached_seg0.raw` | filtered signal, float32, sample-major. `si.load(golden/374ch/filtered)` also opens it. |
| `events.nspe` | golden events |
| `noise.npy` | noise level per 100 ms block, `[600, ch]` (NaN in the first second) |
| `si_peaks.npy` | SpikeInterface `by_channel` peaks, kept as a cross-check |
| `manifest.json` | input SHA-256s, config, filter coefficients and cross-check results |
| `SHA256SUMS` | checksums of the golden data. Only this file and the manifest go in git. |

The cross-checks are sanity checks and not pass criteria (spec S3):

* About 91 % of SpikeInterface `by_channel` peaks have a golden crossing 0–30
  samples before them.
* The golden detector finds a median of 90 % of the spikes from Kilosort's
  189 "good" units in the first 60 s (±15 samples, best-matching channel).

## Dataset notes (from M0)

* `rawDataSample.bin` holds **60 s** of AP data. The 1-hour AP file (87 GB,
  per `.ap.meta`) is not in `data.zip`; the LF band is full length but is
  sampled at 2.5 kHz and is no use for spikes.
* The data has already been common-average referenced (`params.py` points at
  `…ap_CAR.bin`). Per-channel DC is about 0, and the values span −375 to 149
  in 1-LSB steps.
* The sync row contains CAR-processed values, so it is not a clean sync signal.
* `.ap.meta` gives ±0.6 V over a 10-bit ADC at AP gain 500, i.e. 2.34375 µV
  per LSB. Hardware high-pass is at 300 Hz.
