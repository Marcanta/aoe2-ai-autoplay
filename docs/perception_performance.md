# Perception performance

Perception latency is OCR: Tesseract is launched as a process for every pass of
every region read (~100–200 ms each). OpenCV work (HUD template, region crops,
colour and template checks of the panels) totals ~1–2 ms per observation, and
loading/grabbing the frame ~25 ms. This milestone makes observations cheaper
without changing what is recognised:

1. **Instrumentation**: per-stage timings and Tesseract launch counts on every
   observation, plus a benchmark over the labelled screenshots.
2. **Perception profiles**: a controller computes only the fields it reads.
3. **Region cache**: a field's detection is reused while the pixels it was read
   from are bit-identical.

Tesseract, the OCR passes, the vote, the confidence threshold and every parser
and validation rule are unchanged.

## Measuring

```
python -m aoe2_ai.benchmark                                   # full observation, no cache
python -m aoe2_ai.benchmark --profile town_center --cache
python -m aoe2_ai.benchmark --compare --json docs/benchmarks/after.json
python -m aoe2_ai.perception --image a.png --profile economy --cache   # prints ms and launches per frame
```

Each benchmark observation runs exactly what `LiveObserver.observe` runs (HUD
fields, Town Center panel, idle-villager button, villager selection), limited to
the profile, with the screen grab replaced by loading the screenshot (timed as
`capture`). All 17 labelled frames, 3 passes each.

* **cold**: fresh tracker and empty cache for every frame. The cost of a screen not seen before.
* **sequence**: the 10 in-game frames in capture order. A game in progress where every frame differs from the previous one.
* **steady**: each frame observed twice, the second one measured. A screen that did not change between two controller cycles.

Correctness uses the test rules (`tests/fixtures/labels.json`): *ok* = labelled
value read correctly (or a `null` field correctly not VALID), *missed* = a field
listed in `_ocr_misses` not VALID (allowed), *unknown* = a labelled field not VALID
(not allowed), *wrong* = a VALID wrong value.

`Observation.timings` / `PerceptionResult.timings` hold the per-stage seconds
(`capture`, `hud`, `regions`, `cache`, `ocr`, `parse`, `panels`, `track`, and
`total` = wall time); `ocr_launches` holds the Tesseract launches of the
observation. Detectors run in parallel threads, so `ocr` summed over threads is
larger than the wall time.

## Results

Windows 10, Tesseract 5, 1680x1050 frames. Raw numbers:
[`benchmarks/baseline.json`](benchmarks/baseline.json) (before any change) and
[`benchmarks/after.json`](benchmarks/after.json).

### Baseline (before this change)

| scenario | mean | median | p95 | launches / obs | ok | missed | unknown | wrong |
|---|---|---|---|---|---|---|---|---|
| cold | 2509 ms | 2690 ms | 3698 ms | 16.9 (max 26) | 579 | 9 | 0 | 0 |
| sequence | 2650 ms | 2615 ms | 4082 ms | 18.3 (max 26) | 357 | 3 | 0 | 0 |

Stage means (cold): capture 26 ms, ocr 6878 ms summed over threads, everything else < 2 ms.

### After

| profile | cache | scenario | mean | median | p95 | launches / obs | ok | missed | unknown | wrong |
|---|---|---|---|---|---|---|---|---|---|---|
| full | off | cold | 2284 ms | 2397 ms | 3790 ms | 16.9 | 579 | 9 | 0 | 0 |
| full | on | cold | 2115 ms | 2201 ms | 3312 ms | 16.9 | 579 | 9 | 0 | 0 |
| full | on | sequence | 1802 ms | 1729 ms | 3554 ms | 13.9 | 357 | 3 | 0 | 0 |
| full | on | steady | **27 ms** | 26 ms | 32 ms | **0** | 579 | 9 | 0 | 0 |
| town_center | off | cold | 812 ms | 954 ms | 1296 ms | 5.9 | 192 | 6 | 0 | 0 |
| town_center | on | sequence | 771 ms | 890 ms | 1161 ms | 5.7 | 117 | 3 | 0 | 0 |
| town_center | on | steady | **25 ms** | 24 ms | 29 ms | **0** | 192 | 6 | 0 | 0 |
| idle_villager | off | cold | 782 ms | 832 ms | 1416 ms | 3.9 | 252 | 3 | 0 | 0 |
| idle_villager | on | sequence | 669 ms | 827 ms | 1265 ms | 3.1 | 150 | 0 | 0 | 0 |
| idle_villager | on | steady | **26 ms** | 25 ms | 31 ms | **0** | 252 | 3 | 0 | 0 |
| economy | off | cold | 1119 ms | 1238 ms | 1402 ms | 8.8 | 285 | 6 | 0 | 0 |
| economy | on | sequence | 878 ms | 974 ms | 1342 ms | 6.8 | 177 | 3 | 0 | 0 |
| economy | on | steady | **26 ms** | 26 ms | 32 ms | **0** | 285 | 6 | 0 | 0 |

The full table (every profile × cache × scenario) is in `after.json`.

### Reading the numbers

* **Correctness is unchanged.** The full profile gives the same 579 ok / 9 allowed
  misses / 0 unknown / 0 wrong as the baseline, with and without the cache, and
  no configuration produced a wrong value. Profiles score fewer fields because
  they check only the fields they compute.
* **Profiles** cut a controller's observation by roughly 3x. The Town Center
  controller drops from ~2.3–2.5 s / 17 launches to ~0.8 s / 6 launches, and the
  idle-villager controller to ~0.8 s / 4 launches.
* **The cache** costs nothing measurable (fingerprinting all regions takes < 1 ms). It
  makes an unchanged screen nearly free: ~25 ms, almost all of it loading the
  PNG, 0 launches. When every frame differs (`sequence`, frames ~20 s of game
  time apart), it still skips the regions that stayed the same (stone, gold,
  age, population...), saving 10–30 %. In live play the gain lies between those
  two cases. Between two controller cycles a second apart, the clock and
  usually food/wood change, while most other regions do not.
* `full / off / cold` (2284 ms) runs the same code as the baseline's cold
  (2509 ms). The difference is run-to-run noise (Tesseract process start-up
  varies with system load), so compare configurations measured in the same run.

### Live run (2026-11-16, Dark Age skirmish start, 90 cycles per controller)

Recorded with `--metrics DIR` (see `aoe2_ai/controllers/metrics.py`). Every key
press was checked by eye on its before/after frames.

| controller | profile | obs. in game | mean | median | p95 | launches / obs | actions | failed | duplicates | incorrect |
|---|---|---|---|---|---|---|---|---|---|---|
| Town Center | `TOWN_CENTER` + cache | 66 | 132 ms | 43 ms | 681 ms | 0.30 | 2 TC selections, 1 villager trained | 0 | — | 0 |
| Idle villager | `IDLE_VILLAGER` + cache | 80 | 112 ms | 42 ms | 1049 ms | 0.28 | 4 villagers selected | 0 | 0 | 0 |

* Most observations reuse every OCR result (~42 ms, mostly the screen grab). The
  p95 is set by the observations where a read region changed, e.g. a new
  selection (name + HP) or food after training. Those cost 1–6 launches (0.3–1.3 s).
* Town Center: it selected the TC when nothing was selected (twice), trained
  one villager when the queue was idle with food 200 and pop 4/5 (food → 150,
  queue "Création 5% Villageois"), then waited while housed (5/5).
* Idle villager: the 4 idle villagers (the 3 starting ones and the one just
  trained) were each selected once, at 4 different positions. The controller then
  stopped pressing ("4/4 visited") until the end of the run, so there were no
  duplicate selections.
* This sample is small: the start position allows one villager before housing,
  and has 4 idle villagers. Longer runs (houses, more villagers) are still to do.

## Perception profiles

`aoe2_ai.perception.profile`:

| profile | fields |
|---|---|
| `FULL` (default) | every field |
| `TOWN_CENTER` | food, population, selection, town_center (selected / queue / villager button) |
| `IDLE_VILLAGER` | idle (button + count), selection, villager |
| `ECONOMY` | wood, food, gold, stone, population |

`town_center_text` (the progress / item lines, OCR, only used in log text) is not
in `TOWN_CENTER`: its log lines then say "unknown item". Panels that read the
selected-object name pull in `selection` automatically.

A field that is not requested is UNKNOWN with `raw_text == "not requested"`,
never a guess. The tracker treats it like an unread field: its last value is
STALE for `stale_after` seconds, then UNKNOWN. `Perception(...).analyze(frame)`
without a profile still computes every field. The controller CLIs pass their
profile to `run_live`, and `LiveObserver(profile=...)` applies it.

## Region cache

`aoe2_ai.perception.cache.RegionCache`, enabled with `Perception(cache=True)`.
A `LiveObserver` that builds its own `Perception` turns it on. Plain
`Perception()` keeps it off, so existing callers and tests are unaffected.

* **Slots and fingerprints.** One slot per detection (`wood`, `age`,
  `selection`, `tc_progress`, `tc_item`, `idle_count`...). The key is a 128-bit
  BLAKE2b digest of the shape and bytes of every region the detection reads
  (`Detector.cache_regions`, e.g. `age` + `age_progress`, `selection_name` +
  `selection_hp`). Any changed pixel misses and the detection runs again.
  Detectors that do not declare their regions are never cached.
* **What is reused.** The stored `Detection` exactly as recognised: value,
  status, confidence, raw text and the timestamp of the frame where recognition
  ran. A cached UNKNOWN stays UNKNOWN, and nothing becomes VALID that recognition
  did not make VALID. Checks that cost no OCR (HUD presence, idle-button colour,
  queue bar, template matches) run on every frame.
* **Invalidation.** The cache empties when the frame resolution, the region
  set, the OCR engine or its settings, `min_confidence` or a detector's settings
  change. It also empties on `Perception.reset()`.
* **Staleness.** Because a reused detection keeps its original timestamp, a
  value that stays unchanged on screen and then becomes unreadable goes UNKNOWN
  sooner than without the cache. The cache never extends how long a value is
  trusted. The controllers only act on VALID values, so their behaviour does not
  depend on it.

## Not done here

* Digit template matching (out of scope for this change).
* Running the panel OCR (Town Center lines, idle count) in parallel with the HUD
  detectors. It currently runs after them, because the panels need the
  selection read first.
