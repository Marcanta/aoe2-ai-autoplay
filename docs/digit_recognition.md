# OpenCV digit recognition

Perception latency is Tesseract process launches (see
[perception_performance.md](perception_performance.md)). The numeric top-bar
fields are read on every observation and change often, so each new frame costs
several launches even with the region cache. `aoe2_ai/perception/digit_recognizer.py`
reads them with classical OpenCV template matching instead, and keeps Tesseract
as the fallback for anything it is not sure about.

Supported fields: **wood, food, gold, stone, population (`current/cap`), the
game clock (`HH:MM:SS`) and the idle-villager count**. Everything else (age
name, selected unit, selection HP on the parchment panel, Town Center queue
text) still goes to Tesseract unchanged.

## Using it

It is opt-in and changes nothing by default: `Perception()` still uses Tesseract.

```python
from aoe2_ai.perception import Perception
from aoe2_ai.perception.ocr import TesseractOcr
from aoe2_ai.perception.digit_recognizer import HybridOcr, DigitRecognizer

perception = Perception(ocr=HybridOcr(TesseractOcr()))      # same API, same GameState
reading = DigitRecognizer().recognize(crop, "fraction")     # standalone: DigitReading(status, text, value, confidence, reason)
```

```powershell
python -m aoe2_ai.perception.digit_recognizer read --image shot.png   # what OpenCV reads in each numeric field
python -m aoe2_ai.perception.digit_recognizer build                   # rebuild the templates from the labelled fixtures
python -m aoe2_ai.digit_benchmark --extra captures --json docs/benchmarks/digits.json   # per-field comparison
python -m aoe2_ai.benchmark --ocr hybrid --json docs/benchmarks/digits_observations.json   # whole observations
```

`HybridOcr` implements the existing `OcrEngine.read()` protocol, so the
detectors, parsers, validation rules, confidence threshold, timestamps,
`StateTracker`, region cache and controllers are untouched. For a read with a
numeric whitelist (`0123456789`, `0123456789/`, `0123456789:`) on light text it
asks the recognizer first:

* **VALID** reading: returned as an `OcrResult` (text, confidence = weakest
  glyph score, `votes=("opencv:<text>",)`). The detector then parses and
  validates it exactly like a Tesseract read.
* **UNKNOWN** reading: discarded, and the same crop is read by the fallback
  engine with the same arguments, exactly as without `HybridOcr`.
* Any other read (no whitelist, dark text, another whitelist) goes straight
  to the fallback.

`HybridOcr.stats` counts reads answered by OpenCV, sent to the fallback after
an UNKNOWN, and unsupported. `launches` is the fallback's Tesseract launch
count, so `PerceptionResult.ocr_launches` keeps working.

## How it reads a field

The HUD numbers are drawn in two small bitmap-like fonts: the resource /
population / idle-count font (white, about 8–9 px tall at 1680x1050, yellow or
red population when housed) and the clock font (white with a black outline on
a textured grey plate).

1. **Channel.** `min(B, G, R)` keeps white text; population uses `max(B, G, R)`
   so yellow and red text count as ink.
2. **Contrast.** Background = median, ink peak = 99th percentile. Under 70 grey
   levels between them there is no readable text: UNKNOWN. This is what the
   main menu, an empty region and the dimmed (unfocused) game give.
3. **Segmentation.** Binarise half-way between background and peak, take
   8-connected components. Pieces of one glyph stacked vertically are joined
   (anti-aliasing breaks the thin top bar of a 5 away from its body). Components
   at least 60 % of the tallest are glyphs. A glyph touching the crop edge means
   the text is clipped: UNKNOWN. In the resource font any other ink in the text
   line is *not dropped*: the reading becomes UNKNOWN (a silently dropped piece
   could turn `15` into `1`). In the clock font small blobs are the colon dots
   and plate texture and are ignored; a missing clock digit is caught by the
   layout check instead. Single-pixel rows of texture touching a glyph are
   trimmed.
4. **Touching glyphs.** A blob wider than any digit (width > 0.95 × height) is
   cut at low-ink columns, up to three parts. A cut is kept only if every part
   matches a template confidently; otherwise UNKNOWN.
5. **Normalisation.** Each glyph is scaled to 20 px high *keeping its aspect
   ratio* (a `1` stays narrow; different digit widths stay distinguishable) and
   its intensity rescaled to 0..1, so brightness, tint and resolution drop out.
6. **Matching.** Normalised cross-correlation (`TM_CCOEFF_NORMED`, ±2 px shift)
   against every template sample of the font. A glyph is accepted only if its
   best score is ≥ **0.85** *and* it beats the best *other* character by ≥
   **0.06**. Otherwise the reading is ambiguous: UNKNOWN, never a guess.
7. **Reconstruction.** Glyphs are joined left to right and validated: digits
   `0` or no leading zero, at most 6 digits; population `a/b`; clock 6 (or 4)
   digits grouped in pairs by the spacing of the digit centres (the clock is
   drawn on a fixed pitch and the colons widen the step between pairs), minutes
   and seconds < 60. The reading's confidence is its weakest glyph's score.

A font with a character that has no template (e.g. a template set built without
a `9`) is not used at all: its fields are UNKNOWN and go to Tesseract, because a
`9` could otherwise match an `8` above threshold.

Every threshold is in `RecognizerConfig`; channel, font and segmentation rules
per kind are in `KindSpec` / `DEFAULT_KINDS`.

### Templates

`aoe2_ai/perception/digit_templates/<font>/<char>_<n>.png` (20x20 normalised
glyphs, `slash` for `/`, 68 KB in total) are cut from the labelled fixture
screenshots by `python -m aoe2_ai.perception.digit_recognizer build`: every
labelled numeric field whose segmentation yields exactly one glyph per expected
character contributes its glyphs; near-duplicates (correlation ≥ 0.985) are
skipped, at most 8 samples per character. A test checks the committed templates
equal a fresh rebuild.

## Results

Windows 10, Tesseract 5, OpenCV 5.0, the 17 labelled 1680x1050 fixture
frames. Raw numbers: [`benchmarks/digits.json`](benchmarks/digits.json)
(per field) and [`benchmarks/digits_observations.json`](benchmarks/digits_observations.json)
(whole observations).

### Per field

`python -m aoe2_ai.digit_benchmark --extra captures --repeat 3`: 100 labelled
numeric fields × 3 passes, plus 8 negative crops (main menu, dimmed window,
hidden clock) that must not give a value. *Exact* / *wrong* / *unknown* use the
detectors' rules (confidence ≥ 0.6, the detectors' parsers).

| engine | exact | wrong | unknown | negatives read as a value | mean | median | p95 | Tesseract launches / field |
|---|---|---|---|---|---|---|---|---|
| Tesseract | 291 (97.0 %) | 0 | 9 (3.0 %) | 0 | 347 ms | 320 ms | 565 ms | 2.02 |
| **OpenCV** | **300 (100 %)** | **0** | **0** | **0** | **1.50 ms** | **1.35 ms** | **2.42 ms** | 0 |
| Hybrid (OpenCV, Tesseract fallback) | 300 (100 %) | 0 | 0 | 0 | 1.42 ms | 1.32 ms | 2.27 ms | 0 |
| OpenCV, leave-one-frame-out templates | 92 (92 %) | 0 | 8 (8 %) | 0 | | | | |

Tesseract fallback rate of the hybrid: **0 / 300** on the labelled fields;
24 / 24 on the negative crops (the recognizer correctly says UNKNOWN there,
Tesseract finds nothing either).

Per field (exact / unknown / wrong, median latency):

| field | Tesseract | OpenCV |
|---|---|---|
| wood, food, gold, stone | 45/0/0 each, 316–322 ms | 45/0/0 each, 1.3 ms |
| population | 39/**6**/0, 317 ms | 45/0/0, 1.6 ms |
| game clock | 42/0/0, 330 ms | 42/0/0, 2.3 ms |
| idle count | 30/**3**/0, 313 ms | 33/0/0, 0.75 ms |

The Tesseract unknowns are its known misses (`_ocr_misses` in the labels): the
yellow `4/5` population, `7/15`, and the idle count `1`. OpenCV reads them.

**Leave-one-frame-out** rebuilds the templates without the frame being read, so
no glyph is matched against a template cut from its own screenshot. It gives 0
wrong; its 8 UNKNOWNs are the two frames holding the *only* sample of a
character (the HUD `9` in `978`, the clock `9` in `02:59`, the clock `6` in
`00:00:06`): with that character missing the whole font is refused (see above)
and the fields go to Tesseract.

**Unlabelled live captures** (`--extra captures`: 43 screenshots from the
controller and live-metrics runs, frames where values change between cycles,
not used to build templates): OpenCV and Tesseract both gave a value for 223
fields and **agreed on all 223**; OpenCV alone read 10 more (clocks on the
red-tinted frame, the yellow population, `food 0`; each checked by eye), and
Tesseract alone 4: `40` for the idle count on a *grey* button, which shows `0`
next to the icon. Those 4 are Tesseract misreads (harmless today: the pipeline
does not read the count on a grey button); OpenCV says UNKNOWN (the icon ink
touches the region edge).

### Whole observations

`python -m aoe2_ai.benchmark --ocr hybrid`: every labelled frame through
`LiveObserver.observe` (HUD fields, Town Center panel, idle button, villager
selection), 2 passes, Tesseract and hybrid in the same session. *sequence*
= the 10 in-game frames in capture order (every value changes from one frame to
the next); *cold* = each frame on a fresh tracker and cache.

| profile | cache | scenario | Tesseract mean / median / p95 | hybrid mean / median / p95 | launches / obs | ok / missed / wrong (Tesseract → hybrid) |
|---|---|---|---|---|---|---|
| full | off | cold | 1593 / 1669 / 2599 ms | **706 / 729 / 1452 ms** | 16.9 → 5.1 | 386/6/0 → 392/0/0 |
| full | off | sequence | 1645 / 1641 / 2808 ms | **703 / 722 / 1718 ms** | 18.3 → 5.1 | 238/2/0 → 240/0/0 |
| full | on | sequence | 1311 / 1329 / 2820 ms | **610 / 727 / 1741 ms** | 13.9 → 4.5 | 238/2/0 → 240/0/0 |
| town_center | on | sequence | 604 / 729 / 955 ms | **435 / 672 / 841 ms** | 5.7 → 2.5 | 78/2/0 → 80/0/0 |
| economy | on | cold | 772 / 865 / 930 ms | **31 / 31 / 36 ms** | 8.8 → 0 | 190/4/0 → 194/0/0 |
| economy | on | sequence | 621 / 679 / 947 ms | **31 / 32 / 37 ms** | 6.8 → 0 | 118/2/0 → 120/0/0 |
| any | on | steady | 23–25 ms | 23–25 ms | 0 → 0 | unchanged |

Digit fallback in observations: 0 % in *sequence*; in *cold* 7 reads per pass
go to Tesseract, all on the dimmed (unfocused) frame and the frame whose clock
is hidden, where nothing is readable and both engines report UNKNOWN.

What remains in a full observation is Tesseract on text: the age name, the
selected unit's name and HP, and the Town Center queue lines (about 5 launches).
The economy profile, which is only numbers, drops from ~0.6–0.9 s to ~31 ms, of
which 23 ms is loading the screenshot.

## Limitations

* **Small template set.** Templates come from 17 screenshots. The HUD `9` and
  the clock `1`, `6`, `9` have a single sample each. More labelled frames
  (especially high resources, 4-digit values, late-game clocks) would make the
  leave-one-out figure more meaningful; rebuild with the `build` command after
  labelling them.
* **One resolution and UI.** Measured at 1680x1050, 100 % UI scale, default HUD
  skin. Glyphs are size-normalised and regions scale with the frame, and tests
  check that 0.8× and 1.25× frames are never misread, but other resolutions,
  UI scales or HUD skins should get their own templates (and are otherwise
  mostly UNKNOWN → Tesseract).
* **Idle count over the icon.** The count overlaps the yellow villager icon;
  when a digit touches the icon's highlight the reading is UNKNOWN (Tesseract
  then reads it as before).
* **Colon not verified.** The clock layout is checked through digit spacing,
  not by matching the colon glyphs.
* **Not the default.** `Perception()` still uses Tesseract only. Switching the
  controllers to `HybridOcr` is a separate decision (a one-line change in
  `LiveObserver`), deliberately out of this change.
* **What is not covered.** Text fields, selection HP (dark text on parchment,
  another font) and the Town Center queue still use Tesseract.
