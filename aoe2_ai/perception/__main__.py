"""Perception CLI.

    python -m aoe2_ai.perception                         # one live capture, print GameState JSON
    python -m aoe2_ai.perception --image a.png b.png     # analyse saved screenshots
    python -m aoe2_ai.perception --live --debug          # live loop with debug windows (q / Esc quits)
    python -m aoe2_ai.perception --image a.png --save captures/perception   # dump frame, overlay, crops, JSON
    python -m aoe2_ai.perception --calibrate --image a.png   # draw ROIs and save raw crops, no OCR

Debug windows: the frame with ROIs and readings, and a mosaic of each crop next to
what Tesseract saw with the per-pass votes. The GameState JSON is printed to the console.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from aoe2_ai.debug import close_windows, show_image

from .capture import Frame, LiveSource, file_frames
from .debug import crops_mosaic, overlay, save_debug
from .pipeline import Perception
from .profile import PROFILES, PerceptionProfile
from .regions import DEFAULT_REGIONS_PATH, RegionSet


def calibrate(frames: list[Frame], regions: RegionSet, out_dir: Path) -> None:
    for frame in frames:
        scaled = regions.for_frame(frame.image)
        img = frame.image.copy()
        stem = Path(frame.source).stem if frame.source != "screen" else time.strftime("screen_%Y%m%d_%H%M%S")
        crop_dir = out_dir / f"{stem}_crops"
        crop_dir.mkdir(parents=True, exist_ok=True)
        for region in scaled:
            x, y, w, h = region.rect
            color = (80, 200, 80) if region.calibrated else (150, 150, 150)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
            cv2.putText(img, region.name, (x, y + h + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
            crop = region.crop(frame.image)
            if crop.size:
                cv2.imwrite(str(crop_dir / f"{region.name}.png"), cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST))
        path = out_dir / f"{stem}_regions.png"
        cv2.imwrite(str(path), img)
        print(f"{path}  (crops in {crop_dir})")
    print(f"Edit {DEFAULT_REGIONS_PATH} (or pass --regions) to move rectangles; coordinates are in "
          f"{regions.reference_size[0]}x{regions.reference_size[1]} space.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Read the AoE2 game state from the screen or screenshots.")
    parser.add_argument("--image", nargs="+", type=Path, help="analyse these screenshots instead of the screen")
    parser.add_argument("--live", action="store_true", help="capture continuously until q / Esc / Ctrl+C")
    parser.add_argument("--interval", type=float, default=0.5, help="seconds between live analyses")
    parser.add_argument("--debug", action="store_true", help="show debug windows")
    parser.add_argument("--save", type=Path, metavar="DIR", help="save frame, overlay, crops and state JSON to DIR")
    parser.add_argument("--calibrate", action="store_true", help="draw regions and save crops, no OCR")
    parser.add_argument("--regions", type=Path, default=DEFAULT_REGIONS_PATH, help="region JSON file")
    parser.add_argument("--no-hud-check", action="store_true", help="OCR even when the in-game HUD is not detected")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="full", help="fields to read (default: all)")
    parser.add_argument("--cache", action="store_true", help="reuse detections of regions whose pixels did not change")
    args = parser.parse_args()

    regions = RegionSet.load(args.regions)

    if args.calibrate:
        if args.image:
            frames = list(file_frames(args.image))
        else:
            with LiveSource() as src:
                frames = [src.grab()]
        calibrate(frames, regions, args.save or Path("captures/calibration"))
        return

    perception = Perception(regions, hud_check=not args.no_hud_check, profile=PerceptionProfile.named(args.profile), cache=args.cache)
    live = None if args.image else LiveSource()
    frames = file_frames(args.image) if args.image else (iter(live) if args.live else iter([live.grab()]))
    wait_ms = max(1, int(args.interval * 1000)) if args.live else 0

    try:
        for frame in frames:
            started = time.perf_counter()
            result = perception.analyze(frame)
            elapsed = time.perf_counter() - started
            launches = "" if result.ocr_launches is None else f", {result.ocr_launches} Tesseract launches"
            print(f"--- {frame.source}  ({elapsed * 1000:.0f} ms{launches})")
            print(result.state.to_json())
            if args.save:
                print(f"saved {save_debug(args.save, frame.image, result)}")
            if args.debug:
                show_image(crops_mosaic(frame.image, result), "perception crops", wait_ms=1, max_width=900, max_height=900)
                if not show_image(overlay(frame.image, result), "perception", wait_ms=wait_ms):
                    break
            elif args.live:
                time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        if live is not None:
            live.close()
        if args.debug:
            close_windows()


if __name__ == "__main__":
    main()
