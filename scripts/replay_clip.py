#!/usr/bin/env python3
"""Replay recorded clips through Model C + the app's rules, offline.

Writes, per clip, into --out:
  <clip>_frames.csv   per-frame EAR, MAR, jaw, yaw/pitch, PERCLOS, … (plot or eyeball)
  <clip>_log.csv      events in the app's event-log format -> eval/match_events.py

Threshold tuning (Vanessa, validation clips only — never S01–S08):
  python scripts/replay_clip.py ~/Drive/DriveAware/Team_validation/V0*.mp4 \
      --model face_landmarker.task --out results/replay
  python eval/match_events.py --script-dir ../driveaware-app/tests/validation \
      --log-dir results/replay --out results/validation_match
  # change a threshold and re-run, e.g. --set long_closure_s=0.8 --set gaze_max_s=2.5

Name the logs like the CSVs: V01_bindu_day-none_t1.mp4 -> V01_bindu_day-none_t1_log.csv,
which match_events pairs with V01.csv (prefix match on the session id).
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict, fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from driveaware_data.replay import FaceModel, replay_clip  # noqa: E402
from driveaware_data.signals import Thresholds  # noqa: E402

LOG_FIELDS = ["t_s", "event_type", "level", "source", "mode", "model_a_class", "model_a_conf",
              "gemini_event", "gemini_confirmed", "severity", "latency_ms", "fps"]


def parse_overrides(pairs: list[str]) -> Thresholds:
    th = Thresholds()
    names = {f.name for f in fields(Thresholds)}
    for p in pairs:
        k, _, v = p.partition("=")
        if k not in names:
            raise SystemExit(f"unknown threshold {k!r}; choose from {sorted(names)}")
        setattr(th, k, float(v))
    return th


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--model", required=True, help="face_landmarker.task")
    ap.add_argument("--out", default="results/replay")
    ap.add_argument("--fps", type=float, default=15.0, help="processing rate (phone-like)")
    ap.add_argument("--calib-s", type=float, default=10.0, help="seconds of calibration at clip start")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE", help="override a threshold")
    ap.add_argument("--pitch-sign", type=float, default=1.0, choices=[1.0, -1.0])
    a = ap.parse_args(argv)

    th = parse_overrides(a.set)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    face = FaceModel(a.model)
    for clip in a.clips:
        stem = Path(clip).stem
        try:
            r = replay_clip(clip, face, th, fps=a.fps, calib_s=a.calib_s, pitch_sign=a.pitch_sign)
        except RuntimeError as e:
            print(f"! {e}")
            continue
        with (out / f"{stem}_frames.csv").open("w", newline="") as f:
            keys = list(dict.fromkeys(k for row in r.frames for k in row))  # first-seen order
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(r.frames)
        with (out / f"{stem}_log.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            w.writeheader()
            for e in r.events:
                w.writerow({"t_s": f"{e.event.t_s:.2f}", "event_type": e.event.type, "level": e.event.level,
                            "source": "replay", "mode": "rules", "fps": f"{a.fps:.1f}"})
        face_pct = 100 * sum(row["face"] for row in r.frames) / max(1, len(r.frames))
        b = r.baseline
        print(f"{stem}: {len(r.events)} events, face found {face_pct:.0f}% of frames, "
              f"baseline EAR {b.ear:.3f} yaw {b.yaw_deg:.0f} pitch {b.pitch_deg:.0f}")
        for e in r.events:
            print(f"   {e.event.t_s:7.2f}s  L{e.event.level}  {e.event.type:16s} {e.event.detail}")
    print(f"\nthresholds: {asdict(th)}")


if __name__ == "__main__":
    main()
