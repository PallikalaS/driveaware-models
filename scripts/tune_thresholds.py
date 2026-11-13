#!/usr/bin/env python3
"""Threshold tuning on the validation clips (Vanessa #14, #15; app #36).

Runs the face model ONCE per V clip (cached as <clip>_signals.csv), then replays the
app's rules (driveaware_data/signals.py = TemporalEngine.kt) over the cached signals
for many threshold values and scores each against Vanessa's scripts
(driveaware-app tests/validation/V0x.csv) with the same matching as the system test.

One threshold changes at a time, starting from the app defaults (the plan's "change one
setting at a time"); with --rounds 2 the best values are kept and every threshold is
tried again. Validation clips only: never tune on held-out test sessions.

  python scripts/tune_thresholds.py ~/Projects/driveaware/data/Team_validation \\
      --scripts ~/Projects/driveaware/driveaware-app/tests/validation \\
      --model face_landmarker.task --out results/tuning

Writes:
  tuning_table.csv   every (threshold, value): recall per event, false alarms, macro-F1, delay
  thresholds.csv     start vs suggested value per threshold (+ the app's setting name)
  baseline_report.txt  per-event recall / false alarms with the defaults and with the suggestion

Only rule events are scored (eye_closure_long, perclos_high, yawn, nod, gaze_away);
Model A activities are not in the replay. On sunglasses clips (condition *-sun) the app
turns the eye rules off, so eye events are not scored there either.
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "eval"))
from driveaware_data.labels import find_videos  # noqa: E402
from driveaware_data.signals import Baseline, Calibration, Frame, TemporalEngine, Thresholds  # noqa: E402
import match_events  # noqa: E402

RULE_EVENTS = ["eye_closure_long", "perclos_high", "yawn", "nod", "gaze_away"]
EYE_EVENTS = {"eye_closure_long", "perclos_high"}
SIGNAL_COLS = ["t_ms", "face", "ear", "mar", "jaw", "yaw", "pitch", "roll", "blink"]

# candidates per threshold (app defaults are included in each list)
GRID = {
    "ear_closed_ratio": [0.60, 0.65, 0.70, 0.75, 0.80],
    "blink_score_closed": [0.40, 0.50, 0.60],
    "long_closure_s": [0.8, 1.0, 1.2],
    "perclos_high": [0.10, 0.12, 0.15, 0.20],
    "yawn_mar": [0.45, 0.50, 0.55, 0.60, 0.70],
    "yawn_jaw_open": [0.35, 0.45, 0.55, 0.65],
    "yawn_min_s": [0.8, 1.0, 1.5, 2.0],
    "nod_pitch_deg": [10.0, 12.0, 15.0, 20.0],
    "gaze_yaw_deg": [20.0, 25.0, 30.0, 35.0],
    "gaze_max_s": [1.5, 2.0, 2.5],
}


def app_name(py: str) -> str:
    """ear_closed_ratio -> earClosedRatio (the field name in Thresholds.kt / Settings)."""
    head, *rest = py.split("_")
    return head + "".join(w.capitalize() for w in rest)


# ------------------------------------------------------------------ signals cache
def extract_signals(clip: Path, model: str, fps: float, cache_dir: Path) -> pd.DataFrame:
    cache = cache_dir / f"{clip.stem}_signals.csv"
    if cache.exists():
        return pd.read_csv(cache)
    from driveaware_data.replay import FaceModel, iter_frames, PITCH_SIGN
    face = FaceModel(model)
    rows = []
    with face.open() as lmk:
        for t_ms, frame, _ in iter_frames(str(clip), fps):
            f = face.frame_signals(lmk, frame, t_ms)
            rows.append([t_ms, int(f.face_found), f.ear, f.mar, f.jaw_open, f.yaw, f.pitch * PITCH_SIGN,
                         f.roll, f.eye_blink])
    df = pd.DataFrame(rows, columns=SIGNAL_COLS)
    df.to_csv(cache, index=False)
    return df


def to_frames(df: pd.DataFrame) -> list[Frame]:
    return [Frame(int(r.t_ms), bool(r.face), r.ear, r.mar, r.jaw, r.yaw, r.pitch, r.roll, r.blink)
            for r in df.itertuples()]


def calibrate(frames: list[Frame], calib_s: float) -> tuple[Baseline, int]:
    cal = Calibration(duration_ms=int(calib_s * 1000))
    for i, f in enumerate(frames):
        cal.add(f)
        if cal.done(f.t_ms):
            b = cal.result()
            if b is None:
                raise RuntimeError("face found in <60% of the calibration seconds")
            return b, i + 1
    raise RuntimeError("clip shorter than the calibration time")


# ------------------------------------------------------------------ scoring
def run_rules(frames: list[Frame], start: int, base: Baseline, th: Thresholds) -> pd.DataFrame:
    eng = TemporalEngine(th, base)
    out = [(e.t_s, e.type, e.level) for f in frames[start:] for e in eng.process(f)]
    return pd.DataFrame(out, columns=["t_s", "event_type", "level"])


def score(sessions: list[dict], th: Thresholds, pre: float, post: float) -> dict:
    per = {e: {"scripted": 0, "detected": 0, "alerts": 0, "delays": []} for e in RULE_EVENTS}
    false_alarms, minutes = 0, 0.0
    for s in sessions:
        log = run_rules(s["frames"], s["start"], s["base"], th)
        script = s["script"]
        if s["sun"]:
            log = log[~log.event_type.isin(EYE_EVENTS)]
            script = script[~script.event_type.isin(EYE_EVENTS)]
        log = log.reset_index(drop=True)
        ev, alerts = match_events.match_session(script, log, pre, post)
        minutes += (s["frames"][-1].t_ms - s["frames"][s["start"]].t_ms) / 60000
        for e in RULE_EVENTS:
            sub = ev[ev.event_type == e] if len(ev) else ev
            per[e]["scripted"] += len(sub)
            per[e]["detected"] += int(sub["detected"].fillna(False).astype(bool).sum()) if len(sub) else 0
            per[e]["delays"] += [d for d in sub.get("delay_s", pd.Series(dtype=float)).dropna()]
            per[e]["alerts"] += int((alerts.event_type == e).sum()) if len(alerts) else 0
        false_alarms += int((~alerts.matched).sum()) if len(alerts) else 0
    f1s = []
    row = {}
    for e, p in per.items():
        rec = p["detected"] / p["scripted"] if p["scripted"] else math.nan
        prec = p["detected"] / p["alerts"] if p["alerts"] else (math.nan if not p["scripted"] else 0.0)
        row[f"{e}_recall"] = round(rec, 3) if not math.isnan(rec) else ""
        if p["scripted"]:
            f1s.append(0.0 if not (rec and prec) or math.isnan(prec) else 2 * rec * prec / (rec + prec))
    delays = [d for p in per.values() for d in p["delays"]]
    row.update({"false_alarms": false_alarms,
                "fa_per_10min": round(10 * false_alarms / minutes, 2) if minutes else "",
                "macro_f1": round(statistics.mean(f1s), 4) if f1s else 0.0,
                "median_delay_s": round(statistics.median(delays), 2) if delays else ""})
    return row


def better(a: dict, b: dict) -> bool:
    """Higher macro-F1 wins; on a tie, fewer false alarms, then shorter delay."""
    ka = (a["macro_f1"], -a["false_alarms"], -(a["median_delay_s"] or 99))
    kb = (b["macro_f1"], -b["false_alarms"], -(b["median_delay_s"] or 99))
    return ka > kb


def tune(sessions, start: Thresholds, rounds: int, pre: float, post: float):
    table, best_th = [], start
    best = score(sessions, best_th, pre, post)
    table.append({"round": 0, "threshold": "(defaults)", "value": "", **best})
    for r in range(1, rounds + 1):
        for name, values in GRID.items():
            for v in values:
                th = replace(best_th, **{name: v})
                res = score(sessions, th, pre, post)
                table.append({"round": r, "threshold": name, "value": v, **res})
                if better(res, best):
                    best, best_th = res, th
    return table, best_th, best


# ------------------------------------------------------------------ main
def load_sessions(clip_dir: Path, script_dir: Path, model: str, fps: float, calib_s: float, cache: Path):
    scripts = {p.stem.upper(): match_events.load_script(p) for p in sorted(script_dir.glob("*.csv"))}
    sessions = []
    for clip in find_videos(clip_dir):
        sid = clip.stem.split("_")[0].upper()
        if sid not in scripts:
            print(f"  skip {clip.name}: no script {sid}.csv in {script_dir}")
            continue
        sig = extract_signals(clip, model, fps, cache)
        frames = to_frames(sig)
        try:
            base, start = calibrate(frames, calib_s)
        except RuntimeError as e:
            print(f"  skip {clip.name}: {e}")
            continue
        cond = str(scripts[sid]["condition"].iloc[0])
        sessions.append({"clip": clip.name, "script": scripts[sid], "frames": frames, "start": start,
                         "base": base, "sun": cond.endswith("-sun")})
        print(f"  {clip.name}: {len(frames)} frames, baseline EAR {base.ear:.3f}, {cond}")
    return sessions


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips", help="folder with V01..V03 clips")
    ap.add_argument("--scripts", required=True, help="driveaware-app/tests/validation")
    ap.add_argument("--model", required=True, help="face_landmarker.task")
    ap.add_argument("--out", default="results/tuning")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--calib-s", type=float, default=10.0, help="V scripts start with 15 s looking forward")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--pre", type=float, default=1.0)
    ap.add_argument("--post", type=float, default=3.0)
    a = ap.parse_args(argv)

    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    sessions = load_sessions(Path(a.clips).expanduser(), Path(a.scripts).expanduser(), a.model, a.fps,
                             a.calib_s, out)
    if not sessions:
        print("no usable validation clips"); return 1
    start = Thresholds()
    table, best_th, best = tune(sessions, start, a.rounds, a.pre, a.post)

    pd.DataFrame(table).to_csv(out / "tuning_table.csv", index=False)
    with open(out / "thresholds.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["threshold", "app_setting", "start", "suggested", "changed"])
        for k, v in asdict(start).items():
            nv = getattr(best_th, k)
            w.writerow([k, app_name(k), v, nv, "yes" if nv != v else ""])
    first = table[0]
    lines = ["metric                 defaults   suggested"]
    for k in [f"{e}_recall" for e in RULE_EVENTS] + ["false_alarms", "fa_per_10min", "macro_f1", "median_delay_s"]:
        lines.append(f"{k:22s} {str(first[k]):10s} {best[k]}")
    rep = "\n".join(lines)
    (out / "baseline_report.txt").write_text(rep + "\n")
    print("\n" + rep)
    changed = [f"{app_name(k)} {v} -> {getattr(best_th, k)}" for k, v in asdict(start).items()
               if getattr(best_th, k) != v]
    print("\nsuggested changes:", ", ".join(changed) or "none (defaults are best on these clips)")
    print(f"tables in {out}/: tuning_table.csv, thresholds.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
