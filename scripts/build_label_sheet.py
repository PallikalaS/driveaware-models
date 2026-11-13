#!/usr/bin/env python3
"""Build the Gemini label sheet (Bindu's ~250 rows) from team clips.

For every clip it replays Model C + the app's rules (same as the phone), and for
each triggered event writes one row with a 512 px keyframe and the exact signal
line the backend will send to Gemini. Bindu then fills event / confirmed /
severity / message / action using labeling_guide.md.

Row sources (column trigger_source):
  rules            eye_closure_long, perclos_high, yawn, nod, gaze_away from the replay
  clip_label       Model A activity triggers taken from the clip name
                   (phone_use / drinking / smoking / face_touch) until model_a_v1.tflite
                   exists; model_a conf is left at 0.00 and marked here, so these rows
                   must be regenerated with real Model A outputs before tuning
  confusion        a phone_use trigger on a frame from a non-phone clip (safe, face
                   touch, drinking) — candidate false alarms for Model A mistakes

Splits are by person: held-out members (default mahitha, vanessa) are test only;
--val-subjects go to val; everyone else is train.

  python scripts/build_label_sheet.py \
      ~/Drive/DriveAware/Team_raw ~/Drive/DriveAware/Team_test_only ~/Drive/DriveAware/Team_validation \
      --model face_landmarker.task --out ~/Drive/DriveAware/Gemini_events
Then in Google Sheets: File -> Import -> Upload label_sheet.csv; upload keyframes/ to the same
Drive folder so labelers can open each keyframe by name.
"""
from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from collections import Counter
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from driveaware_data.gemini_rows import SHEET_COLUMNS, SignalSummary, build_prompt  # noqa: E402
from driveaware_data.labels import find_videos  # noqa: E402
from driveaware_data.labels import CONDITIONS, SUBCLASS_TO_APP, SUBCLASSES, assign_split  # noqa: E402
from driveaware_data.replay import FaceModel, iter_frames, replay_clip  # noqa: E402

COND_RE = "|".join(re.escape(c) for c in CONDITIONS)
SESSION_RE = re.compile(rf"^(?P<sid>[sv]\d\d)_(?P<name>[a-z]+)_(?P<cond>{COND_RE})(_t\d+)?$")
ACTIVITY_RE = re.compile(rf"^(?P<name>[a-z]+)_(?P<act>[a-z_]+?)_(?P<cond>{COND_RE})_t\d+$")
LOOSE_RE = re.compile(rf"^(?P<name>[a-z]+)_.*?(?P<cond>{COND_RE})")
KEYFRAME_W = 512


def describe_clip(path: Path) -> dict:
    """subject, condition and (for activity clips) the scripted subclass."""
    stem = path.stem.lower()
    if m := SESSION_RE.match(stem):
        return {"subject": m["name"], "condition": m["cond"], "activity": None}
    if (m := ACTIVITY_RE.match(stem)) and m["act"] in SUBCLASSES:
        return {"subject": m["name"], "condition": m["cond"], "activity": m["act"]}
    if m := LOOSE_RE.match(stem):  # e.g. bindu_drowsy_day-none_t1
        return {"subject": m["name"], "condition": m["cond"], "activity": None}
    return {"subject": stem.split("_")[0], "condition": "", "activity": None}


def save_keyframe(img, out_dir: Path, name: str) -> str:
    h, w = img.shape[:2]
    small = cv2.resize(img, (KEYFRAME_W, round(h * KEYFRAME_W / w)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out_dir / name), small, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return name


def frame_at(path: Path, t_s: float):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_MSEC, t_s * 1000)
    ok, img = cap.read()
    cap.release()
    return img if ok else None


def summary_from_state(state, sunglasses: bool, model_a=None, hand_occupied=False) -> SignalSummary:
    cls, conf = model_a if model_a else (None, None)
    return SignalSummary(
        perclos=state.perclos, blinks=state.blinks, longest_closure_s=state.longest_closure_s,
        yawns_5min=state.yawns_5min, nods_5min=state.nods_5min, gaze_away_s=state.gaze_away_s,
        yaw_deg=state.yaw_delta, pitch_deg=state.pitch_delta,
        model_a_class=cls, model_a_conf=conf, hand_occupied=hand_occupied, sunglasses=sunglasses)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="clip files or folders of .mp4")
    ap.add_argument("--model", required=True, help="face_landmarker.task")
    ap.add_argument("--out", default="Gemini_events")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--calib-s", type=float, default=3.0,
                    help="calibration seconds at clip start (team clips start with 3 s of safe driving)")
    ap.add_argument("--activity-every-s", type=float, default=8.0, help="spacing of clip_label rows")
    ap.add_argument("--max-activity-rows", type=int, default=4, help="per activity clip")
    ap.add_argument("--confusion-rows", type=int, default=2, help="per non-phone clip")
    ap.add_argument("--held-out", default="mahitha,vanessa")
    ap.add_argument("--val-subjects", default="")
    ap.add_argument("--val-dir", default="",
                    help="clips under this folder go to val (e.g. Team_validation: Bindu's V01-V03), "
                         "so Gemini tuning has a validation set even with only two training people")
    ap.add_argument("--val-frac", type=float, default=0.0,
                    help="put this share of each training person's clips (whole clips, chosen by seed) "
                         "in val; use when Team_validation is empty")
    ap.add_argument("--id-prefix", default="r",
                    help="row_id prefix; use a new one (e.g. b) when adding clips to an existing sheet")
    ap.add_argument("--seed", type=int, default=663)
    a = ap.parse_args(argv)

    rng = random.Random(a.seed)
    held = {s.strip() for s in a.held_out.split(",") if s.strip()}
    val = {s.strip() for s in a.val_subjects.split(",") if s.strip()}
    missing = [x for x in a.inputs if not Path(x).expanduser().exists()]
    if missing:
        ap.error(f"input not found: {missing}. Check the folder path (an empty $D gives '/Team_raw').")
    out = Path(a.out).expanduser()
    kdir = out / "keyframes"
    kdir.mkdir(parents=True, exist_ok=True)
    clips = []
    for p in map(lambda x: Path(x).expanduser(), a.inputs):
        clips += find_videos(p, recursive=True) if p.is_dir() else [p]
    val_dir = Path(a.val_dir).expanduser().resolve() if a.val_dir else None
    # whole clips to val (never frames of the same clip in two splits), per person, reproducible
    val_clips: set[Path] = set()
    if a.val_frac > 0:
        by_person: dict[str, list[Path]] = {}
        for c in clips:
            try:
                subj = describe_clip(c)["subject"]
            except Exception:
                continue
            in_test_dir = any(p.name.lower() == "team_test_only" for p in c.resolve().parents)
            if assign_split(subj, held, val) == "train" and not in_test_dir:
                by_person.setdefault(subj, []).append(c)
        for subj, cs in sorted(by_person.items()):
            cs = sorted(cs)
            random.Random(f"{a.seed}-{subj}").shuffle(cs)
            val_clips.update(cs[: max(1, round(len(cs) * a.val_frac))])

    def split_of(clip: Path, subject: str) -> str:
        # Anything inside a Team_test_only folder is test, whatever its name says
        # (Vanessa's pilot clips are named S01.mp4 ... with no person in the name).
        if any(p.name.lower() == "team_test_only" for p in clip.resolve().parents):
            return "test"
        s = assign_split(subject, held, val)
        if s == "train" and (clip in val_clips or (val_dir and val_dir in clip.resolve().parents)):
            return "val"
        return s

    if not clips:
        ap.error(f"no .mp4/.mov clips found in {a.inputs}")
    print(f"{len(clips)} clips")
    known = held | val | {"swathi", "bindu", "mahitha", "vanessa"}
    odd = sorted({c.name for c in clips if describe_clip(c)["subject"] not in known})
    if odd:
        print(f"! {len(odd)} clips with no team member in the name (split by folder only): {odd[:6]}")
    face = FaceModel(a.model)
    rows = []

    def add(clip, info, t_s, trigger, level, source, img, summary):
        rid = f"{a.id_prefix}{len(rows) + 1:04d}"
        rows.append({
            "row_id": rid, "split": split_of(clip, info["subject"]), "subject": info["subject"],
            "clip": clip.name, "condition": info["condition"], "t_s": f"{t_s:.2f}", "trigger": trigger,
            "level": level, "trigger_source": source,
            "keyframe": save_keyframe(img, kdir, f"{rid}_{clip.stem}_{t_s:06.1f}s.jpg"),
            "signal_line": build_prompt(trigger, level, summary),
        })

    for clip in clips:
        info = describe_clip(clip)
        sun = info["condition"].endswith("-sun")
        print(f"{clip.name}: subject={info['subject']} condition={info['condition'] or '?'} "
              f"activity={info['activity'] or '-'}")
        try:
            r = replay_clip(str(clip), face, fps=a.fps, calib_s=a.calib_s)
        except RuntimeError as e:
            print(f"  ! skipped: {e}")
            continue

        # 1) rule-triggered events
        for e in r.events:
            add(clip, info, e.event.t_s, e.event.type, e.event.level, "rules", e.keyframe,
                summary_from_state(e.state, sun))

        # engine state nearest to a time, for activity rows (yawns / PERCLOS context)
        monitored = [f for f in r.frames if "perclos" in f]

        def state_at(t):
            from driveaware_data.signals import State
            f = min(monitored, key=lambda x: abs(x["t_s"] - t)) if monitored else {}
            return State(face_found=bool(f.get("face")), perclos=f.get("perclos", 0.0),
                         blinks=f.get("blinks10s", 0), yawns_5min=f.get("yawns5m", 0),
                         nods_5min=f.get("nods5m", 0), gaze_away_s=f.get("gaze_s", 0.0),
                         yaw_delta=f.get("yaw_d", 0.0), pitch_delta=f.get("pitch_d", 0.0))

        dur = r.frames[-1]["t_s"] if r.frames else 0
        act = info["activity"]
        # 2) Model A stand-in: activity clips -> app-class trigger rows
        if act and act != "safe":
            app_cls = SUBCLASS_TO_APP[act]
            times = [t for t in frange(a.calib_s + 2, dur - 1, a.activity_every_s)][: a.max_activity_rows]
            for t in times:
                img = frame_at(clip, t)
                if img is not None:
                    add(clip, info, t, app_cls, 2 if app_cls == "phone_use" else 1, "clip_label", img,
                        summary_from_state(state_at(t), sun, (app_cls, 0.0),
                                           hand_occupied=app_cls in ("phone_use", "drinking", "smoking")))
        # 3) candidate false alarms: phone_use trigger on non-phone frames
        if act and SUBCLASS_TO_APP[act] != "phone_use" and dur > a.calib_s + 3:
            for t in sorted(rng.uniform(a.calib_s + 1, dur - 1) for _ in range(a.confusion_rows)):
                img = frame_at(clip, t)
                if img is not None:
                    add(clip, info, t, "phone_use", 2, "confusion", img,
                        summary_from_state(state_at(t), sun, ("phone_use", 0.0)))

    sheet = out / ("label_sheet.csv" if a.id_prefix == "r" else f"label_sheet_{a.id_prefix}.csv")
    with sheet.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SHEET_COLUMNS)
        w.writeheader()
        w.writerows(rows)

    print(f"\n{len(rows)} rows -> {sheet}   keyframes -> {kdir}")
    print("by split:  ", dict(Counter(r["split"] for r in rows)))
    print("by trigger:", dict(Counter(r["trigger"] for r in rows)))
    print("by source: ", dict(Counter(r["trigger_source"] for r in rows)))
    print("Target (To-Dos plan): ~250 rows, >= 25 false_alarm, >= 15 of each event type; "
          "~170 train / 40 val / 40 test.")


def frange(start, stop, step):
    t = start
    while t < stop:
        yield t
        t += step


if __name__ == "__main__":
    main()
