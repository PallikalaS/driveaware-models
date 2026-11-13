"""End-to-end plumbing test with a stub face model (no MediaPipe model file needed).

The stub 'sees' eyes closed from 20.0-21.5 s and a yawn at 30-32 s; the real
FaceModel is only swapped out, everything else (sampling, calibration, engine,
CSV / keyframe writing) is the production code path.
"""
import csv
import sys
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_label_sheet  # noqa: E402
from driveaware_data import replay as replay_mod  # noqa: E402
from driveaware_data.signals import Frame  # noqa: E402


class StubFace:
    def __init__(self, *_):
        pass

    @contextmanager
    def open(self):
        yield None

    def frame_signals(self, _lmk, _bgr, t_ms):
        t = t_ms / 1000
        ear = 0.08 if 20.0 <= t < 21.5 else 0.30
        m = 0.8 if 30.0 <= t < 32.0 else 0.1
        return Frame(t_ms, True, ear, m, 0.0, 0.0, 0.0)


def make_clip(path: Path, seconds=40, fps=30):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 240))
    for i in range(seconds * fps):
        img = np.full((240, 320, 3), 80, np.uint8)
        cv2.putText(img, f"{i / fps:.1f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        w.write(img)
    w.release()


def test_replay_finds_scripted_events(tmp_path):
    clip = tmp_path / "V01_bindu_day-none_t1.mp4"
    make_clip(clip)
    r = replay_mod.replay_clip(str(clip), StubFace(), fps=15, calib_s=10)
    kinds = [(e.event.type, e.event.level, round(e.event.t_s)) for e in r.events]
    assert ("eye_closure_long", 3, 21) in kinds
    assert any(k[0] == "yawn" and 31 <= k[2] <= 32 for k in kinds)
    assert abs(r.baseline.ear - 0.30) < 1e-6


def test_label_sheet_end_to_end(tmp_path, monkeypatch):
    clips = tmp_path / "clips"
    clips.mkdir()
    make_clip(clips / "V01_bindu_day-none_t1.mp4")
    make_clip(clips / "mahitha_texting_day-sun_t1.mp4", seconds=20)
    make_clip(clips / "swathi_drinking_cup_night-none_t1.mp4", seconds=20)
    monkeypatch.setattr(build_label_sheet, "FaceModel", StubFace)
    out = tmp_path / "out"
    build_label_sheet.main([str(clips), "--model", "x", "--out", str(out), "--calib-s", "3"])
    rows = list(csv.DictReader((out / "label_sheet.csv").open()))
    assert rows and all((out / "keyframes" / r["keyframe"]).exists() for r in rows)
    by = {(r["clip"], r["trigger_source"], r["trigger"]) for r in rows}
    assert ("V01_bindu_day-none_t1.mp4", "rules", "eye_closure_long") in by
    assert ("mahitha_texting_day-sun_t1.mp4", "clip_label", "phone_use") in by
    assert ("swathi_drinking_cup_night-none_t1.mp4", "confusion", "phone_use") in by
    test_rows = [r for r in rows if r["subject"] == "mahitha"]
    assert test_rows and all(r["split"] == "test" for r in test_rows)
    sun = next(r for r in rows if r["clip"].startswith("mahitha"))
    assert sun["signal_line"].endswith("sunglasses=yes")
    assert all(r["event"] == "" for r in rows)  # left for the labeler


def test_label_sheet_val_dir_and_prefix(tmp_path, monkeypatch):
    raw, valdir = tmp_path / "Team_raw", tmp_path / "Team_validation"
    raw.mkdir(); valdir.mkdir()
    make_clip(raw / "bindu_safe_day-none_t1.mp4")
    make_clip(valdir / "V01_bindu_day-none_t1.mp4")
    monkeypatch.setattr(build_label_sheet, "FaceModel", StubFace)
    out = tmp_path / "out"
    build_label_sheet.main([str(raw), str(valdir), "--model", "x", "--out", str(out), "--calib-s", "3",
                            "--val-dir", str(valdir), "--id-prefix", "b"])
    rows = list(csv.DictReader((out / "label_sheet_b.csv").open()))
    assert rows and all(r["row_id"].startswith("b") for r in rows)
    assert {r["split"] for r in rows if r["clip"].startswith("V01")} == {"val"}
    assert {r["split"] for r in rows if r["clip"].startswith("bindu")} <= {"train"}
    assert not (out / "label_sheet.csv").exists()


def test_label_sheet_val_frac_whole_clips_per_person(tmp_path, monkeypatch):
    raw = tmp_path / "Team_raw"; raw.mkdir()
    for who in ("swathi", "bindu"):
        for k in range(1, 5):
            make_clip(raw / f"{who}_safe_day-none_t{k}.mp4")
    monkeypatch.setattr(build_label_sheet, "FaceModel", StubFace)
    out = tmp_path / "out"
    build_label_sheet.main([str(raw), "--model", "x", "--out", str(out), "--calib-s", "3", "--val-frac", "0.25"])
    rows = list(csv.DictReader((out / "label_sheet.csv").open()))
    split_by_clip = {}
    for r in rows:
        split_by_clip.setdefault(r["clip"], set()).add(r["split"])
    assert all(len(s) == 1 for s in split_by_clip.values())       # a clip never spans splits
    for who in ("swathi", "bindu"):                                # 1 of 4 clips each in val
        clips = {c for c in split_by_clip if c.startswith(who)}
        assert sum(split_by_clip[c] == {"val"} for c in clips) == 1


def test_team_test_only_folder_is_always_test(tmp_path, monkeypatch):
    raw, test_dir = tmp_path / "Team_raw", tmp_path / "Team_test_only" / "pilot"
    raw.mkdir(); test_dir.mkdir(parents=True)
    make_clip(raw / "bindu_safe_day-none_t1.mp4")
    make_clip(test_dir / "S01.mp4")                     # pilot clip with no name in it
    monkeypatch.setattr(build_label_sheet, "FaceModel", StubFace)
    out = tmp_path / "out"
    build_label_sheet.main([str(raw), str(tmp_path / "Team_test_only"), "--model", "x", "--out", str(out),
                            "--calib-s", "3", "--val-frac", "0.15"])
    rows = list(csv.DictReader((out / "label_sheet.csv").open()))
    assert {r["split"] for r in rows if r["clip"] == "S01.mp4"} == {"test"}
