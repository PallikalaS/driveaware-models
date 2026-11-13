import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import tune_thresholds as tt  # noqa: E402
from driveaware_data.signals import Frame, Thresholds  # noqa: E402


def session(sun=False):
    frames = []
    for t in range(0, 60_000, 66):              # 15 fps, 60 s
        mar = 0.52 if 30_000 <= t < 32_500 else 0.05   # a modest yawn (MAR 0.52 < default 0.60)
        ear = 0.08 if 45_000 <= t < 46_500 else 0.30    # 1.5 s eye closure
        frames.append(Frame(t, True, ear=ear, mar=mar, jaw_open=0.0, eye_blink=0.0))
    base, start = tt.calibrate(frames, 10.0)
    cond = "day-sun" if sun else "day-none"
    script = pd.DataFrame([
        ["V01", "v.mp4", "bindu", cond, 1, 0, 15, "none", "intro"],
        ["V01", "v.mp4", "bindu", cond, 2, 30, 33, "yawn", ""],
        ["V01", "v.mp4", "bindu", cond, 3, 45, 47, "eye_closure_long", ""],
    ], columns=match_events_cols())
    return {"clip": "v.mp4", "script": script, "frames": frames, "start": start, "base": base, "sun": sun}


def match_events_cols():
    import match_events
    return match_events.SCRIPT_COLS


def test_tuning_finds_a_yawn_threshold_that_catches_the_scripted_yawn():
    s = [session()]
    default = tt.score(s, Thresholds(), 1.0, 3.0)
    assert default["yawn_recall"] == 0.0 and default["eye_closure_long_recall"] == 1.0
    table, best_th, best = tt.tune(s, Thresholds(), rounds=1, pre=1.0, post=3.0)
    assert best["yawn_recall"] == 1.0 and best["false_alarms"] == 0
    assert best_th.yawn_mar <= 0.50 or best_th.yawn_jaw_open <= 0.0
    assert len(table) == 1 + sum(len(v) for v in tt.GRID.values())


def test_sunglasses_clip_does_not_score_eye_events():
    r = tt.score([session(sun=True)], Thresholds(), 1.0, 3.0)
    assert r["eye_closure_long_recall"] == "" and r["false_alarms"] == 0


def test_app_setting_names():
    assert tt.app_name("ear_closed_ratio") == "earClosedRatio"
    assert tt.app_name("yawn_min_s") == "yawnMinS"
