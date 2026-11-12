"""Mirrors driveaware-app TemporalEngineTest.kt so the replay matches the phone."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from driveaware_data.signals import (Baseline, Calibration, Frame, TemporalEngine,  # noqa: E402
                                     head_pose, mean_ear)

BASE = Baseline(ear=0.30)


def frame(t, ear=0.30, mar=0.1, yaw=0.0, pitch=0.0, blink=0.0):
    return Frame(t, True, ear, mar, 0.0, yaw, pitch, eye_blink=blink)


def run(e, start, end, f):
    out = []
    for t in range(start, end, 33):
        out += e.process(f(t))
    return out


def test_long_closure_is_level3_at_one_second():
    e = TemporalEngine(base=BASE)
    ev = [x for x in run(e, 0, 2000, lambda t: frame(t, ear=0.10)) if x.type == "eye_closure_long"]
    assert len(ev) == 1 and ev[0].level == 3 and 1.0 <= ev[0].t_s <= 1.1


def test_blinks_do_not_alert():
    e = TemporalEngine(base=BASE)
    ev = run(e, 0, 5000, lambda t: frame(t, ear=0.1 if t % 1000 < 150 else 0.3))
    assert not [x for x in ev if x.type == "eye_closure_long"]
    assert e.state.blinks >= 4


def test_mirror_check_no_alert_long_gaze_alerts():
    e = TemporalEngine(base=BASE)
    assert run(e, 0, 1000, lambda t: frame(t, yaw=45)) == []
    run(e, 1000, 1500, lambda t: frame(t))
    ev = run(e, 1500, 4000, lambda t: frame(t, yaw=45))
    assert [x.type for x in ev] == ["gaze_away"]


def test_yawn_and_nod_alone_level2():
    e = TemporalEngine(base=BASE)
    y = run(e, 0, 2000, lambda t: frame(t, mar=0.8))
    assert [x.type for x in y] == ["yawn"]
    n = run(e, 2000, 3000, lambda t: frame(t, pitch=-25))
    assert [(x.type, x.level) for x in n] == [("nod", 2)]


def test_perclos_needs_full_window():
    e = TemporalEngine(base=BASE)
    ev = run(e, 0, 70_000, lambda t: frame(t, ear=0.1 if t % 1500 < 300 else 0.3))
    p = [x for x in ev if x.type == "perclos_high"]
    assert p and p[0].t_s >= 54


def test_calibration_ignores_blinks():
    c = Calibration()
    for t in range(0, 10_100, 33):
        c.add(frame(t, ear=0.1 if t % 2000 < 150 else 0.31, yaw=2, pitch=-3))
    b = c.result()
    assert abs(b.ear - 0.31) < 1e-6 and b.yaw_deg == 2 and b.pitch_deg == -3


def test_geometry():
    lm = [(0.5, 0.5, 0)] * 478
    lm = list(lm)
    for idx, (x, y) in zip((33, 160, 158, 133, 153, 144), [(0.40, .5), (.42, .49), (.44, .49), (.46, .5), (.44, .51), (.42, .51)]):
        lm[idx] = (x, y, 0)
    for idx, (x, y) in zip((362, 385, 387, 263, 373, 380), [(.54, .5), (.56, .49), (.58, .49), (.60, .5), (.58, .51), (.56, .51)]):
        lm[idx] = (x, y, 0)
    assert 0.1 < mean_ear(lm) < 0.4
    yaw, pitch, roll = head_pose(np.eye(4))
    assert (round(yaw), round(pitch), round(roll)) == (0, 0, 0)


def test_blink_score_counts_quick_blinks_ear_misses():
    e = TemporalEngine(base=BASE)
    run(e, 0, 5000, lambda t: frame(t, ear=0.26, blink=0.8) if t % 1000 < 150 else frame(t))
    assert e.state.blinks >= 4


def test_no_eye_closure_alert_during_yawn():
    e = TemporalEngine(base=BASE)
    ev = run(e, 0, 2500, lambda t: frame(t, ear=0.10, mar=0.8))
    assert [x.type for x in ev] == ["yawn"]
