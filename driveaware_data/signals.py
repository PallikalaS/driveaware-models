"""Python port of the app's signal rules, for replaying recorded clips.

Mirrors, line for line, in driveaware-app:
  android/.../signals/FaceSignals.kt     -> ear(), mar(), head_pose()
  android/.../signals/Thresholds.kt      -> Thresholds, Baseline
  android/.../signals/TemporalEngine.kt  -> TemporalEngine
  android/.../signals/Calibration.kt     -> Calibration

If you change a rule or a default in the Kotlin, change it here too (and the
other way round); tests/test_signals.py mirrors TemporalEngineTest.kt.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field, replace

# ------------------------------------------------------------------ geometry
LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)
LIP_UP, LIP_LOW, MOUTH_L, MOUTH_R = 13, 14, 78, 308


def _d(a, b, aspect):
    return math.hypot((a[0] - b[0]) * aspect, a[1] - b[1])


def ear(lm, idx, aspect=4 / 3):
    """Eye aspect ratio from normalized (x, y[, z]) landmarks."""
    p = [lm[i] for i in idx]
    horiz = _d(p[0], p[3], aspect)
    if horiz < 1e-6:
        return 0.0
    return (_d(p[1], p[5], aspect) + _d(p[2], p[4], aspect)) / (2 * horiz)


def mean_ear(lm, aspect=4 / 3):
    return (ear(lm, LEFT_EYE, aspect) + ear(lm, RIGHT_EYE, aspect)) / 2


def mar(lm, aspect=4 / 3):
    w = _d(lm[MOUTH_L], lm[MOUTH_R], aspect)
    return 0.0 if w < 1e-6 else _d(lm[LIP_UP], lm[LIP_LOW], aspect) / w


def head_pose(m):
    """(yaw, pitch, roll) in degrees from a 4x4 facial transformation matrix.

    Accepts a 4x4 nested list / numpy array (row-major, as MediaPipe Python
    returns it) — same formulas as FaceSignals.headPose on the Android side.
    Head DOWN = negative pitch (the rules' convention); MediaPipe's matrix gives
    head-down as positive (verified on video and the app overlay), so pitch is negated.
    """
    def r(row, col):
        return float(m[row][col])
    sy = math.sqrt(r(0, 0) ** 2 + r(1, 0) ** 2)
    pitch = -math.degrees(math.atan2(r(2, 1), r(2, 2)))
    yaw = math.degrees(math.atan2(-r(2, 0), sy))
    roll = math.degrees(math.atan2(r(1, 0), r(0, 0)))
    return yaw, pitch, roll


# ------------------------------------------------------------------ settings
@dataclass
class Thresholds:
    ear_closed_ratio: float = 0.60   # tuned on V01-V04 (start 0.70)
    blink_score_closed: float = 0.50   # eyeBlink blendshape >= this also counts as closed
    long_closure_s: float = 1.0
    severe_closure_s: float = 1.0      # labeling_guide: any closure >= 1.0 s is high
    blink_max_s: float = 0.5
    perclos_window_s: float = 60.0
    perclos_high: float = 0.20       # tuned (start 0.15)
    yawn_mar: float = 0.45           # tuned (start 0.60)
    yawn_jaw_open: float = 0.55
    yawn_min_s: float = 0.8          # tuned (start 1.5)
    nod_pitch_deg: float = 20.0      # tuned (start 15)
    nod_min_s: float = 0.4
    gaze_yaw_deg: float = 30.0
    gaze_pitch_deg: float = 20.0
    gaze_max_s: float = 2.0
    refractory_s: float = 10.0


@dataclass
class Baseline:
    ear: float = 0.28
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0


@dataclass
class Frame:
    t_ms: int
    face_found: bool
    ear: float = 0.0
    mar: float = 0.0
    jaw_open: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    eye_blink: float = 0.0   # mean of eyeBlinkLeft/Right blendshapes


@dataclass
class Event:
    type: str      # event CSV word: eye_closure_long, perclos_high, yawn, nod, gaze_away
    t_s: float
    level: int
    detail: str = ""


@dataclass
class State:
    face_found: bool = False
    eyes_closed: bool = False
    closure_s: float = 0.0
    perclos: float = 0.0
    blinks: int = 0
    longest_closure_s: float = 0.0
    yawns_5min: int = 0
    nods_5min: int = 0
    gaze_away_s: float = 0.0
    yaw_delta: float = 0.0
    pitch_delta: float = 0.0


# ------------------------------------------------------------------ engine
class TemporalEngine:
    def __init__(self, th: Thresholds | None = None, base: Baseline | None = None):
        self.th = th or Thresholds()
        self.base = base or Baseline()
        self.reset()

    def reset(self):
        self._perclos = deque()          # (t_ms, closed)
        self._yawns, self._nods, self._blinks = deque(), deque(), deque()
        self._last_fired: dict[str, int] = {}
        self._last_level: dict[str, int] = {}
        self._close_start = self._yawn_start = self._nod_start = self._gaze_start = None
        self._closure_fired = self._yawn_fired = self._nod_fired = self._gaze_fired = False
        self._last_mouth_open = -10**12   # eye-closure alert clock restarts after a yawn (app Issue #11)
        self._last_head_away = -10**12    # … and after the head was turned far away (app v0.9)
        self._longest_closure_ms = 0
        self.state = State()

    def reset_window_stats(self):
        self._longest_closure_ms = 0

    @staticmethod
    def _prune(q: deque, now: int, keep_ms: int):
        while q and now - q[0] > keep_ms:
            q.popleft()

    def _fire(self, typ, t, level, detail, refractory_s=None):
        refractory_s = self.th.refractory_s if refractory_s is None else refractory_s
        last = self._last_fired.get(typ)
        if last is not None and t - last < int(refractory_s * 1000):
            return None
        self._last_fired[typ] = t
        self._last_level[typ] = level
        return Event(typ, t / 1000, level, detail)

    def process(self, f: Frame) -> list[Event]:
        th, base, out = self.th, self.base, []
        t = f.t_ms
        if not f.face_found:
            self._close_start = self._yawn_start = self._nod_start = None
            self.state = replace(self.state, face_found=False)
            return out

        # mouth first: no eye-closure alert while yawning (eyes squeeze shut in a real yawn)
        mouth_open = f.mar >= th.yawn_mar or f.jaw_open >= th.yawn_jaw_open
        if mouth_open:
            self._last_mouth_open = t
        # head turned far up / down / sideways: eyes look narrow even when open (app v0.9)
        pd = f.pitch - base.pitch_deg
        head_away = abs(f.yaw - base.yaw_deg) >= th.gaze_yaw_deg or pd >= th.gaze_pitch_deg or pd <= -th.nod_pitch_deg
        if head_away:
            self._last_head_away = t

        # eyes: blink, long closure, PERCLOS (EAR or the eyeBlink blendshape)
        closed = f.ear < base.ear * th.ear_closed_ratio or f.eye_blink >= th.blink_score_closed
        if closed:
            if self._close_start is None:
                self._close_start = t
                self._closure_fired = False
                self._last_level.pop("eye_closure_long", None)
            dur = (t - max(self._close_start, self._last_mouth_open, self._last_head_away)) / 1000
            if not self._closure_fired and not mouth_open and not head_away and dur >= th.long_closure_s:
                self._closure_fired = True
                lvl = 3 if dur >= th.severe_closure_s else 2
                e = self._fire("eye_closure_long", t, lvl, f"closed {dur:.1f}s")
                if e:
                    out.append(e)
            if (self._closure_fired and not mouth_open and not head_away and dur >= th.severe_closure_s
                    and self._last_level.get("eye_closure_long", 0) < 3):
                self._last_level["eye_closure_long"] = 3
                out.append(Event("eye_closure_long", t / 1000, 3, f"closed {dur:.1f}s"))
        elif self._close_start is not None:
            dur_ms = t - self._close_start
            self._longest_closure_ms = max(self._longest_closure_ms, dur_ms)
            if dur_ms / 1000 <= th.blink_max_s:
                self._blinks.append(t)
            self._close_start = None

        self._perclos.append((t, closed))
        win_ms = int(th.perclos_window_s * 1000)
        while self._perclos and t - self._perclos[0][0] > win_ms:
            self._perclos.popleft()
        self._prune(self._blinks, t, 10_000)
        perclos = sum(1 for _, c in self._perclos if c) / len(self._perclos)
        window_full = len(self._perclos) > 1 and (t - self._perclos[0][0]) >= win_ms * 0.9
        if window_full and perclos >= th.perclos_high:
            e = self._fire("perclos_high", t, 2, f"perclos {perclos:.2f}", refractory_s=30)
            if e:
                out.append(e)

        # yawn
        if mouth_open:
            if self._yawn_start is None:
                self._yawn_start, self._yawn_fired = t, False
            if not self._yawn_fired and (t - self._yawn_start) / 1000 >= th.yawn_min_s:
                self._yawn_fired = True
                self._yawns.append(t)
                self._prune(self._yawns, t, 300_000)
                lvl = 2 if len(self._yawns) >= 3 else 1
                e = self._fire("yawn", t, lvl, f"yawns5min={len(self._yawns)}", refractory_s=3)
                if e:
                    out.append(e)
        else:
            self._yawn_start = None

        # head: nod and gaze-away
        pitch_d = f.pitch - base.pitch_deg
        yaw_d = f.yaw - base.yaw_deg
        if pitch_d <= -th.nod_pitch_deg:
            if self._nod_start is None:
                self._nod_start, self._nod_fired = t, False
            if not self._nod_fired and (t - self._nod_start) / 1000 >= th.nod_min_s:
                self._nod_fired = True
                self._nods.append(t)
                self._prune(self._nods, t, 300_000)
                lvl = 3 if (window_full and perclos >= th.perclos_high) else 2
                e = self._fire("nod", t, lvl, f"pitch {pitch_d:.0f}", refractory_s=3)
                if e:
                    out.append(e)
        else:
            self._nod_start = None

        gaze_s = 0.0
        if abs(yaw_d) >= th.gaze_yaw_deg or pitch_d >= th.gaze_pitch_deg:
            if self._gaze_start is None:
                self._gaze_start, self._gaze_fired = t, False
            gaze_s = (t - self._gaze_start) / 1000
            if not self._gaze_fired and gaze_s >= th.gaze_max_s:
                self._gaze_fired = True
                e = self._fire("gaze_away", t, 1, f"yaw {yaw_d:.0f}")
                if e:
                    out.append(e)
        else:
            self._gaze_start = None

        self._prune(self._yawns, t, 300_000)
        self._prune(self._nods, t, 300_000)
        open_ms = (t - self._close_start) if self._close_start is not None else 0
        self.state = State(
            face_found=True, eyes_closed=closed, closure_s=open_ms / 1000, perclos=perclos,
            blinks=len(self._blinks), longest_closure_s=max(self._longest_closure_ms, open_ms) / 1000,
            yawns_5min=len(self._yawns), nods_5min=len(self._nods), gaze_away_s=gaze_s,
            yaw_delta=yaw_d, pitch_delta=pitch_d,
        )
        return out


# ------------------------------------------------------------------ calibration
@dataclass
class Calibration:
    """Same as Calibration.kt: median baseline over the first `duration_ms` of face frames."""
    duration_ms: int = 10_000
    _ears: list = field(default_factory=list)
    _yaws: list = field(default_factory=list)
    _pitches: list = field(default_factory=list)
    _start: int | None = None
    face_frames: int = 0
    total_frames: int = 0

    def add(self, f: Frame):
        if self._start is None:
            self._start = f.t_ms
        self.total_frames += 1
        if f.face_found:
            self.face_frames += 1
            self._ears.append(f.ear); self._yaws.append(f.yaw); self._pitches.append(f.pitch)

    def done(self, now_ms: int) -> bool:
        return self._start is not None and now_ms - self._start >= self.duration_ms

    def result(self) -> Baseline | None:
        if self.total_frames == 0 or self.face_frames < self.total_frames * 0.6:
            return None
        med = lambda v: sorted(v)[len(v) // 2] if v else 0.0  # noqa: E731
        open_ears = sorted(self._ears)[len(self._ears) // 4:]
        return Baseline(ear=med(open_ears), yaw_deg=med(self._yaws), pitch_deg=med(self._pitches))
