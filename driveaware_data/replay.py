"""Replay a recorded clip through Model C + the app's rules (offline).

Same pipeline as the phone: MediaPipe Face Landmarker -> EAR / MAR / jawOpen /
head pose -> 10 s calibration -> TemporalEngine. Used by
  scripts/replay_clip.py        (threshold tuning on V01-V03, rule checks)
  scripts/build_label_sheet.py  (Gemini label-sheet rows + keyframes)

Needs the Face Landmarker model file (same one the app uses):
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from .signals import (Baseline, Calibration, Event, Frame, State, TemporalEngine, Thresholds,
                      head_pose, mar, mean_ear)


@dataclass
class ReplayEvent:
    event: Event
    state: State          # engine state right after the event (feeds the signal line)
    keyframe: np.ndarray  # BGR frame at the event


@dataclass
class ReplayResult:
    clip: str
    fps: float
    baseline: Baseline | None
    frames: list[dict] = field(default_factory=list)       # per processed frame
    events: list[ReplayEvent] = field(default_factory=list)


class FaceModel:
    """MediaPipe Face Landmarker in VIDEO mode (CPU), one clip at a time."""

    def __init__(self, model_path: str):
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"{model_path} not found. Download face_landmarker.task (see driveaware_data/replay.py)")
        self.model_path = model_path

    def open(self):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5, min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=True, output_facial_transformation_matrixes=True,
        )
        self._mp = mp
        return vision.FaceLandmarker.create_from_options(opts)

    def frame_signals(self, landmarker, bgr: np.ndarray, t_ms: int) -> Frame:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = landmarker.detect_for_video(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb), t_ms)
        if not res.face_landmarks:
            return Frame(t_ms, False)
        h, w = bgr.shape[:2]
        lm = [(p.x, p.y, p.z) for p in res.face_landmarks[0]]
        shapes = {c.category_name: c.score for c in res.face_blendshapes[0]} if res.face_blendshapes else {}
        jaw = shapes.get("jawOpen", 0.0)
        blink = (shapes.get("eyeBlinkLeft", 0.0) + shapes.get("eyeBlinkRight", 0.0)) / 2
        yaw = pitch = roll = 0.0
        if res.facial_transformation_matrixes:
            yaw, pitch, roll = head_pose(np.asarray(res.facial_transformation_matrixes[0]).reshape(4, 4))
        aspect = w / h
        return Frame(t_ms, True, mean_ear(lm, aspect), mar(lm, aspect), jaw, yaw, pitch, roll, blink)


def iter_frames(path: str, target_fps: float) -> Iterator[tuple[int, np.ndarray, float]]:
    """Yield (t_ms, frame, src_fps), sampled at ~target_fps (phone-like rate)."""
    cap = cv2.VideoCapture(path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(src_fps / target_fps)) if target_fps > 0 else 1
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            yield int(round(i / src_fps * 1000)), frame, src_fps
        i += 1
    cap.release()


# head_pose() already returns head-down as negative (same as the app's
# FaceSignals.headPose). Keep 1.0; -1 only for debugging a different camera setup.
PITCH_SIGN = 1.0


def replay_clip(path: str, face: FaceModel, th: Thresholds | None = None, fps: float = 15.0,
                calib_s: float = 10.0, baseline: Baseline | None = None,
                pitch_sign: float = PITCH_SIGN) -> ReplayResult:
    """Run one clip. The first `calib_s` seconds calibrate (like the app's Calibrate
    button) unless a `baseline` is given; events are only detected after that.
    Head pitch is multiplied by `pitch_sign` so head-down is negative (see PITCH_SIGN)."""
    res = ReplayResult(clip=Path(path).name, fps=fps, baseline=baseline)
    engine = TemporalEngine(th or Thresholds(), baseline or Baseline())
    cal = None if baseline else Calibration(duration_ms=int(calib_s * 1000))
    start_ms = None
    with face.open() as lmk:
        for t_ms, frame, _ in iter_frames(path, fps):
            f = face.frame_signals(lmk, frame, t_ms)
            f.pitch *= pitch_sign
            row = {"t_s": round(t_ms / 1000, 3), "face": int(f.face_found), "ear": round(f.ear, 4),
                   "blink": round(f.eye_blink, 3),
                   "mar": round(f.mar, 4), "jaw": round(f.jaw_open, 3), "yaw": round(f.yaw, 1),
                   "pitch": round(f.pitch, 1), "phase": "calibrating" if cal else "monitoring"}
            if cal is not None:
                cal.add(f)
                if cal.done(t_ms):
                    b = cal.result()
                    if b is None:
                        raise RuntimeError(f"{res.clip}: face found in <60% of the first {calib_s:.0f} s; "
                                           "can't calibrate (check framing / lighting)")
                    engine.base = res.baseline = b
                    cal = None
                res.frames.append(row)
                continue
            if start_ms is None:
                start_ms = t_ms
            for ev in engine.process(f):
                res.events.append(ReplayEvent(ev, engine.state, frame.copy()))
            s = engine.state
            row.update({"closed": int(s.eyes_closed), "perclos": round(s.perclos, 3), "blinks10s": s.blinks,
                        "yawns5m": s.yawns_5min, "nods5m": s.nods_5min, "gaze_s": round(s.gaze_away_s, 2),
                        "yaw_d": round(s.yaw_delta, 1), "pitch_d": round(s.pitch_delta, 1)})
            res.frames.append(row)
    return res
