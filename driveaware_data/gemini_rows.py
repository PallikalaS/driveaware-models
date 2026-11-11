"""Gemini label-sheet rows: the exact signal line the backend sends to Gemini.

build_prompt() is a copy of driveaware-app backend/app/logic.py build_prompt().
Training rows MUST use the same text the backend sends at serving time, so if
you change one, change the other (tests/test_gemini_rows.py pins the format).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SignalSummary:
    perclos: float = 0.0
    blinks: int = 0
    longest_closure_s: float = 0.0
    yawns_5min: int = 0
    nods_5min: int = 0
    gaze_away_s: float = 0.0
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    model_a_class: str | None = None
    model_a_conf: float | None = None
    hand_occupied: bool = False
    sunglasses: bool = False


def build_prompt(trigger: str, level: int, s: SignalSummary) -> str:
    parts = [
        f"trigger={trigger}", f"level={level}",
        f"perclos={s.perclos:.2f}", f"blinks={s.blinks}",
        f"longest_closure_s={s.longest_closure_s:.1f}",
        f"yawns_5min={s.yawns_5min}", f"nods_5min={s.nods_5min}",
        f"gaze_away_s={s.gaze_away_s:.1f}",
        f"yaw={s.yaw_deg:.0f}", f"pitch={s.pitch_deg:.0f}",
        f"model_a={s.model_a_class or 'na'}:{(s.model_a_conf or 0):.2f}",
        f"hand_occupied={'yes' if s.hand_occupied else 'no'}",
        f"sunglasses={'yes' if s.sunglasses else 'no'}",
    ]
    return "Signals (last 10 s): " + " ".join(parts)


# Label-sheet columns. Labeler fills event..checked_by using labeling_guide.md.
SHEET_COLUMNS = [
    "row_id", "split", "subject", "clip", "condition", "t_s", "trigger", "level",
    "trigger_source", "keyframe", "signal_line",
    "event", "confirmed", "severity", "message", "action", "labeler", "checked_by",
]
