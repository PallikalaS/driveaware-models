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


# --------------------------------------------------------------------------------
# Tuning data (task #22). SYSTEM_INSTRUCTION is a copy of driveaware-app
# backend/app/logic.py SYSTEM_INSTRUCTION: train == serve, change both together.
SYSTEM_INSTRUCTION = (
    "You are DriveAware's in-car safety checker. You get one dashboard-camera frame of the "
    "driver and a summary of the last 10 seconds of measured face and activity signals. "
    "Decide what is happening and reply ONLY with JSON matching the schema: event, confirmed, "
    "severity, message, action. The message is one calm sentence of 20 words or fewer, spoken "
    "to the driver. Measured signals (long eye closure, PERCLOS, nodding) are facts: never call "
    "them false_alarm. Use false_alarm only for ambiguous visual events such as a mirror check."
)
ANSWER_KEYS = ["event", "confirmed", "severity", "message", "action"]
MEASURED_TRIGGERS = {"eye_closure_long", "perclos_high", "nod"}
TRIGGERS = ["eye_closure_long", "perclos_high", "yawn", "nod", "gaze_away",
            "phone_use", "drinking", "smoking", "face_touch"]
# labeling_guide.md section 4: which actions go with each event / severity
ALLOWED_ACTIONS = {
    ("drowsy", "high"): {"pull_over_now"}, ("drowsy", "medium"): {"take_break_soon"},
    ("early_fatigue", "low"): {"take_break_soon"}, ("early_fatigue", "medium"): {"take_break_soon"},
    ("distracted_phone", "medium"): {"refocus"}, ("distracted_phone", "high"): {"refocus"},
    ("distracted_gaze", "low"): {"refocus"}, ("distracted_gaze", "medium"): {"refocus"},
    ("drinking", "low"): {"refocus"}, ("smoking", "low"): {"refocus"},
    ("false_alarm", "low"): {"none"},
}
FALSE_ALARM_MESSAGE = "No alert needed."


def parse_signal_line(line: str) -> dict[str, str]:
    """'Signals (last 10 s): trigger=yawn level=1 ...' -> {'trigger': 'yawn', ...}"""
    body = line.split(":", 1)[1] if ":" in line else line
    return dict(p.split("=", 1) for p in body.split() if "=" in p)


def answer_of(row: dict) -> dict:
    return {k: (row.get(k) or "").strip() for k in ANSWER_KEYS}


def validate_answer(a: dict, trigger: str, signals: dict[str, str]) -> list[str]:
    """Labeling-guide checks for one answer (a labeled row or a model reply). [] = valid."""
    from .labels import GEMINI_ACTIONS, GEMINI_CONFIRMED, GEMINI_EVENTS, GEMINI_SEVERITY, MESSAGE_MAX_WORDS
    err = []
    for k, allowed in [("event", GEMINI_EVENTS), ("confirmed", GEMINI_CONFIRMED),
                       ("severity", GEMINI_SEVERITY), ("action", GEMINI_ACTIONS)]:
        if a.get(k) not in allowed:
            err.append(f"{k}={a.get(k)!r} not one of {allowed}")
    if err:
        return err
    ev, sev, act, msg = a["event"], a["severity"], a["action"], a.get("message", "")
    if (a["confirmed"] == "no") != (ev == "false_alarm"):
        err.append("confirmed must be 'no' only for false_alarm and 'yes' otherwise")
    f = lambda k: float(signals.get(k, "0") or 0)  # noqa: E731
    measured = trigger in MEASURED_TRIGGERS or f("longest_closure_s") >= 1.0 or \
        f("perclos") >= 0.20 or f("nods_5min") >= 1   # perclosHigh (tuned, was 0.15)
    if measured and ev == "false_alarm":
        err.append("authority rule: measured signal can never be false_alarm")
    if trigger in MEASURED_TRIGGERS and ev != "drowsy":
        err.append(f"authority rule: trigger {trigger} must be labeled drowsy")
    if (ev, sev) not in ALLOWED_ACTIONS:
        err.append(f"severity {sev} is not used for event {ev} (guide section 4)")
    elif act not in ALLOWED_ACTIONS[(ev, sev)]:
        err.append(f"action {act} does not match {ev}/{sev}; expected {sorted(ALLOWED_ACTIONS[(ev, sev)])}")
    if signals.get("sunglasses") == "yes" and trigger in ("eye_closure_long", "perclos_high") and sev == "high":
        err.append("sunglasses: eye-based severity is capped at medium")
    words = msg.split()
    if not msg:
        err.append("message is empty")
    elif len(words) > MESSAGE_MAX_WORDS:
        err.append(f"message has {len(words)} words (max {MESSAGE_MAX_WORDS})")
    if msg and not msg.endswith("."):
        err.append("message must end with a period")
    if "!" in msg:
        err.append("message must not contain '!'")
    if ev == "false_alarm" and msg != FALSE_ALARM_MESSAGE:
        err.append(f"false_alarm message must be exactly {FALSE_ALARM_MESSAGE!r}")
    return err


def validate_row(row: dict, allow_clip_label: bool = False) -> list[str]:
    """All checks for one label-sheet row before it becomes a tuning example."""
    err = []
    if (row.get("split") or "").strip() not in ("train", "val", "test"):
        err.append(f"split={row.get('split')!r} must be train, val or test")
    trig = (row.get("trigger") or "").strip()
    if trig not in TRIGGERS:
        err.append(f"trigger={trig!r} unknown")
    if not (row.get("keyframe") or "").strip():
        err.append("keyframe is empty")
    line = (row.get("signal_line") or "").strip()
    if not line.startswith("Signals (last 10 s): "):
        err.append("signal_line must be the exact line from build_label_sheet.py")
    if not (row.get("labeler") or "").strip():
        err.append("labeler is empty (row not labeled yet)")
    if (row.get("trigger_source") or "").strip() == "clip_label" and not allow_clip_label:
        err.append("clip_label row uses the Model A stand-in (conf 0.00); regenerate the sheet "
                   "with model_a_v1, or pass --allow-clip-label")
    return err + validate_answer(answer_of(row), trig, parse_signal_line(line))


def to_example(row: dict, image_uri_prefix: str) -> dict:
    """One Vertex AI supervised-tuning line. Part order matches the backend:
    image first, then the signal line; the model turn is the compact JSON answer."""
    uri = image_uri_prefix.rstrip("/") + "/" + row["keyframe"].strip().lstrip("/")
    import json
    return {
        "systemInstruction": {"role": "system", "parts": [{"text": SYSTEM_INSTRUCTION}]},
        "contents": [
            {"role": "user", "parts": [
                {"fileData": {"mimeType": "image/jpeg", "fileUri": uri}},
                {"text": row["signal_line"].strip()},
            ]},
            {"role": "model", "parts": [{"text": json.dumps(answer_of(row), separators=(",", ":"))}]},
        ],
    }
