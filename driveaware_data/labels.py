"""Single source of truth for every label word in DriveAware.

Words here MUST match the To-Dos doc "Shared rules and formats" exactly.
Never rename them; the Android app, the backend schema and the Gemini
label sheet all import / copy these lists.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------- Model A
SUBCLASSES = [
    "safe",
    "phone_ear", "texting", "phone_in_hand",
    "drinking_cup", "drinking_bottle",
    "smoking",
    "eye_rub", "face_scratch", "hand_forehead",
]

SUBCLASS_TO_APP = {
    "safe": "safe_driving",
    "phone_ear": "phone_use", "texting": "phone_use", "phone_in_hand": "phone_use",
    "drinking_cup": "drinking", "drinking_bottle": "drinking",
    "smoking": "smoking",
    "eye_rub": "face_touch", "face_scratch": "face_touch", "hand_forehead": "face_touch",
}
APP_CLASSES = ["safe_driving", "phone_use", "drinking", "smoking", "face_touch"]

CONDITIONS = ["day-none", "day-clear", "day-sun", "night-none", "night-clear", "night-sun"]

# ------------------------------------------------- event CSV (val + test sessions)
EVENT_TYPES = [
    "eye_closure_long", "perclos_high", "yawn", "nod", "gaze_away", "mirror_check",
    "phone_use", "drinking", "smoking", "face_touch", "none",
]
# Event types that the app should NOT alert on (used by match_events.py).
NON_ALERT_EVENT_TYPES = {"mirror_check", "none"}

# ------------------------------------------------------- Gemini label sheet
GEMINI_EVENTS = ["drowsy", "distracted_phone", "distracted_gaze", "drinking",
                 "smoking", "early_fatigue", "false_alarm"]
GEMINI_CONFIRMED = ["yes", "no"]
GEMINI_SEVERITY = ["low", "medium", "high"]
GEMINI_ACTIONS = ["none", "refocus", "take_break_soon", "pull_over_now"]
MESSAGE_MAX_WORDS = 20

# ------------------------------------------------------------ file names
CLIP_RE = re.compile(
    r"^(?P<name>[a-z]+)_(?P<activity>[a-z_]+?)_(?P<condition>(day|night)-(none|clear|sun))_t(?P<take>\d+)\.mp4$"
)


@dataclass(frozen=True)
class ClipName:
    name: str
    activity: str
    condition: str
    take: int


def parse_clip_name(filename: str) -> ClipName:
    """Parse `name_activity_condition_take.mp4`, e.g. bindu_texting_night-clear_t1.mp4."""
    m = CLIP_RE.match(filename)
    if not m:
        raise ValueError(
            f"Bad clip name {filename!r}. Expected name_activity_condition_tN.mp4, lowercase."
        )
    activity = m["activity"]
    if activity not in SUBCLASSES:
        raise ValueError(f"Unknown activity {activity!r} in {filename!r}. Allowed: {SUBCLASSES}")
    return ClipName(m["name"], activity, m["condition"], int(m["take"]))


# ------------------------------------------------- public dataset mappings
# State Farm Distracted Driver (Kaggle) class folders c0..c9.
STATE_FARM_TO_SUBCLASS = {
    "c0": "safe",
    "c1": "texting",        # texting - right
    "c2": "phone_ear",      # talking on the phone - right
    "c3": "texting",        # texting - left
    "c4": "phone_ear",      # talking on the phone - left
    "c5": None,             # operating the radio  -> not in our classes, dropped
    "c6": "drinking_cup",   # drinking (container type unknown; spot-check & fix)
    "c7": None,             # reaching behind -> dropped
    "c8": "hand_forehead",  # hair and makeup -> closest face_touch subclass
    "c9": None,             # talking to passenger -> dropped
}

# DMD (dmd.vicomtech.org) distraction "driver_actions" labels -> subclass.
# DMD has no smoking / face-touch classes: those come only from team clips.
DMD_TO_SUBCLASS = {
    "safe_drive": "safe",
    "texting_right": "texting", "texting_left": "texting",
    "phonecall_right": "phone_ear", "phonecall_left": "phone_ear",
    "drinking": "drinking_cup",
    "hair_and_makeup": "hand_forehead",
    "radio": None, "reach_side": None, "reach_backseat": None, "talking_to_passenger": None,
}


def map_public_label(source: str, raw_label: str) -> str | None:
    """Return our subclass for a public-dataset label, or None to drop the frame."""
    table = {"statefarm": STATE_FARM_TO_SUBCLASS, "dmd": DMD_TO_SUBCLASS}[source.lower()]
    if raw_label not in table:
        raise KeyError(f"Unknown {source} label {raw_label!r}")
    return table[raw_label]


def assign_split(subject: str, held_out: set[str], val_subjects: set[str]) -> str:
    """Subject-based split: held-out members are test only, never train/val."""
    s = subject.lower()
    if s in held_out:
        return "test"
    if s in val_subjects:
        return "val"
    return "train"
