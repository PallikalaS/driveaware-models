import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from driveaware_data.gemini_rows import SignalSummary, build_prompt  # noqa: E402
from build_label_sheet import describe_clip  # noqa: E402


def test_signal_line_matches_backend_format():
    # Same expected prefix as driveaware-app backend/tests/test_logic.py::test_prompt_is_stable
    p = build_prompt("yawn", 1, SignalSummary(perclos=0.12, yawns_5min=2))
    assert p.startswith("Signals (last 10 s): trigger=yawn level=1 perclos=0.12")
    assert p.endswith("model_a=na:0.00 hand_occupied=no sunglasses=no")


def test_describe_clip_names():
    assert describe_clip(Path("bindu_texting_night-clear_t1.mp4")) == \
        {"subject": "bindu", "condition": "night-clear", "activity": "texting"}
    assert describe_clip(Path("swathi_phone_in_hand_day-sun_t2.mp4"))["activity"] == "phone_in_hand"
    assert describe_clip(Path("V01_bindu_day-none_t1.mp4")) == \
        {"subject": "bindu", "condition": "day-none", "activity": None}
    assert describe_clip(Path("S03_vanessa_night-none_t1.mp4"))["subject"] == "vanessa"
    assert describe_clip(Path("bindu_drowsy_day-none_t1.mp4"))["condition"] == "day-none"
