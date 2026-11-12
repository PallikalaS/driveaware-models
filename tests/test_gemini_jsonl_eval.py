import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "eval"))
from driveaware_data.gemini_rows import SHEET_COLUMNS, SignalSummary, build_prompt, validate_row  # noqa: E402
import sheet_to_jsonl  # noqa: E402
import gemini_eval  # noqa: E402


def row(**kw):
    r = {c: "" for c in SHEET_COLUMNS}
    r.update(row_id="r1", split="train", subject="bindu", trigger="eye_closure_long", level="3",
             trigger_source="rules", keyframe="r1.jpg",
             signal_line=build_prompt("eye_closure_long", 3, SignalSummary(perclos=0.2, longest_closure_s=1.4)),
             event="drowsy", confirmed="yes", severity="high",
             message="Your eyes closed for over a second, so please pull over safely and rest.",
             action="pull_over_now", labeler="Bindu")
    r.update(kw)
    return r


def test_valid_row_passes():
    assert validate_row(row()) == []


def test_guide_rules_are_enforced():
    assert any("authority" in e for e in validate_row(row(event="false_alarm", confirmed="no", severity="low",
                                                          action="none", message="No alert needed.")))
    assert any("action" in e for e in validate_row(row(action="refocus")))
    assert any("words" in e for e in validate_row(row(message="word " * 21 + "end.")))
    assert any("period" in e for e in validate_row(row(message="Please pull over safely")))
    assert any("labeler" in e for e in validate_row(row(labeler="")))
    sun = build_prompt("eye_closure_long", 3, SignalSummary(perclos=0.2, longest_closure_s=1.4, sunglasses=True))
    assert any("sunglasses" in e for e in validate_row(row(signal_line=sun)))
    assert any("stand-in" in e for e in validate_row(row(trigger_source="clip_label")))
    # a mirror check is a legitimate false alarm
    gaze = build_prompt("gaze_away", 1, SignalSummary(perclos=0.05, gaze_away_s=2.3))
    assert validate_row(row(trigger="gaze_away", signal_line=gaze, event="false_alarm", confirmed="no",
                            severity="low", action="none", message="No alert needed.")) == []


def write_sheet(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, SHEET_COLUMNS); w.writeheader(); w.writerows(rows)


def test_converter_writes_vertex_format(tmp_path):
    sheet = tmp_path / "sheet.csv"
    write_sheet(sheet, [row(), row(row_id="r2", split="val", keyframe="r2.jpg"), row(row_id="r3", split="test")])
    assert sheet_to_jsonl.main([str(sheet), "--image-prefix", "gs://b/k", "--out", str(tmp_path / "o")]) == 0
    ex = json.loads((tmp_path / "o" / "train.jsonl").read_text().splitlines()[0])
    user, model = ex["contents"]
    assert ex["systemInstruction"]["parts"][0]["text"].startswith("You are DriveAware")
    assert user["parts"][0]["fileData"] == {"mimeType": "image/jpeg", "fileUri": "gs://b/k/r1.jpg"}
    assert user["parts"][1]["text"].startswith("Signals (last 10 s): trigger=eye_closure_long")
    assert list(json.loads(model["parts"][0]["text"])) == ["event", "confirmed", "severity", "message", "action"]
    assert len((tmp_path / "o" / "test.jsonl").read_text().splitlines()) == 1


def test_converter_stops_on_bad_rows(tmp_path, capsys):
    sheet = tmp_path / "sheet.csv"
    write_sheet(sheet, [row(), row(row_id="r2", severity="medium")])
    assert sheet_to_jsonl.main([str(sheet), "--image-prefix", "gs://b/k", "--out", str(tmp_path / "o")]) == 1
    assert "row 3 (r2)" in capsys.readouterr().out
    assert not (tmp_path / "o" / "train.jsonl").exists()


def test_eval_scores_replies(tmp_path):
    sheet = tmp_path / "sheet.csv"
    gaze = build_prompt("gaze_away", 1, SignalSummary(gaze_away_s=2.3))
    write_sheet(sheet, [row(split="test"),
                        row(row_id="r2", split="test", trigger="gaze_away", signal_line=gaze, event="false_alarm",
                            confirmed="no", severity="low", action="none", message="No alert needed.")])
    sheet_to_jsonl.main([str(sheet), "--image-prefix", "gs://b/k", "--out", str(tmp_path / "o")])
    ex = gemini_eval.load_examples(tmp_path / "o" / "test.jsonl")
    # "base" calls the eye closure a false alarm (authority violation) and returns junk for the other
    replies = iter(['{"event":"false_alarm","confirmed":"no","severity":"low","message":"No alert needed.","action":"none"}',
                    "not json"])
    p = gemini_eval.run(ex, lambda e: next(replies), "base", tmp_path / "r")
    s = gemini_eval.summarize(p)
    assert s["event_accuracy"] == 0 and s["authority_violations"] == 1 and s["valid_json_rate"] == 0.5
    # "tuned" answers exactly the labels
    p2 = gemini_eval.run(ex, lambda e: json.dumps(e["label"]), "tuned", tmp_path / "r")
    s2 = gemini_eval.summarize(p2)
    assert s2["event_accuracy"] == 1 and s2["guide_valid_rate"] == 1 and s2["false_alarm_recall"] == 1
    assert gemini_eval.main(["compare", str(p), str(p2)]) == 0
    assert "| event_accuracy | 0.000 | 1.000 |" in (tmp_path / "r" / "comparison.md").read_text()


def test_system_instruction_matches_backend_copy():
    backend = ROOT.parent / "driveaware-app" / "backend" / "app" / "logic.py"
    if backend.exists():
        import re
        from driveaware_data.gemini_rows import SYSTEM_INSTRUCTION
        src = backend.read_text()
        body = re.search(r"SYSTEM_INSTRUCTION = \((.*?)\n\)", src, re.S).group(1)
        assert "".join(re.findall(r'"(.*?)"', body)) == SYSTEM_INSTRUCTION
