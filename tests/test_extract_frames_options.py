import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "tests"))
import extract_frames  # noqa: E402
from test_replay_pipeline import make_clip  # noqa: E402


def run(tmp_path, *extra):
    out = tmp_path / "Frames"
    extract_frames.main(["--team", str(tmp_path / "Team_raw"), "--out", str(out), "--crop", "center",
                         "--fps", "1", "--min-sharpness", "0", "--dup-bits", "-1", *extra])
    return list(csv.DictReader((out / "labels.csv").open()))


def test_skip_start_spans_and_team_val(tmp_path):
    raw = tmp_path / "Team_raw"; raw.mkdir()
    for who in ("swathi", "bindu"):
        for act in ("texting", "eye_rub", "safe", "smoking"):
            make_clip(raw / f"{who}_{act}_day-none_t1.mp4", seconds=12)
    spans = tmp_path / "activity_spans.csv"
    spans.write_text("clip,start_s,end_s\nbindu_texting_day-none_t1.mp4,7,9\n")
    rows = run(tmp_path, "--spans", str(spans), "--team-val-frac", "0.25")
    t = lambda clip: sorted(float(r["t_s"]) for r in rows if r["clip"] == clip)  # noqa: E731
    assert min(t("swathi_texting_day-none_t1.mp4")) >= 5.0          # first 5 s dropped
    assert min(t("swathi_safe_day-none_t1.mp4")) == 0.0             # safe clips keep the start
    assert t("bindu_texting_day-none_t1.mp4") == [7.0, 8.0, 9.0]    # only inside the span
    split = {r["clip"]: r["split"] for r in rows}
    for who in ("swathi", "bindu"):                                  # 1 of 4 clips each in val
        assert sum(v == "val" for c, v in split.items() if c.startswith(who)) == 1
