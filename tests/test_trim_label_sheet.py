import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import trim_label_sheet as t  # noqa: E402


def rows():
    out, i = [], 0
    def add(subject, split, trigger, n):
        nonlocal i
        for _ in range(n):
            i += 1
            out.append({"row_id": f"r{i:04d}", "subject": subject, "split": split, "trigger": trigger})
    add("bindu", "train", "phone_use", 30); add("bindu", "val", "nod", 8)
    add("s01", "val", "eye_closure_long", 3)            # pilot clip that went to val
    add("vanessa", "test", "phone_use", 200); add("mahitha", "test", "eye_closure_long", 100)
    add("mahitha", "test", "yawn", 1); add("vanessa", "test", "smoking", 6)
    return out


def test_pilot_moves_to_test_and_test_is_trimmed_stratified():
    out = t.trim(rows(), test_rows=40)
    split = Counter(r["split"] for r in out)
    assert split["train"] == 30 and split["val"] == 8          # pilot left val, train/val whole
    test = [r for r in out if r["split"] == "test"]
    assert 38 <= len(test) <= 42
    trig = Counter(r["trigger"] for r in test)
    assert trig["yawn"] == 1 and trig["smoking"] >= 2 and trig["phone_use"] > trig["smoking"]
    assert [r["row_id"] for r in out] == sorted(r["row_id"] for r in out)   # original order kept
