#!/usr/bin/env python3
"""Make the built label sheet the right size for labeling (Bindu's ~250 rows).

1. Rows from clips with no team member in the name (Vanessa's pilot S01..S04) move
   to test: they are held-out recordings (needed for sheets built before the
   Team_test_only folder rule).
2. Test is cut to --test-rows by stratified sampling over trigger (each trigger
   keeps at least 2 rows when it has them), so labeling time goes to train/val.
   Train and val are kept whole.

  python scripts/trim_label_sheet.py ~/Projects/driveaware/data/Gemini_events/label_sheet.csv --test-rows 50

Writes label_sheet_trimmed.csv next to the input (the input is not changed).
Keyframes are shared, so nothing else moves.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

TEAM = {"swathi", "bindu", "mahitha", "vanessa"}


def trim(rows: list[dict], test_rows: int, seed: int = 663) -> list[dict]:
    for r in rows:
        if r["subject"].lower() not in TEAM:
            r["split"] = "test"
    keep = [r for r in rows if r["split"] != "test"]
    test = [r for r in rows if r["split"] == "test"]
    if len(test) > test_rows:
        rng = random.Random(seed)
        by = defaultdict(list)
        for r in test:
            by[r["trigger"]].append(r)
        for v in by.values():
            rng.shuffle(v)
        # at least 2 per trigger, rest proportional to its share
        quota = {t: min(len(v), 2) for t, v in by.items()}
        left = test_rows - sum(quota.values())
        total = sum(len(v) - quota[t] for t, v in by.items())
        for t, v in by.items():
            if total > 0 and left > 0:
                quota[t] += min(len(v) - quota[t], round(left * (len(v) - quota[t]) / total))
        test = [r for t, v in by.items() for r in v[:quota[t]]]
    out = keep + test
    order = {r["row_id"]: i for i, r in enumerate(rows)}
    return sorted(out, key=lambda r: order[r["row_id"]])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sheet")
    ap.add_argument("--test-rows", type=int, default=50)
    ap.add_argument("--seed", type=int, default=663)
    a = ap.parse_args(argv)
    src = Path(a.sheet).expanduser()
    rows = list(csv.DictReader(open(src, newline="", encoding="utf-8")))
    cols = list(rows[0].keys()) if rows else []
    out = trim(rows, a.test_rows, a.seed)
    dst = src.with_name(src.stem + "_trimmed.csv")
    with open(dst, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, cols); w.writeheader(); w.writerows(out)
    print(f"{len(rows)} -> {len(out)} rows -> {dst}")
    print("by split:  ", dict(Counter(r["split"] for r in out)))
    for s in ("train", "val", "test"):
        print(f"{s:5s} by trigger:", dict(Counter(r["trigger"] for r in out if r["split"] == s)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
