#!/usr/bin/env python3
"""Final Gemini label sheet (CSV) -> Vertex AI tuning JSONL (task #22).

Every row is checked against labeling_guide.md (allowed words, authority rule,
event/severity/action table, sunglasses cap, message rules). Any invalid row
stops the run and is listed, so nothing half-labeled reaches the tuning job.

  python scripts/sheet_to_jsonl.py label_sheet_final.csv \\
      --image-prefix gs://driveaware-cs663-swathi-data/gemini/keyframes \\
      --out ~/Drive/DriveAware/Gemini_events/jsonl

Writes train.jsonl and val.jsonl (for tuning) and test.jsonl (held-out, only for
eval/gemini_eval.py, never uploaded to the tuning job), plus counts.csv.
Then upload:  gcloud storage cp -r <keyframes dir> gs://.../gemini/keyframes
              gcloud storage cp <out>/train.jsonl <out>/val.jsonl gs://.../gemini/
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from driveaware_data.gemini_rows import to_example, validate_row  # noqa: E402

MIN_FALSE_ALARM, MIN_PER_EVENT = 25, 15   # Bindu #9 target counts (train + val)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sheet", help="label sheet exported as CSV")
    ap.add_argument("--image-prefix", required=True, help="gs:// folder the keyframes are uploaded to")
    ap.add_argument("--out", required=True)
    ap.add_argument("--keyframes", help="local keyframes folder: check every file exists")
    ap.add_argument("--allow-clip-label", action="store_true", help="accept Model A stand-in rows")
    ap.add_argument("--skip-blank", action="store_true",
                    help="skip rows left completely blank on purpose (no answer and no labeler); "
                         "half-filled rows still stop the run")
    a = ap.parse_args(argv)
    if not a.image_prefix.startswith("gs://"):
        ap.error("--image-prefix must be a gs:// path")

    rows = list(csv.DictReader(open(Path(a.sheet).expanduser(), newline="", encoding="utf-8-sig")))
    rows = [r for r in rows if (r.get("row_id") or "").strip() and r.get("row_id", "").strip() != "row_id"]
    if a.skip_blank:
        blank = [r for r in rows if not any((r.get(k) or "").strip()
                                           for k in ("event", "confirmed", "severity", "message", "action", "labeler"))]
        if blank:
            print(f"skipping {len(blank)} blank rows: " + ", ".join(r["row_id"] for r in blank))
        rows = [r for r in rows if r not in blank]
    bad = []
    for i, r in enumerate(rows, start=2):           # row 1 is the header in Sheets
        err = validate_row(r, a.allow_clip_label)
        if a.keyframes and r.get("keyframe") and not (Path(a.keyframes).expanduser() / r["keyframe"].strip()).exists():
            err.append(f"keyframe file not found: {r['keyframe']}")
        if err:
            bad.append((i, r.get("row_id", ""), err))
    if bad:
        print(f"{len(bad)} of {len(rows)} rows need fixing (sheet row number, row_id):")
        for i, rid, err in bad:
            print(f"  row {i} ({rid}): " + "; ".join(err))
        return 1

    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    files = {s: open(out / f"{s}.jsonl", "w", encoding="utf-8") for s in ("train", "val", "test")}
    for r in rows:
        s = r["split"].strip()
        files[s].write(json.dumps(to_example(r, a.image_prefix), ensure_ascii=False) + "\n")
        counts[(s, r["event"].strip())] += 1
    for f in files.values():
        f.close()

    events = sorted({e for _, e in counts})
    with open(out / "counts.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["event", "train", "val", "test"])
        for e in events:
            w.writerow([e] + [counts[(s, e)] for s in ("train", "val", "test")])
    print(f"{len(rows)} rows ok -> {out}/train.jsonl, val.jsonl, test.jsonl")
    print(open(out / "counts.csv").read())
    for e in events:
        n = counts[("train", e)] + counts[("val", e)]
        need = MIN_FALSE_ALARM if e == "false_alarm" else MIN_PER_EVENT
        if n < need:
            print(f"! {e}: {n} train+val rows, target {need}. Tell Bindu which type is short.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
