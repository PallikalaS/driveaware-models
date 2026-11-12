#!/usr/bin/env python3
"""Base vs tuned Gemini on the held-out test set (task #29).

Sends every test.jsonl example (from scripts/sheet_to_jsonl.py) to a model with
the SAME request the backend makes (system instruction, image + signal line,
JSON schema, minimal thinking) and scores the reply against the label.

  # base model, then tuned endpoint, then the comparison table for the report
  python eval/gemini_eval.py run test.jsonl --model gemini-3.5-flash --name base
  python eval/gemini_eval.py run test.jsonl --model projects/<num>/locations/us/endpoints/<id> --name tuned
  python eval/gemini_eval.py compare results/gemini/base_predictions.csv results/gemini/tuned_predictions.csv

Needs: pip install google-genai, and gcloud auth application-default login
(as an account with Vertex AI User on driveaware-cs663-swathi).

Metrics: event accuracy and macro-F1, false_alarm precision/recall, severity /
action / confirmed accuracy, valid-JSON rate, guide-valid rate (labeling_guide.md
checks on the reply), authority-rule violations, latency p50/p95.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from driveaware_data.gemini_rows import (ANSWER_KEYS, MEASURED_TRIGGERS, SYSTEM_INSTRUCTION,  # noqa: E402
                                         parse_signal_line, validate_answer)
from driveaware_data.labels import GEMINI_ACTIONS, GEMINI_EVENTS, GEMINI_SEVERITY  # noqa: E402

RESPONSE_SCHEMA = {   # copy of driveaware-app backend/app/schema.py GEMINI_RESPONSE_SCHEMA
    "type": "OBJECT",
    "properties": {
        "event": {"type": "STRING", "enum": GEMINI_EVENTS},
        "confirmed": {"type": "STRING", "enum": ["yes", "no"]},
        "severity": {"type": "STRING", "enum": GEMINI_SEVERITY},
        "message": {"type": "STRING"},
        "action": {"type": "STRING", "enum": GEMINI_ACTIONS},
    },
    "required": ANSWER_KEYS, "propertyOrdering": ANSWER_KEYS,
}
PRED_COLUMNS = ["i", "trigger", "image_uri", "latency_ms", "valid_json", "guide_errors"] + \
    [f"label_{k}" for k in ANSWER_KEYS] + [f"pred_{k}" for k in ANSWER_KEYS]


def load_examples(path: Path) -> list[dict]:
    out = []
    for line in open(path, encoding="utf-8"):
        ex = json.loads(line)
        user = ex["contents"][0]["parts"]
        uri = next(p["fileData"]["fileUri"] for p in user if "fileData" in p)
        text = next(p["text"] for p in user if "text" in p)
        label = json.loads(ex["contents"][1]["parts"][0]["text"])
        out.append({"uri": uri, "text": text, "label": label, "signals": parse_signal_line(text)})
    return out


def gemini_caller(model: str, project: str, location: str):
    from google import genai
    from google.genai import types
    client = genai.Client(vertexai=True, project=project, location=location)
    cfg = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.MINIMAL),
        max_output_tokens=512, response_mime_type="application/json", response_schema=RESPONSE_SCHEMA)

    def call(ex: dict) -> str:
        parts = [types.Part.from_uri(file_uri=ex["uri"], mime_type="image/jpeg"), types.Part.from_text(text=ex["text"])]
        return client.models.generate_content(model=model, contents=[types.Content(role="user", parts=parts)],
                                              config=cfg).text
    return call


def run(examples: list[dict], call, name: str, out_dir: Path, pause_s: float = 0.0) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}_predictions.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, PRED_COLUMNS); w.writeheader()
        for i, ex in enumerate(examples):
            t0 = time.perf_counter()
            try:
                raw, err = call(ex), ""
            except Exception as e:  # quota, timeout: recorded as an invalid reply
                raw, err = "", f"call failed: {e}"[:200]
            ms = int((time.perf_counter() - t0) * 1000)
            try:
                pred = {k: str(json.loads(raw).get(k, "")) for k in ANSWER_KEYS}
                ok = True
            except Exception:
                pred, ok = {k: "" for k in ANSWER_KEYS}, False
            trig = ex["signals"].get("trigger", "")
            guide = validate_answer(pred, trig, ex["signals"]) if ok else [err or "invalid JSON"]
            row = {"i": i, "trigger": trig, "image_uri": ex["uri"], "latency_ms": ms, "valid_json": int(ok),
                   "guide_errors": "; ".join(guide)}
            row.update({f"label_{k}": ex["label"][k] for k in ANSWER_KEYS})
            row.update({f"pred_{k}": pred[k] for k in ANSWER_KEYS})
            w.writerow(row)
            print(f"{i + 1}/{len(examples)} {trig}: label {ex['label']['event']} -> {pred['event'] or 'INVALID'} ({ms} ms)")
            if pause_s:
                time.sleep(pause_s)
    return path


def summarize(pred_csv: Path) -> dict:
    rows = list(csv.DictReader(open(pred_csv)))
    n = len(rows)
    acc = lambda k: sum(r[f"label_{k}"] == r[f"pred_{k}"] for r in rows) / n  # noqa: E731
    f1s = []
    for e in sorted({r["label_event"] for r in rows}):
        tp = sum(r["label_event"] == e and r["pred_event"] == e for r in rows)
        fp = sum(r["label_event"] != e and r["pred_event"] == e for r in rows)
        fn = sum(r["label_event"] == e and r["pred_event"] != e for r in rows)
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    fa_tp = sum(r["label_event"] == "false_alarm" and r["pred_event"] == "false_alarm" for r in rows)
    fa_pred = sum(r["pred_event"] == "false_alarm" for r in rows)
    fa_true = sum(r["label_event"] == "false_alarm" for r in rows)
    lat = sorted(int(r["latency_ms"]) for r in rows)
    return {
        "n": n,
        "event_accuracy": acc("event"), "event_macro_f1": statistics.mean(f1s) if f1s else 0.0,
        "false_alarm_precision": fa_tp / fa_pred if fa_pred else float("nan"),
        "false_alarm_recall": fa_tp / fa_true if fa_true else float("nan"),
        "severity_accuracy": acc("severity"), "action_accuracy": acc("action"),
        "confirmed_accuracy": acc("confirmed"),
        "valid_json_rate": sum(int(r["valid_json"]) for r in rows) / n,
        "guide_valid_rate": sum(not r["guide_errors"] for r in rows) / n,
        "authority_violations": sum(r["trigger"] in MEASURED_TRIGGERS and r["pred_event"] == "false_alarm" for r in rows),
        "latency_p50_ms": lat[n // 2] if n else 0,
        "latency_p95_ms": lat[min(n - 1, int(0.95 * n))] if n else 0,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("test_jsonl"); r.add_argument("--model", required=True); r.add_argument("--name", required=True)
    r.add_argument("--project", default="driveaware-cs663-swathi"); r.add_argument("--location", default="us")
    r.add_argument("--out", default="results/gemini"); r.add_argument("--limit", type=int, default=0)
    r.add_argument("--pause-s", type=float, default=0.0, help="sleep between calls if you hit quota")
    c = sub.add_parser("compare"); c.add_argument("pred_csvs", nargs="+")
    a = ap.parse_args(argv)

    if a.cmd == "run":
        ex = load_examples(Path(a.test_jsonl).expanduser())
        ex = ex[: a.limit] if a.limit else ex
        p = run(ex, gemini_caller(a.model, a.project, a.location), a.name, Path(a.out), a.pause_s)
        s = summarize(p)
        json.dump({"model": a.model, **s}, open(Path(a.out) / f"{a.name}_summary.json", "w"), indent=2)
        for k, v in s.items():
            print(f"{k:24s} {v:.3f}" if isinstance(v, float) else f"{k:24s} {v}")
        return 0

    sums = {Path(p).stem.replace("_predictions", ""): summarize(Path(p)) for p in a.pred_csvs}
    names = list(sums)
    lines = ["| metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for k in next(iter(sums.values())):
        cells = [f"{sums[n][k]:.3f}" if isinstance(sums[n][k], float) else str(sums[n][k]) for n in names]
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    print(table)
    Path(a.pred_csvs[0]).parent.joinpath("comparison.md").write_text(table + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
