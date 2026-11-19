#!/usr/bin/env python3
"""Upload tuning data and run the Vertex AI supervised tuning job (tasks #23, #24, #31).

  # 1. upload keyframes + train/val JSONL from sheet_to_jsonl.py (#23)
  python scripts/gemini_tuning.py upload --keyframes <Gemini_events>/keyframes --jsonl <Gemini_events>/jsonl
  # 2. start the job (#24); writes tuning_job.txt
  python scripts/gemini_tuning.py tune
  # 3. check it (re-run any time); when it succeeds it prints the backend redeploy command (#31)
  python scripts/gemini_tuning.py status

Needs: pip install google-genai, gcloud auth application-default login (Vertex AI User).
Keyframe URIs inside the JSONL must match where they are uploaded: use the same
--image-prefix in sheet_to_jsonl.py as --bucket-dir here (default below).
If the job fails with a permission error on the bucket, the error names the
Vertex AI tuning service agent: give it Storage Object Viewer on the bucket.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT = "driveaware-cs663-swathi"
BUCKET_DIR = "gs://driveaware-cs663-swathi-data/gemini"
BASE_MODEL = "gemini-3.5-flash"
LOCATION = "us"   # Gemini 3+ tuning: "us"/"eu" multi-region; tuned model is served there too
MIN_TRAIN = 100            # Vertex AI recommends at least 100-500 examples
JOB_FILE = Path("tuning_job.txt")


def count_lines(p: Path) -> int:
    return sum(1 for line in open(p, encoding="utf-8") if line.strip())


def check_jsonl(jsonl_dir: Path, bucket_dir: str) -> tuple[int, int]:
    """Train/val files exist, parse, and point at images under bucket_dir/keyframes."""
    n = {}
    for split in ("train", "val"):
        p = jsonl_dir / f"{split}.jsonl"
        if not p.exists():
            raise SystemExit(f"{p} not found: run scripts/sheet_to_jsonl.py first")
        for i, line in enumerate(open(p, encoding="utf-8"), 1):
            uri = json.loads(line)["contents"][0]["parts"][0]["fileData"]["fileUri"]
            if not uri.startswith(bucket_dir.rstrip("/") + "/keyframes/"):
                raise SystemExit(f"{p.name} line {i}: image {uri} is not under {bucket_dir}/keyframes/. "
                                 f"Re-run sheet_to_jsonl.py with --image-prefix {bucket_dir}/keyframes")
        n[split] = count_lines(p)
    return n["train"], n["val"]


def upload(a) -> int:
    jsonl, kf = Path(a.jsonl).expanduser(), Path(a.keyframes).expanduser()
    train, val = check_jsonl(jsonl, a.bucket_dir)
    print(f"train {train} rows, val {val} rows")
    if train < MIN_TRAIN:
        print(f"! only {train} training rows; Vertex AI recommends 100-500. It will run, but expect a weak tune.")
    if val == 0:
        raise SystemExit("val.jsonl is empty: build the sheet with --val-frac or --val-dir")
    cmds = [["gcloud", "storage", "rsync", "-r", str(kf), f"{a.bucket_dir}/keyframes"],
            ["gcloud", "storage", "cp", str(jsonl / "train.jsonl"), str(jsonl / "val.jsonl"), f"{a.bucket_dir}/"]]
    for c in cmds:
        print("$", " ".join(c))
        if not a.dry_run:
            subprocess.run(c, check=True)
    return 0


def client(location: str):
    from google import genai
    return genai.Client(vertexai=True, project=PROJECT, location=location)


def tune(a, make_client=client) -> int:
    if not a.i_accept_cost:
        raise SystemExit("STOP: a Gemini 3.5 Flash tuning job is expensive. Our Nov 18 re-tune (291 examples, "
                         "automatic epochs) used ~13M training tokens and cost $130. Do not re-tune; point the "
                         "backend at the base model instead. To really start one, add --i-accept-cost.")
    from google.genai import types
    cfg = types.CreateTuningJobConfig(
        tuned_model_display_name=a.name,
        # newer google-genai wants TuningValidationDataset here; older versions only have TuningDataset
        validation_dataset=getattr(types, "TuningValidationDataset", types.TuningDataset)(gcs_uri=f"{a.bucket_dir}/val.jsonl"),
        **({"epoch_count": a.epochs} if a.epochs else {}),
    )
    # keep the client in a variable: a temporary Client is garbage-collected and closes its
    # HTTP connection mid-call ("Cannot send a request, as the client has been closed")
    c = make_client(a.location)
    job = c.tunings.tune(
        base_model=a.base_model,
        training_dataset=types.TuningDataset(gcs_uri=f"{a.bucket_dir}/train.jsonl"),
        config=cfg,
    )
    JOB_FILE.write_text(job.name + "\n")
    print(f"started {job.name} ({job.state})\nsaved to {JOB_FILE}; check with: python scripts/gemini_tuning.py status")
    print(f"console: https://console.cloud.google.com/vertex-ai/generative/tuning?project={PROJECT}")
    return 0


def status(a, make_client=client) -> int:
    name = a.job or (JOB_FILE.read_text().strip() if JOB_FILE.exists() else "")
    if not name:
        raise SystemExit("no job name: pass --job or run 'tune' first")
    c = make_client(a.location)
    while True:
        job = c.tunings.get(name=name)
        state = str(job.state).split(".")[-1]
        print(time.strftime("%H:%M"), state)
        if state in ("JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED") or not a.wait:
            break
        time.sleep(120)
    if state == "JOB_STATE_SUCCEEDED":
        ep = job.tuned_model.endpoint
        print(f"\ntuned endpoint: {ep}\n\nevaluate (#29):\n"
              f"  python eval/gemini_eval.py run test.jsonl --model {ep} --location {a.location} --name tuned\n"
              f"redeploy the backend with it (#31), from driveaware-app/backend:\n"
              f"  GCP_PROJECT={PROJECT} VERTEX_LOCATION={a.location} GEMINI_MODEL={ep} bash deploy.sh")
    elif state == "JOB_STATE_FAILED":
        print(getattr(job, "error", None))
        return 1
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bucket-dir", default=BUCKET_DIR)
    ap.add_argument("--location", default=LOCATION)
    sub = ap.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("upload"); u.add_argument("--keyframes", required=True); u.add_argument("--jsonl", required=True)
    u.add_argument("--dry-run", action="store_true")
    t = sub.add_parser("tune"); t.add_argument("--base-model", default=BASE_MODEL)
    t.add_argument("--i-accept-cost", action="store_true", help="required: tuning cost ~$130 per job here")
    t.add_argument("--name", default="driveaware-verifier-v1"); t.add_argument("--epochs", type=int, default=0,
                                                                              help="0 = Vertex AI default for the data size")
    s = sub.add_parser("status"); s.add_argument("--job", default=""); s.add_argument("--wait", action="store_true")
    a = ap.parse_args(argv)
    return {"upload": upload, "tune": tune, "status": status}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
