# driveaware-models

Model development code for **DriveAware** (CS663 Project 2): data preparation, Model A training,
Gemini fine-tuning, evaluation and system-test matching. App code lives in
[`driveaware-app`](../driveaware-app).

> Images, videos and model binaries never go to GitHub — they live in the shared Drive folder
> `DriveAware/` (and the GCS bucket for tuning). Commit CSVs, notebooks (with outputs) and results.

## Layout

| Path | What | Owner |
|---|---|---|
| `driveaware_data/labels.py` | Every label word (classes, event types, Gemini sheet words), clip-name parser, public-dataset mapping, subject split | Swathi |
| `driveaware_data/crop.py` | Center vs dynamic upper-body crop (same geometry as the app) | Swathi |
| `scripts/extract_frames.py` | Clips/images → 224×224 frames + pre-filled `labels.csv` + `class_counts.csv` | Swathi wrote · Mahitha runs |
| `scripts/crop_compare.py` | Side-by-side crop images for choosing the crop | Swathi wrote · Mahitha runs |
| `eval/match_events.py` | App event log vs scripted session CSV → recall, precision, false alarms/10 min, delay | Swathi wrote · Vanessa runs |
| `infra/gcp_setup.sh` | Project, billing budget, APIs, bucket, service account, teammate IAM | Swathi |
| `notebooks/` | Model A training notebook, Gemini tuning notebook (next) | Swathi wrote · Mahitha/Bindu run |
| `schema/`, `results/` | Gemini JSON schema; committed metrics, curves, confusion matrices | |
| `tests/` | `pytest -q` | |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

Face detection for the dynamic crop: download MediaPipe's
`blaze_face_short_range.tflite` and pass `--face-model path/to/it`. Without it the scripts fall back
to OpenCV's Haar cascade, and if that is missing too, to a center crop (rows are marked
`center_fallback` in `labels.csv`).

## Mahitha — Wed: crop choice, frames, labels

```bash
# 1. crop comparison on 3–4 clips (include texting held low) → post the images on Discord
python scripts/crop_compare.py ~/Drive/DriveAware/Team_raw/*texting*.mp4 --face-model blaze_face_short_range.tflite

# 2. frames for everything (team + public); held-out members land in split=test automatically
python scripts/extract_frames.py \
  --team ~/Drive/DriveAware/Team_raw ~/Drive/DriveAware/Team_test_only \
  --public ~/Drive/DriveAware/StateFarm_raw/imgs/train --public-source statefarm \
  --out ~/Drive/DriveAware/Frames --crop dynamic --fps 2 --held-out mahitha,vanessa \
  --face-model blaze_face_short_range.tflite
```
Then fix the `subclass` column in `Frames/labels.csv`, and commit `labels.csv` + `class_counts.csv`
to `data/`. Clip names must be `name_activity_condition_tN.mp4` (lowercase); badly named clips are
skipped with a message.

**Public-label mapping** (`driveaware_data/labels.py`): State Farm c0 safe, c1/c3 texting,
c2/c4 phone_ear, c6 drinking_cup, c8 hand_forehead; c5 radio, c7 reaching behind, c9 passenger are
dropped. Public data is train/val only; **test = held-out team members only**.

## Vanessa — event CSV questions (answered)

1. **`event_type=none` for normal driving?** Yes. Every `none` row is a stretch of plain driving;
   any alert inside it counts as a false alarm, and the total `none` time is the denominator for
   *false alarms per 10 min*. Make the `none` rows add up to ≥ 8 min across the 8 sessions.
2. **Partial occlusion / mount angle?** Keep `event_type` as the real behaviour and put tags in
   `notes`: `occlusion=hand; mount=B`. The script reports recall broken down by every tag key it
   finds, so you get occlusion and mount-angle results for free.
3. **Filenames starting with the session id?** Yes, `S01_mahitha_day-none.mp4` is fine. Name the
   CSV `S01.csv` and the exported app log `S01_log_*.csv` so `--script-dir/--log-dir` pairs them.
4. **`start_s`/`end_s` used as written?** Yes, seconds from video start. A detection counts if the
   app alert of the same type starts between `start_s − 1 s` and `end_s + 3 s`
   (`--pre/--post`). Detection delay = alert time − `start_s`. Start the app and the recording
   together; if they're off, pass `--offset`.
5. `mirror_check` rows must **not** alert — an alert there is reported as a false alarm.

```bash
python eval/match_events.py --script-dir ../driveaware-app/tests/scripts \
  --log-dir ~/Drive/DriveAware/Test_sessions --out results/system
```
Outputs `per_behavior.csv`, `summary.csv`, `by_condition.csv`, `matched_events.csv` (add a
`failed_stage` column for misses: `face_not_found / wrong_class / rule_threshold /
wrong_gemini_verdict`) and `classified_alerts.csv`.

## Google Cloud

```bash
PROJECT_ID=driveaware-cs663 BILLING_ACCOUNT=XXXXXX-XXXXXX-XXXXXX bash infra/gcp_setup.sh
```
Creates the project, a $50 budget alert, enables Vertex AI / Cloud Run / Cloud Build / Artifact
Registry / Cloud Storage, bucket `gs://<project>-data`, the `driveaware-backend` service account
(roles/aiplatform.user, no keys) and gives teammates Vertex AI Studio + bucket access.

**Gemini model for tuning (task #4, confirmed Nov 11, 2026):**

| Field | Value |
|---|---|
| Tunable model ID | `gemini-3.5-flash` (GA on Vertex AI, retirement May 19, 2027 or later) |
| Tuning location | `us` multi-region (3.5 Flash tunes on `us` / `eu` only) |
| Tuned-model serving | `us` (also `eu`, `global`) |
| Thinking | `MINIMAL` for the tuned task |
| Limits | 131,072 tokens per example; up to 300K multimodal examples; JSONL ≤ 1 GB |

`gemini-2.5-flash` was retired on Vertex AI on Oct 20, 2026, so it can't be used for the base
check or tuning. Use **Vertex AI Studio** (console.cloud.google.com) for the base check, not
Google AI Studio, where `gemini-3.5-flash` is deprecated and silently routed to 3.6 Flash.

## Still to build (in plan order)
- Gemini label-sheet builder (replay clips → keyframes + signal summaries → Google Sheet)
- Model A training notebook (Model Maker EfficientNet-Lite0) + CLIP zero-shot and frozen-backbone baselines; Keras MobileNetV3 fallback
- Model A evaluation script
- Sheet → JSONL converter (uses `build_prompt()` from the backend so train == serve) + upload
- Base-vs-tuned Gemini evaluation script
