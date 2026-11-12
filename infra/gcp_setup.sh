#!/usr/bin/env bash
# One-time Google Cloud setup for DriveAware (Swathi tasks #2–#3).
# Usage:
#   PROJECT_ID=driveaware-cs663-swathi BILLING_ACCOUNT=XXXXXX-XXXXXX-XXXXXX \
#   TEAM_EMAILS="teammate1@example.edu teammate2@example.edu" bash infra/gcp_setup.sh
# TEAM_EMAILS is passed at run time so no personal addresses live in the repo.
set -euo pipefail
: "${PROJECT_ID:?}"; : "${BILLING_ACCOUNT:?}"
REGION="${REGION:-us-central1}"
BUCKET="gs://${PROJECT_ID}-data"
SA="driveaware-backend@${PROJECT_ID}.iam.gserviceaccount.com"
read -r -a TEAM <<< "${TEAM_EMAILS:-}"   # space-separated; empty = skip teammate access

gcloud projects create "$PROJECT_ID" --name="DriveAware CS663" || true
gcloud billing projects link "$PROJECT_ID" --billing-account="$BILLING_ACCOUNT"
gcloud config set project "$PROJECT_ID"

# Budget alert at $50 (50/90/100 %)
gcloud billing budgets create --billing-account="$BILLING_ACCOUNT" \
  --display-name="DriveAware budget" --budget-amount=50USD \
  --filter-projects="projects/${PROJECT_ID}" \
  --threshold-rule=percent=0.5 --threshold-rule=percent=0.9 --threshold-rule=percent=1.0 || true

gcloud services enable aiplatform.googleapis.com run.googleapis.com cloudbuild.googleapis.com \
  artifactregistry.googleapis.com storage.googleapis.com iam.googleapis.com

gcloud storage buckets create "$BUCKET" --location="$REGION" --uniform-bucket-level-access || true

# Least-privilege backend identity: call Vertex AI, read the bucket. No keys are created.
gcloud iam service-accounts create driveaware-backend --display-name="DriveAware backend" || true
gcloud projects add-iam-policy-binding "$PROJECT_ID" --member="serviceAccount:${SA}" --role=roles/aiplatform.user --condition=None
gcloud storage buckets add-iam-policy-binding "$BUCKET" --member="serviceAccount:${SA}" --role=roles/storage.objectViewer

# Teammates: Vertex AI Studio (Bindu's base-Gemini check) + bucket read/write for keyframes
for u in "${TEAM[@]+"${TEAM[@]}"}"; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" --member="user:${u}" --role=roles/aiplatform.user --condition=None
  gcloud storage buckets add-iam-policy-binding "$BUCKET" --member="user:${u}" --role=roles/storage.objectAdmin
done

echo "Done. Bucket: $BUCKET   Backend SA: $SA"
echo "Now confirm the tunable Gemini Flash model + region in the Vertex AI tuning docs and record it in README.md."
