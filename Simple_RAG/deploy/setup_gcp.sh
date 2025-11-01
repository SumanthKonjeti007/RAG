#!/usr/bin/env bash
set -euo pipefail

: "${GCP_PROJECT_ID:?set GCP_PROJECT_ID}"
: "${REGION:=us-central1}"

echo "Setting default project and region..."
gcloud config set project "$GCP_PROJECT_ID"
gcloud config set run/region "$REGION"

echo "Enabling required APIs..."
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  aiplatform.googleapis.com \
  cloudbuild.googleapis.com

echo "Auth: obtaining Application Default Credentials (follow browser prompt if needed)"
gcloud auth application-default login || true

echo "Done. You can now deploy with deploy/cloudrun_deploy.sh"


