#!/usr/bin/env bash
set -euo pipefail

# Required vars
: "${GCP_PROJECT_ID:?set GCP_PROJECT_ID}"
: "${REGION:=us-central1}"
: "${SERVICE_NAME:=simple-rag}"
: "${REPO:=simple-rag}"

IMAGE="${REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${REPO}/${SERVICE_NAME}:$(date +%Y%m%d-%H%M%S)"

echo "Enabling APIs..."
gcloud services enable run.googleapis.com artifactregistry.googleapis.com aiplatform.googleapis.com --project "$GCP_PROJECT_ID"

echo "Creating Artifact Registry repo (if not exists)..."
gcloud artifacts repositories create "$REPO" \
  --repository-format=docker \
  --location="$REGION" \
  --description="Simple RAG images" \
  --project "$GCP_PROJECT_ID" || true

echo "Building and pushing image: $IMAGE"
gcloud builds submit --tag "$IMAGE" --project "$GCP_PROJECT_ID" .

echo "Deploying to Cloud Run: $SERVICE_NAME"
gcloud run deploy "$SERVICE_NAME" \
  --image "$IMAGE" \
  --region "$REGION" \
  --platform managed \
  --allow-unauthenticated \
  --cpu=1 \
  --memory=1Gi \
  --min-instances=0 \
  --max-instances=3 \
  --port=8080 \
  --set-env-vars "EMBEDDINGS_PROVIDER=${EMBEDDINGS_PROVIDER:-mistral}" \
  --set-env-vars "OPENAI_API_KEY=${OPENAI_API_KEY:-}" \
  --set-env-vars "MISTRAL_API_KEY=${MISTRAL_API_KEY:-}" \
  --set-env-vars "OPENAI_MODEL=${OPENAI_MODEL:-gpt-4o-mini}" \
  --set-env-vars "VERTEX_PROJECT_ID=${VERTEX_PROJECT_ID:-$GCP_PROJECT_ID}" \
  --set-env-vars "VERTEX_LOCATION=${VERTEX_LOCATION:-$REGION}" \
  --set-env-vars "VERTEX_EMBEDDINGS_MODEL=${VERTEX_EMBEDDINGS_MODEL:-textembedding-gecko@003}" \
  --project "$GCP_PROJECT_ID"

echo "Done. Service URL:"
gcloud run services describe "$SERVICE_NAME" --region "$REGION" --format='value(status.url)' --project "$GCP_PROJECT_ID"


