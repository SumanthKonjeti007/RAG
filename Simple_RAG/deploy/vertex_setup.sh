#!/usr/bin/env bash
set -euo pipefail

: "${GCP_PROJECT_ID:?set GCP_PROJECT_ID}"
: "${REGION:=us-central1}"

echo "Enabling Vertex AI API..."
gcloud services enable aiplatform.googleapis.com --project "$GCP_PROJECT_ID"

echo "Setting environment examples for Vertex AI embeddings:"
cat <<EOF
Export the following before deploying to use Vertex embeddings:

export EMBEDDINGS_PROVIDER=vertex
export GCP_PROJECT_ID=${GCP_PROJECT_ID}
export VERTEX_PROJECT_ID=${GCP_PROJECT_ID}
export VERTEX_LOCATION=${REGION}
export VERTEX_EMBEDDINGS_MODEL=textembedding-gecko@003

# If using OpenAI for generation:
# export OPENAI_API_KEY=sk-... 
# export OPENAI_MODEL=gpt-4o-mini
EOF

echo "Done."


