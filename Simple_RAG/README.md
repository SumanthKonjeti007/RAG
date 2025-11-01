# Simple RAG Pipeline

A Retrieval-Augmented Generation (RAG) system built with FastAPI, using Mistral embeddings and custom vector search without external RAG frameworks.

## System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   PDF Upload    │    │   Text Chunking  │    │   Embeddings    │
│   via /ingest   │───▶│   Header-Aware   │───▶│   Mistral API   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐             │
│  User Query     │    │ Intent Detection │             ▼
│  via /chat      │───▶│ & Query Rewrite  │    ┌─────────────────┐
└─────────────────┘    └──────────────────┘    │ Vector Store +  │
                                │               │ TF-IDF Index    │
                                ▼               └─────────────────┘
┌─────────────────┐    ┌──────────────────┐             │
│ Chat Response   │    │ Hybrid Search &  │◀────────────┘
│ with Citations  │◀───│ Threshold Filter │
└─────────────────┘    └──────────────────┘
```

## Key Components

### 1. **Data Ingestion** (`POST /ingest`)
- Upload multiple PDF files via multipart form
- Extract text using `pypdf`
- **Header-aware chunking**: Basic header detection from chunk content using regex patterns
- Chunk size: ~100 tokens with 20% overlap
- Store embeddings using Mistral API

### 2. **Query Processing**
- **PII Detection**: Regex and keyword matching to block sensitive information
- **Intent classification**: OpenAI determines if query needs knowledge base
- **Query rewrite**: Expand with synonyms and key terms for better retrieval
- Skip RAG for chitchat queries or queries containing PII

### 3. **Hybrid Search**
- **Semantic search**: Mistral embeddings + cosine similarity
- **Keyword search**: Custom TF-IDF implementation (no external libs)
- **Score fusion**: 65% semantic + 35% TF-IDF with phrase boosting
- **Threshold filtering**: 0.3 minimum fused score (returns "insufficient evidence" if below)

### 4. **Response Generation**
- OpenAI GPT for natural language generation
- Context-aware responses with source citations
- Structured JSON output format

### 5. **Hallucination Detection**
- **Post-hoc evidence checking**: LLM analyzes response statements against retrieved chunks
- **Evidence rating**: 1-10 scale for each key factual claim
- **Risk assessment**: Overall hallucination risk (low/medium/high)
- **Focused analysis**: 3-5 most important statements with grouped similar claims

### 6. **Simple Web UI**
- Knowledge base table with upload functionality
- Side-by-side chat and retrieved chunks display
- Real-time chunk scores and source attribution
- Hallucination analysis panel with color-coded evidence ratings
- PII detection warnings and blocking

## Installation & Setup

### Prerequisites
- Python 3.8+
- OpenAI API key
- Mistral API key

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Environment Setup
Copy `src/api/ENV_EXAMPLE.txt` to `src/api/.env` and fill values. Minimal local setup:
```
EMBEDDINGS_PROVIDER=mistral
MISTRAL_API_KEY=your_mistral_key_here
OPENAI_API_KEY=your_openai_key_here
```

### Run the Server
```bash
python -m uvicorn src.api.server:app --host 127.0.0.1 --port 8000
```

Visit: http://127.0.0.1:8000

## API Endpoints

### `GET /files`
List processed PDF files and total count.

### `POST /ingest`
Upload PDF files for processing.
```bash
curl -X POST "http://127.0.0.1:8000/ingest" -F "files=@document.pdf"
```

### `POST /chat`
Query the knowledge base.
```bash
curl -X POST "http://127.0.0.1:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the SOC 2 trust service criteria?", "top_k": 5}'
```

**Response Format:**
```json
{
  "query": "What are the SOC 2 trust service criteria?",
  "response": "The SOC 2 framework includes five Trust Services Criteria...",
  "chunks_retrieved": [
    {
      "id": "doc.pdf:chunk:0",
      "score": 0.85,
      "text": "SOC 2 Trust Services Criteria include...",
      "source_file": "doc.pdf",
      "chunk_index": 0,
      "heading_path": "Security > Access Controls"
    }
  ],
  "hallucination_analysis": {
    "statements": [
      {
        "statement": "SOC 2 includes five trust service criteria",
        "evidence_rating": 9,
        "explanation": "Directly mentioned in context"
      }
    ],
    "overall_hallucination_risk": "low"
  },
  "pii_detection": {
    "has_pii": false,
    "risk_level": "none"
  }
}
```

### `GET /search`
Direct search endpoint (used internally by chat).

## Technical Implementation

### No External RAG Libraries
- **Vector Store**: Custom numpy-based storage with JSON metadata
- **TF-IDF**: Built from scratch using basic math operations
- **Chunking**: Basic header detection with character-based text splitting
- **Search**: Pure Python cosine similarity and score fusion

### Libraries Used
- **FastAPI**: Web framework and API endpoints
- **pydantic**: Data validation and serialization (FastAPI uses it internally, so using it for clean structure)
- **uvicorn**: ASGI server for running FastAPI
- **Mistral API**: Text embeddings via HTTP requests
- **OpenAI API**: Intent classification, query rewrite, response generation, hallucination detection
- **pypdf**: PDF text extraction
- **numpy**: Vector operations and similarity calculations
- **requests**: HTTP API calls to external services
- **python-dotenv**: Environment variable management

### Performance Features
- **Threshold filtering**: Prevents low-quality responses
- **Hybrid search**: Combines semantic and keyword matching
- **Header-aware chunking**: Preserves document structure
- **Batch embedding**: Efficient API usage
- **PII blocking**: Prevents processing of sensitive information
- **Hallucination detection**: Post-hoc evidence verification with focused analysis

## Project Structure

```
Simple-RAG/
├── src/
│   ├── api/
│   │   ├── server.py          # FastAPI endpoints
│   │   ├── templates/
│   │   │   └── index.html     # Web UI
│   │   └── .env              # API keys
│   └── rag/
│       ├── embeddings.py     # Mistral client
│       ├── vector_store.py   # Custom vector DB
│       ├── chunking.py       # Header-aware chunking
│       └── retrieval.py      # Hybrid search + TF-IDF
├── vector_db_files/          # PDF storage
├── index_pdf/               # Generated indices
├── requirements.txt
└── README.md
```

## Vertex AI (GCP) Embeddings Integration

- Set `EMBEDDINGS_PROVIDER=vertex` in `src/api/.env`.
- Ensure Google ADC is available locally: `gcloud auth application-default login`.
- Required envs:
  - `VERTEX_PROJECT_ID`: your GCP project ID
  - `VERTEX_LOCATION`: e.g., `us-central1`
  - `VERTEX_EMBEDDINGS_MODEL`: default `textembedding-gecko@003`

The app will automatically switch to Vertex AI for embeddings. OpenAI is still used for generation by default (configure `OPENAI_API_KEY`).

## Deploy to Google Cloud Run

Prerequisites:
- `gcloud` CLI installed and initialized
- Billing enabled on your project

One-time setup:
```bash
export GCP_PROJECT_ID=your-project
export REGION=us-central1
bash deploy/setup_gcp.sh
```

Build and deploy:
```bash
export GCP_PROJECT_ID=your-project
export REGION=us-central1
export SERVICE_NAME=simple-rag

# Choose embeddings provider and keys
export EMBEDDINGS_PROVIDER=vertex   # or mistral
export OPENAI_API_KEY=sk-...        # required for generation
export VERTEX_PROJECT_ID=$GCP_PROJECT_ID
export VERTEX_LOCATION=$REGION
export VERTEX_EMBEDDINGS_MODEL=textembedding-gecko@003

bash deploy/cloudrun_deploy.sh
```

The script will:
- Enable required APIs
- Build a Docker image with Cloud Build and push to Artifact Registry
- Deploy to Cloud Run with appropriate environment variables

View the service URL printed at the end, or run:
```bash
gcloud run services describe $SERVICE_NAME --region $REGION \
  --format='value(status.url)' --project $GCP_PROJECT_ID
```

Notes:
- Secrets should be managed with Cloud Run env vars or Secret Manager in production.
- For private services, remove `--allow-unauthenticated` and set IAM or IAP.

## Usage Examples

1. **Upload documents**: Use the web UI or POST to `/ingest`
2. **Ask questions**: 
   - "What are the SOC 2 security controls?"
   - "Explain data encryption requirements"
   - "Tell me about Ansh's machine learning experience"
3. **View results**: Chat responses with source citations and chunk scores

## Design Decisions

- **Mistral for embeddings**: High-quality semantic representations (developed using personal key)
- **OpenAI for generation**: Superior reasoning and structured outputs (developed using personal key)
- **Custom implementations**: Full control, no black-box dependencies
- **Threshold filtering**: Quality over quantity in responses
- **Header-aware chunking**: Extracts headers from chunk content for better organization
- **Hybrid search**: Balances semantic understanding with keyword precision
- **PII detection**: Regex + keyword matching for comprehensive coverage
- **Hallucination detection**: LLM-based evidence checking with focused analysis

## Limitations

- Single-language support (English)
- No user authentication
- In-memory TF-IDF (rebuilds on restart)
- Basic UI styling
- No conversation history persistence
- Hallucination detection limited to 3-5 key statements
- PII detection may have false positives/negatives