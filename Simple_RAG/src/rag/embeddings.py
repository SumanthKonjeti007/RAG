import os
import time
import json
from typing import List, Optional, Protocol

import numpy as np
import requests

try:
    # Optional dependency for Vertex AI; only required if provider is set to vertex
    from google.cloud import aiplatform
    _HAS_VERTEX = True
except Exception:
    _HAS_VERTEX = False


class EmbeddingsClient(Protocol):
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray: ...


class MistralEmbeddingsClient:
    """Thin client for Mistral embeddings API.

    Reads API key from MISTRAL_API_KEY env var unless provided.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        request_timeout_s: float = 30.0,
        max_retries: int = 3,
        retry_backoff_s: float = 1.0,
    ) -> None:
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY is not set")
        self.model = model or os.getenv("MISTRAL_EMBEDDINGS_MODEL", "mistral-embed")
        self.base_url = base_url or os.getenv("MISTRAL_API_BASE", "https://api.mistral.ai")
        self.timeout = request_timeout_s
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        vectors: List[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            vectors.extend(self._embed_batch(batch))
        mat = np.vstack(vectors).astype(np.float32)
        return mat

    def _embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        url = f"{self.base_url}/v1/embeddings"
        payload = {"model": self.model, "input": texts}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=self.timeout)
                if resp.status_code == 200:
                    data = resp.json().get("data", [])
                    return [np.array(item["embedding"], dtype=np.float32) for item in data]
                # Retry on 429/5xx
                if resp.status_code in (429, 500, 502, 503, 504):
                    time.sleep(self.retry_backoff_s * (2 ** attempt))
                    continue
                raise RuntimeError(f"Embeddings API error {resp.status_code}: {resp.text}")
            except requests.RequestException as e:
                last_exc = e
                time.sleep(self.retry_backoff_s * (2 ** attempt))
        raise RuntimeError(f"Failed to fetch embeddings after retries: {last_exc}")


class VertexAIEmbeddingsClient:
    """Embeddings via Google Cloud Vertex AI Text Embedding models.

    Uses ADC for authentication. Configure project/region/model via env vars:
      - VERTEX_PROJECT_ID
      - VERTEX_LOCATION (e.g., us-central1, europe-west4)
      - VERTEX_EMBEDDINGS_MODEL (default: textembedding-gecko@003)
    """

    def __init__(
        self,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        if not _HAS_VERTEX:
            raise RuntimeError("google-cloud-aiplatform is not installed; add it to requirements.txt")

        self.project_id = project_id or os.getenv("VERTEX_PROJECT_ID")
        self.location = location or os.getenv("VERTEX_LOCATION", "us-central1")
        self.model_name = model_name or os.getenv("VERTEX_EMBEDDINGS_MODEL", "textembedding-gecko@003")

        if not self.project_id:
            raise ValueError("VERTEX_PROJECT_ID is not set")

        # Initialize only once; safe to call multiple times
        aiplatform.init(project=self.project_id, location=self.location)
        self._model = aiplatform.TextEmbeddingModel.from_pretrained(self.model_name)

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        vectors: List[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            embeddings = self._model.get_embeddings(batch)
            # API returns list of objects with .values
            vectors.extend([np.array(e.values, dtype=np.float32) for e in embeddings])
        return np.vstack(vectors).astype(np.float32)


def get_embeddings_client() -> EmbeddingsClient:
    """Factory to select embeddings provider via env EMBEDDINGS_PROVIDER.

    Supported providers:
      - mistral (default)
      - vertex
    """
    provider = (os.getenv("EMBEDDINGS_PROVIDER", "mistral") or "mistral").lower()
    if provider == "vertex":
        return VertexAIEmbeddingsClient()
    return MistralEmbeddingsClient()
