"""
HuggingFace embedding model wrapper using sentence-transformers.
Encodes code chunks into dense vectors for similarity search.
"""

from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL_NAME, EMBEDDING_DIMENSION


class Embedder:
    """Wraps a sentence-transformers model for generating embeddings."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        print(f"  Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = EMBEDDING_DIMENSION
        print(f"  ✓ Embedding model loaded ({self.dimension}-dim)")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embeds a batch of texts and returns their vector representations."""
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embeds a single query string."""
        embedding = self.model.encode(
            query,
            normalize_embeddings=True,
        )
        return embedding.tolist()
