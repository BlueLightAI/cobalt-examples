from abc import ABC, abstractmethod
from typing import List


class TextEmbeddingModel(ABC):
    """Base Class for Text Embedding Models."""

    @abstractmethod
    def embed(self, texts: List[str], device: str, batch_size: int):
        """Embed method implemented by all embedding models."""


class SentenceTransformerEmbeddingModel(TextEmbeddingModel):
    """You can find all embeddings available at https://www.sbert.net/docs/pretrained_models.html."""

    def __init__(self, model_id: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "The sentence_transformers package must be installed "
                "to create text embeddings. Run `pip install sentence_transformers`."
            ) from e

        self._model = SentenceTransformer(model_id)
        super().__init__()

    def embed(
        self,
        texts: List[str],
        device: str,
        batch_size: int = 64,
    ):
        """Returns embeddings of texts.

        Args:
            texts (List[str]): The texts that you want to embed.
            device (str): Device to run model embedding computation on.
            batch_size (int): Batch Size.

        Returns:
            np.ndarray: Embedding.

        Example:
            >>> embedding = SentenceTransformerEmbeddingModel().embed([
                "My friend has a dog",
                "My friend has a cat"
            ], device = 'cpu')
            >>> embedding.shape
            (2, 384)

        Notes:
            - Be careful about sequence length of the models.
            - Consider sequence length while choosing batch size.
            - Consult https://www.sbert.net/docs/pretrained_models.html for performance.

        """
        embedding = self._model.encode(
            texts, show_progress_bar=True, device=device, batch_size=batch_size
        )
        return embedding
