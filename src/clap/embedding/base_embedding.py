import abc
from typing import List, Protocol

class EmbeddingFunctionInterface(Protocol):
    """
    A protocol for embedding functions to ensure they can provide
    their output dimensionality and embed documents.
    """

    @abc.abstractmethod
    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        Embeds a list of texts.

        Args:
            input: A list of document texts.

        Returns:
            A list of embeddings (list of floats).
        """
        ...

    @abc.abstractmethod
    def get_embedding_dimension(self) -> int:
        """
        Returns the dimensionality of the embeddings produced by this function.
        """
        ...