# Copyright (C) 2023-2025 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

from __future__ import annotations

import json
import re
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional

import numpy as np

if TYPE_CHECKING:
    from cobalt.schema.dataset import DatasetBase


class Embedding(ABC):
    """Encapsulates metadata about a dataset embedding."""

    def __init__(self, name=None) -> None:
        self._name = name

    @property
    def name(self):
        if self._name is not None:
            return self._name
        else:
            return str(self)

    @property
    @abstractmethod
    def dimension(self) -> int:
        """The dimension of the embedding."""

    @abstractmethod
    def get(self, dataset: DatasetBase) -> np.ndarray:
        """Get the values of this embedding for a dataset."""

    @property
    def distance_metrics(self) -> List[str]:
        """Suggested distance metrics for use with this embedding."""
        warnings.warn(
            "Embedding.distance_metrics is deprecated; "
            "use default_distance_metrics or admissible_distance_metrics instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get_available_distance_metrics()

    @property
    @abstractmethod
    def default_distance_metric(self) -> str:
        """Default distance metric to use with this embedding."""

    @property
    def admissible_distance_metrics(self) -> List[str]:
        """Distance metrics that are reasonable to use with this embedding.

        Other distance metrics may still be useful, but these are metrics that
        are known to make sense for the data.
        """
        return self.get_available_distance_metrics()

    @abstractmethod
    def get_available_distance_metrics(self) -> List[str]:
        """Return the list of distance metrics that could be used."""

    def __repr__(self):
        if self._name is not None:
            return f'Embedding(dim={self.dimension}, name="{self.name}")'
        return f"Embedding(dim={self.dimension})"


class ColumnEmbedding(Embedding):
    """Represents an embedding as a column range.

    Attributes:
        columns: List of strings naming the columns to include in this
            embedding.
    """

    def __init__(self, columns: List[str], metric: str, name=None) -> None:
        self._metric = metric
        self.columns = columns
        super().__init__(name)

    def get(self, dataset: DatasetBase) -> np.ndarray:
        """Return a np.ndarray of the embedding rows at specified indices.

        Only columns specified in the `columns` attribute are included.

        Args:
            dataset: Data(sub)set for which to get the embedding values.

        Returns:
            The np.ndarray containing the embedding values for the rows in the given dataset.
        """
        if not hasattr(dataset, "df"):
            raise ValueError(
                "Embedding.get() takes the requested dataset or subset as its argument, "
                "not a list of indices."
            )
        selected_col_data = dataset.df[self.columns]

        return selected_col_data.to_numpy()

    def get_available_distance_metrics(self) -> List[str]:
        warnings.warn(
            "Embedding.get_available_distance_metrics() is deprecated; "
            "use default_distance_metrics or admissible_distance_metrics instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return [self._metric]

    @property
    def default_distance_metric(self) -> str:
        return self._metric

    @property
    def metric(self) -> str:
        warnings.warn(
            "ColumnEmbedding.metric is deprecated; "
            "use default_distance_metrics or admissible_distance_metrics instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._metric

    @property
    def admissible_distance_metrics(self) -> List[str]:
        return [self._metric]

    @property
    def dimension(self) -> int:
        return len(self.columns)

    def __eq__(self, other) -> bool:
        if not isinstance(other, ColumnEmbedding):
            return NotImplemented

        return all(
            [
                self.columns == other.columns,
                self._metric == other._metric,
                self.name == other.name,
            ]
        )

    def to_dict(self) -> dict:
        return {
            "columns": self.columns,
            "metric": self._metric,
            "name": self.name,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, serialized_data) -> ColumnEmbedding:
        data = json.loads(serialized_data)
        return cls(**data)

    @classmethod
    def from_dict(cls, data: dict) -> ColumnEmbedding:
        instance = cls(**data)
        return instance


# TODO: add support for multiple admissible metrics
class ArrayEmbedding(Embedding):
    """An embedding stored in an array associated with a Dataset.

    Attributes:
        array_name: The name of the array in the dataset storing the embedding values
    """

    def __init__(
        self, array_name: str, dimension: int, metric: str, name: Optional[str] = None
    ) -> None:
        self._metric = metric
        self._dimension = dimension
        self.array_name = array_name
        super().__init__(name)

    @property
    def dimension(self) -> int:
        """The dimension of the embedding."""
        return self._dimension

    def get(self, dataset: DatasetBase) -> np.ndarray:
        """Return a np.ndarray of the embedding rows at specified indices.

        Args:
            dataset: Data(sub)set for which to get the embedding values.

        Returns:
            The np.ndarray containing the embedding values for the rows in the given dataset.
        """
        if not hasattr(dataset, "get_array"):
            raise ValueError(
                "Embedding.get() takes the requested dataset or subset as its argument, "
                "not a list of indices."
            )
        return dataset.get_array(self.array_name)

    def get_available_distance_metrics(self) -> List[str]:
        warnings.warn(
            "Embedding.get_available_distance_metrics() is deprecated; "
            "use default_distance_metrics or admissible_distance_metrics instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return [self._metric]

    @property
    def metric(self) -> str:
        warnings.warn(
            "ArrayEmbedding.metric is deprecated; "
            "use default_distance_metrics or admissible_distance_metrics instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._metric

    @property
    def default_distance_metric(self) -> str:
        return self._metric

    @property
    def admissible_distance_metrics(self) -> List[str]:
        return [self._metric]

    def __eq__(self, other) -> bool:
        if not isinstance(other, ArrayEmbedding):
            return NotImplemented

        return all(
            [
                self.array_name == other.array_name,
                self.dimension == other.dimension,
                self._metric == other._metric,
                self.name == other.name,
            ]
        )

    def to_dict(self) -> dict:
        return {
            "array_name": self.array_name,
            "dimension": self.dimension,
            "metric": self._metric,
            "name": self.name,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, serialized_data) -> ArrayEmbedding:
        data = json.loads(serialized_data)
        return cls(**data)

    @classmethod
    def from_dict(cls, data: dict) -> ArrayEmbedding:
        instance = cls(**data)
        return instance


class EmbeddingManager:
    """Manages embedding logic."""

    def __init__(self, embeddings: List[Embedding]):
        self._embeddings = embeddings if embeddings else []
        self.validate_embeddings(self._embeddings)
        self._embedding_name_to_index = {
            emb.name: i for i, emb in enumerate(self._embeddings)
        }

    def validate_embeddings(self, embeddings):
        self._validate_embedding_names(embeddings)

    @staticmethod
    def _validate_embedding_names(embeddings: List[Embedding]):
        unique_set = {emb.name for emb in embeddings}
        if len(unique_set) < len(embeddings):
            raise Exception("Embeddings have non-unique names.")

    def map_name_to_embedding(self, name):
        index = self._embedding_name_to_index[name]
        return self._embeddings[index]

    def add_embedding(self, embedding: Embedding):
        if embedding.name in self._embedding_name_to_index:
            raise Exception(f"Name {embedding.name!r} already used")

        self._embeddings.append(embedding)
        self._embedding_name_to_index[embedding.name] = len(self._embeddings) - 1

    def get_embedding_array_by_name(self, name: str, indices=None):
        if indices is None:
            return self.map_name_to_embedding(name).get()
        return self.map_name_to_embedding(name).get(indices)

    def embedding_names(self) -> List[str]:
        return [emb.name for emb in self._embeddings]

    def get_new_embedding_name(self) -> str:
        names = self.embedding_names()
        matches = [re.match(r"embedding_(\d+)", name) for name in names]
        numbers = [int(m.group(1)) for m in matches if m is not None]
        number = max(numbers) + 1 if numbers else 1
        return f"embedding_{number}"

    @property
    def embedding_metadata(self):
        return self._embeddings.copy()


INITIAL_EMBEDDING_INDEX = 0
