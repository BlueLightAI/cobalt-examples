# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from cobalt.schema import CobaltDataset, Embedding
from cobalt.schema.dataset import CobaltDataSubset


class Embeddable(ABC):
    @abstractmethod
    def embed(self, X: np.ndarray) -> np.ndarray:
        """Returns the embedding for an array of input vectors."""


def sample_for_unsupervised_rf_embedding(
    X: np.ndarray, n_samples: Optional[int] = None
):
    if n_samples is None:
        n_samples = X.shape[0] // 2
    # sample (with replacement) from the original array to half the size
    sample_idx = np.random.randint(0, X.shape[0], n_samples)
    X_sample = X[sample_idx, :]

    # Create a new array of synthetic data where we sample independently from each column.
    # Each column has the same distribution as before, but marginals are independent.
    X_marginal_sample = np.empty_like(X_sample)
    for i in range(X.shape[1]):
        sample_idx = np.random.randint(0, X.shape[0], n_samples)
        X_marginal_sample[:, i] = X[sample_idx, i]

    X_train = np.vstack((X_sample, X_marginal_sample))
    y_train = np.zeros(X_train.shape[0], dtype=np.bool_)
    y_train[:n_samples] = True
    return X_train, y_train


# TODO: different backends (esp LightGBM which is much faster)
# TODO: support for categorical input variables
# TODO: support for regression tasks
# TODO: pass in parameters
# TODO: make it convenient to train on a subset of an embedding
# TODO: make it convenient to create an ArrayEmbedding by applying to new data
class RFEmbedding(Embedding, Embeddable):
    def __init__(
        self,
        train_X: np.ndarray,
        train_y: np.ndarray,
        pred_X: np.ndarray,
        name: Optional[str],
    ):
        self.train_X = train_X
        self.train_y = train_y
        self.pred_X = pred_X
        self.model: Optional[RandomForestClassifier] = None
        self.emb: Optional[np.ndarray] = None
        super().__init__(name)

    @classmethod
    def from_dataset(
        cls,
        dataset: CobaltDataset,
        input_cols: List[str],
        objective: Optional[Union[str, pd.Series]] = None,
        name: Optional[str] = None,
        **kwargs,
    ) -> RFEmbedding:
        """Creates a random forest embedding from the specified input columns of a CobaltDataset.

        If the provided dataset object is a CobaltDataSubset, uses only this
        subset to train the embedding, but embeds the whole dataset.

        If a training objective is provided, it may either be a Pandas Series or
        a column name in the dataset.
        """
        train_X = dataset.df[input_cols].to_numpy()
        if isinstance(dataset, CobaltDataSubset):
            root_dataset = dataset.source_dataset
            pred_X = root_dataset.df[input_cols].to_numpy()
        else:
            pred_X = train_X

        if isinstance(objective, str):
            train_y = dataset.df[objective].to_numpy()
        elif objective is None:
            train_X, train_y = sample_for_unsupervised_rf_embedding(train_X)
        else:
            train_y = objective.to_numpy()

        if name is None:
            obj_name = objective.name if isinstance(objective, pd.Series) else None
            name = f"rf_{dataset.name}_{obj_name}" if obj_name else f"rf_{dataset.name}"

        return RFEmbedding(train_X, train_y, pred_X, name)

    @classmethod
    def from_embedding(
        cls,
        embedding: Embedding,
        objective: Optional[Union[pd.Series, np.ndarray]] = None,
        name: Optional[str] = None,
        **kwargs,
    ) -> RFEmbedding:
        """Creates a random forest embedding based on an already created Embedding.

        If a training objective is provided, it may either be a Pandas Series or
        a column name in the dataset.
        """
        # TODO: this wastes memory
        X = np.array(embedding.get())
        if objective is not None:
            # TODO: branch based on objective dtype
            train_X = X
            train_y = np.array(objective)
        else:
            train_X, train_y = sample_for_unsupervised_rf_embedding(X)

        if name is None:
            obj_name = objective.name if isinstance(objective, pd.Series) else None
            name = (
                f"rf_{embedding.name}_{obj_name}"
                if obj_name
                else f"rf_{embedding.name}"
            )

        return RFEmbedding(train_X, train_y, X, name)

    def _train_model(self, params=None):
        if params is None:
            params = {}
        model = RandomForestClassifier(max_depth=7, n_jobs=-1, **params)
        model.fit(self.train_X, self.train_y)
        return model

    def fit(self):
        self.model = self._train_model()

    def embed(self, X: np.ndarray) -> np.ndarray:
        return self.model.apply(X)

    def get_available_distance_metrics(self) -> List[str]:
        return ["hamming"]

    @property
    def default_distance_metric(self) -> str:
        return "hamming"

    @property
    def admissible_distance_metrics(self) -> List[str]:
        return ["hamming"]

    def get(self, indices=None) -> np.ndarray:
        if self.model is None:
            self.model = self._train_model()
        if self.emb is None:
            self.emb = self.embed(self.pred_X)
        if indices is not None:
            return self.emb[indices, :]

        return self.emb

    def group_classifier(self, indices: np.ndarray):
        """Builds a classifier for the selected groups based on the embedding model."""
        # TODO
