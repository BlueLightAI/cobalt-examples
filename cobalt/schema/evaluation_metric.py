# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Union

import numpy as np
import pandas as pd

# TODO: remove this once CobaltDataset does not depend on ModelMetadata or EvaluationMetric
if TYPE_CHECKING:
    from cobalt.schema.dataset import DatasetBase


class EvaluationMetric(ABC):
    """A metric that can be used to score a given dataset or subset."""

    lower_values_are_better: bool
    serialized_key = "metric_class_name"

    @abstractmethod
    def calculate(self, dataset: DatasetBase) -> Dict[str, np.ndarray]:
        """Calculate the values of the evaluation metric on the given dataset.

        Returns a dictionary of type Dict[str, np.ndarray] where each array has
        the same length as ``dataset``.
        """
        raise NotImplementedError("Subclasses must implement the calculate method")

    def overall_score(self, dataset: DatasetBase) -> Dict[str, float]:
        return {key: np.mean(values) for key, values in self.calculate(dataset).items()}

    @abstractmethod
    def get_key(self) -> str:
        raise NotImplementedError("subclasses must implement `get_key`")

    def get_class_name(self):
        return self.__class__.__name__

    @abstractmethod
    def to_dict(self) -> Dict: ...


# We use template subclassing here instead of composition so that the class name
# for implementations can be easily controlled. If we had a factory constructor
# for TargetPredMetrics, all classes would be returned with the same name absent
# some manipulation of the class internals.


class TargetPredMetric(EvaluationMetric):
    key: str

    def __init__(self, target_col: str, pred_col: str):
        self.target_col = target_col
        self.pred_col = pred_col

    @abstractmethod
    def _calculate(self, target: pd.Series, preds: pd.Series) -> Dict[str, np.ndarray]:
        """Calculate the pointwise performance metric from target and prediction values."""

    def calculate(self, dataset: DatasetBase):
        target = dataset.select_col(self.target_col)
        preds = dataset.select_col(self.pred_col)
        return self._calculate(target, preds)

    def __repr__(self):
        return (
            f"{self.get_class_name()}(target_col='{self.target_col}', "
            f"pred_col='{self.pred_col}')"
        )

    def to_dict(self):
        return {
            f"{self.serialized_key}": self.get_class_name(),
            "target_col": self.target_col,
            "pred_col": self.pred_col,
        }

    def __eq__(self, other) -> bool:
        if not isinstance(other, TargetPredMetric):
            return NotImplemented
        return all(
            [
                self.target_col == other.target_col,
                self.pred_col == other.pred_col,
                self.key == other.key,
            ]
        )

    def __hash__(self):
        return hash((self.target_col, self.pred_col, self.key))


class AbsoluteError(TargetPredMetric):
    key = "absolute error"
    lower_values_are_better = True

    def _calculate(
        self, target: Union[np.ndarray, pd.Series], preds: Union[np.ndarray, pd.Series]
    ):
        return {self.key: np.abs(target - preds)}

    def get_key(self):
        return self.key


class SquaredError(TargetPredMetric):
    key = "squared error"
    lower_values_are_better = True

    def _calculate(
        self, target: Union[np.ndarray, pd.Series], preds: Union[np.ndarray, pd.Series]
    ):
        return {self.key: (target - preds) ** 2}

    def get_key(self):
        return self.key


class ClassificationInaccuracy(TargetPredMetric):
    key = "error"
    lower_values_are_better = True

    def _calculate(
        self, target: Union[np.ndarray, pd.Series], preds: Union[np.ndarray, pd.Series]
    ):
        return {self.key: target != preds}

    def get_key(self):
        return self.key


class ClassificationAccuracy(TargetPredMetric):
    key = "accuracy"
    lower_values_are_better = False

    def _calculate(
        self, target: Union[np.ndarray, pd.Series], preds: Union[np.ndarray, pd.Series]
    ):
        return {self.key: target == preds}

    def get_key(self):
        return self.key


class ColumnEvaluationMetric(EvaluationMetric):
    def __init__(
        self,
        name: str,
        column: str,
        lower_values_are_better: bool = True,
    ):
        self.key = name
        self.column = column
        self.lower_values_are_better = lower_values_are_better

    def calculate(self, dataset: DatasetBase):
        return {self.key: dataset.select_col(self.column)}

    def get_key(self):
        return self.key

    def __repr__(self):
        return f"ColumnEvaluationMetric(name='{self.key}', column='{self.column}')"

    def to_dict(self):
        return {
            f"{self.serialized_key}": self.get_class_name(),
            "name": self.key,
            "column": self.column,
            "lower_values_are_better": self.lower_values_are_better,
        }

    def __eq__(self, other) -> bool:
        if not isinstance(other, ColumnEvaluationMetric):
            return NotImplemented

        return all(
            [
                self.key == other.key,
                self.column == other.column,
                self.lower_values_are_better == other.lower_values_are_better,
            ]
        )

    def __hash__(self):
        return hash((self.key, self.column, self.lower_values_are_better))


ALLOWED_METRICS = {
    "ColumnEvaluationMetric": ColumnEvaluationMetric,
    "ClassificationAccuracy": ClassificationAccuracy,
    "ClassificationInaccuracy": ClassificationInaccuracy,
    "SquaredError": SquaredError,
    "AbsoluteError": AbsoluteError,
}


def json_hook(dct: dict):
    metric_key = EvaluationMetric.serialized_key
    if metric_key in dct:
        metric_class = ALLOWED_METRICS.get(dct[metric_key])
        if not metric_class:
            raise ValueError(f"Metric name should be on of {ALLOWED_METRICS.keys()}")
        del dct[metric_key]
        return metric_class(**dct)
    return dct
