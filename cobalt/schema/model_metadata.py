# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

from __future__ import annotations

import json
import uuid
from collections import defaultdict
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Union,
)

import numpy as np
import pandas as pd

from cobalt.schema.evaluation_metric import (
    AbsoluteError,
    ClassificationAccuracy,
    ClassificationInaccuracy,
    ColumnEvaluationMetric,
    EvaluationMetric,
    SquaredError,
    json_hook,
)

if TYPE_CHECKING:
    from cobalt.schema.dataset import DatasetBase


class ModelTask(Enum):
    classification = "classification"
    regression = "regression"
    custom = "custom"  # does no performance metrics by default


def classification_model_metrics(model: ModelMetadata) -> Dict[str, EvaluationMetric]:
    elements: List[EvaluationMetric] = []
    if model.outcome_column and model.prediction_column:
        elements = [
            ClassificationInaccuracy(model.prediction_column, model.outcome_column),
            ClassificationAccuracy(model.prediction_column, model.outcome_column),
        ]
    metrics = {element.get_key(): element for element in elements}
    return metrics


def regression_model_metrics(model: ModelMetadata) -> Dict[str, EvaluationMetric]:
    elements: List[EvaluationMetric] = []
    if model.outcome_column and model.prediction_column:
        elements = [
            AbsoluteError(model.prediction_column, model.outcome_column),
            SquaredError(model.prediction_column, model.outcome_column),
        ]
    metrics = {element.get_key(): element for element in elements}
    return metrics


model_task_evaluation_metric_generators: Dict[
    ModelTask, Callable[[ModelMetadata], Dict[str, EvaluationMetric]]
] = {
    ModelTask.classification: classification_model_metrics,
    ModelTask.regression: regression_model_metrics,
    ModelTask.custom: lambda x: {},
}


class ModelMetadata:
    """Information about a model and its relationship to a dataset.

    Stores information about the model's inputs and outputs (as names of columns
    in the dataset), as well as ground truth data. Provides access to model
    performance metrics.

    Attributes:
        name: An optional name for the model.
        task: The task performed by the model. Can be "classification",
            "regression", or "custom" (the default). This determines which
            performance metrics are available by default.
        input_columns: A list of column(s) in the dataset containing the input
            data for the model.
        prediction_columns: A list of column(s) in the dataset containing the
            outputs produced by the model.
        outcome_columns: A list of column(s) in the dataset containing the
            target outputs for the model.
    """

    def __init__(
        self,
        outcome_columns: List[str],
        prediction_columns: List[str],
        task: ModelTask,
        input_columns: Optional[List[str]] = None,
        error_columns: Optional[List[str]] = None,
        evaluation_metrics: Optional[Sequence[Union[EvaluationMetric, Dict]]] = None,
        name: Optional[str] = None,
    ) -> None:
        self.outcome_columns = outcome_columns
        self.prediction_columns = prediction_columns
        self.input_columns = input_columns
        self.task = ModelTask(task)
        self.error_columns = [] if error_columns is None else error_columns
        self.name = (
            name
            if name is not None
            else f"model_{self.task.value}_{str(uuid.uuid4())[:8]}"
        )
        # get evaluation metrics for the specified task
        self.evaluation_metrics = model_task_evaluation_metric_generators[self.task](
            self
        )
        # add any user-specified evaluation metrics
        if evaluation_metrics:
            for m in evaluation_metrics:
                if isinstance(m, EvaluationMetric):
                    self.evaluation_metrics[m.get_key()] = m
                elif isinstance(m, dict):
                    metric = ColumnEvaluationMetric(**m)
                    self.evaluation_metrics[metric.get_key()] = metric
                else:
                    raise ValueError(f"Invalid evaluation metric specification {m}.")
        if error_columns:
            self.evaluation_metrics.update(
                {col: ColumnEvaluationMetric(col, col) for col in error_columns}
            )

    def __eq__(self, other) -> bool:
        if not isinstance(other, ModelMetadata):
            return NotImplemented

        return all(
            [
                self.outcome_columns == other.outcome_columns,
                self.prediction_columns == other.prediction_columns,
                self.input_columns == other.input_columns,
                self.task == other.task,
                self.error_columns == other.error_columns,
                self.evaluation_metrics == other.evaluation_metrics,
                self.name == other.name,
            ]
        )

    def to_dict(self) -> dict:
        evaluation_metrics = {}
        for metric_key, metric_value in self.evaluation_metrics.items():
            evaluation_metrics[metric_key] = metric_value.to_dict()
        return {
            "outcome_columns": self.outcome_columns,
            "prediction_columns": self.prediction_columns,
            "input_columns": self.input_columns,
            "error_columns": self.error_columns,
            "evaluation_metrics": evaluation_metrics,
            "name": self.name,
            "task": self.task.value,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, serialized_data) -> ModelMetadata:
        data = json.loads(serialized_data, object_hook=json_hook)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> ModelMetadata:
        data["task"] = ModelTask(data["task"])
        evaluation_metrics = data.pop("evaluation_metrics", None)
        instance = cls(**data)
        # TODO: this is a workaround because dataset deserialization does not call from_json()
        if evaluation_metrics:
            evaluation_metrics = {
                key: (
                    json_hook(metric)
                    if not isinstance(metric, EvaluationMetric)
                    else metric
                )
                for key, metric in evaluation_metrics.items()
            }

        instance.evaluation_metrics = evaluation_metrics if evaluation_metrics else {}
        return instance

    @property
    def prediction_column(self):
        """Returns the first prediction column if len(prediction_columns) > 0, else None."""
        if len(self.prediction_columns) > 0:
            return self.prediction_columns[0]
        return None

    @property
    def outcome_column(self):
        """Returns the first outcome column if len(outcome_columns) > 0, else None."""
        if len(self.outcome_columns) > 0:
            return self.outcome_columns[0]
        return None

    def add_metric_column(
        self, metric_name: str, column: str, lower_values_are_better: bool = True
    ):
        """Add a column from the dataset as a performance metric for this model.

        Args:
            metric_name: The name for the metric. If you want to compare
                different models using this metric, use the same name for the metric
                in each.
            column: The name of the column in the dataset that contains the
                values of this metric for the model.
            lower_values_are_better: Whether lower or higher values of the
                metric indicate better performance.
        """
        self.evaluation_metrics[metric_name] = ColumnEvaluationMetric(
            metric_name, column, lower_values_are_better
        )

    def performance_metrics(self) -> Dict[str, EvaluationMetric]:
        """Return the relevant performance metrics for this model.

        The returned objects have a ``calculate()`` method, which computes
        pointwise performance metrics, and an ``overall_score()`` method, which
        computes the overall performance for a group. These methods accept
        ``CobaltDataSubset`` objects and return dictionaries mapping metric
        names to values.
        """
        return self.evaluation_metrics

    def performance_metric_keys(self) -> List[str]:
        return list(self.performance_metrics().keys())

    def get_performance_metric_for(self, key: str) -> EvaluationMetric:
        if key not in self.performance_metric_keys():
            raise ValueError(
                f"Metric {key} not supported. Use one of {self.performance_metric_keys()}"
            )
        return self.performance_metrics()[key]

    def calculate_performance_metric(
        self, metric_name: str, dataset: DatasetBase
    ) -> np.ndarray:
        method = self.get_performance_metric_for(metric_name)
        return np.array(method.calculate(dataset)[metric_name])

    def overall_performance_metric(
        self, metric_name: str, dataset: DatasetBase
    ) -> float:
        metric = self.get_performance_metric_for(metric_name)
        return metric.overall_score(dataset)[metric.get_key()]

    def overall_performance_metrics(self, dataset: DatasetBase) -> Dict[str, float]:
        result = {
            metric.get_key(): metric.overall_score(dataset)[metric.get_key()]
            for metric in self.performance_metrics().values()
        }
        return result

    def get_confusion_matrix(
        self,
        dataset: DatasetBase,
        normalize_mode: Union[bool, Literal["all", "index", "columns"]] = "index",
        selected_classes: Optional[List[str]] = None,
    ) -> Optional[pd.DataFrame]:
        """Calculate the confusion matrix for the model if applicable.

        Args:
            dataset: The dataset containing the outcomes and predictions.
            normalize_mode: Specifies the normalization mode for the confusion matrix.
            selected_classes: Specifies the classes to include in the matrix, with all others
                aggregated as "other".

        Returns:
            Optional[pd.DataFrame]: Confusion matrix as a DataFrame, or None if not applicable.

        Raises:
            ValueError: If the model task is not classification.
        """
        # TODO: find a better way to handle this polymorphism
        # Right now this seems to be ok because there is only one method impacted,
        # but if there is any additional functionality that varies by model task,
        # we should seriously consider creating one class for each model task.
        if self.task != ModelTask.classification:
            raise ValueError(
                f"The confusion matrix cannot be calculated for model type ({self.task.value})."
            )
        if not self.prediction_column or not self.outcome_column:
            raise ValueError(
                "Model overview statistics are available only if the model "
                "attributes for outcome and prediction columns have been set."
            )

        y_true = dataset.df[self.outcome_column]
        x_pred = dataset.df[self.prediction_column]
        categories = set(y_true.unique()).union(x_pred.unique())

        if selected_classes is not None and len(selected_classes) < len(categories):
            categories = [*selected_classes, "other"]

        y_true = y_true.astype("category").cat.set_categories(categories)
        x_pred = x_pred.astype("category").cat.set_categories(categories)

        true_class_idx = y_true.isin(categories)
        y_true[~true_class_idx] = "other"

        pred_class_idx = x_pred.isin(categories)
        x_pred[~pred_class_idx] = "other"

        confusion_matrix = pd.crosstab(
            y_true,
            x_pred,
            rownames=["True Label"],
            colnames=["Predicted Label"],
            normalize=normalize_mode,
            dropna=False,
        )

        return confusion_matrix

    def get_statistic_metrics(
        self, dataset: DatasetBase, selected_classes: Optional[List[str]] = None
    ):
        """Return a DataFrame containing recall, precision, F1 score, and accuracy for each class.

        This method uses the model's confusion matrix and can filter metrics to only
        selected classes.
        Metrics calculated include recall, precision, F1 score, and accuracy.

        Args:
            dataset: The dataset to compute the confusion matrix.
            selected_classes: List of classes to include in the metrics calculation.
                If None, metrics for all classes are calculated.

        Returns:
            pd.DataFrame: A DataFrame with recall, precision, F1 score, and accuracy for each class.
        """
        metrics = defaultdict(list)

        conf_matrix = self.get_confusion_matrix(
            dataset, selected_classes=selected_classes
        )
        labels = (
            [*selected_classes, "other"]
            if selected_classes is not None
            and len(selected_classes) < conf_matrix.columns.size
            else conf_matrix.columns
        )

        def confusion_matrix_for(class_idx, matrix_df):
            TP = matrix_df[class_idx, class_idx]
            FN = matrix_df[class_idx].sum() - TP
            FP = matrix_df[:, class_idx].sum() - TP
            TN = matrix_df.sum() - TP - FN - FP
            return TP, FN, FP, TN

        conf_matrix = conf_matrix.to_numpy()
        for class_index in range(conf_matrix.shape[0]):
            TP, FN, FP, TN = confusion_matrix_for(class_index, conf_matrix)
            recall = TP / (TP + FN) if TP + FN != 0 else 0
            precision = TP / (TP + FP) if TP + FP != 0 else 0
            f1_score = (
                2 * (precision * recall) / (precision + recall)
                if precision + recall != 0
                else 0
            )
            accuracy = (TP + TN) / (TP + TN + FP + FN) if TP + TN + FP + FN != 0 else 0
            fpr = FP / (FP + TN) if FP + TN != 0 else 0

            metrics["Recall"].append(recall)
            metrics["Precision"].append(precision)
            metrics["F1"].append(f1_score)
            metrics["Accuracy"].append(accuracy)
            metrics["FPR"].append(fpr)

        return pd.DataFrame(metrics, index=labels)

    def __repr__(self):
        return (
            f"{type(self).__name__}(name={self.name}, "
            f"task={self.task}, "
            f"outcome_columns={self.outcome_columns}, "
            f"prediction_columns={self.prediction_columns}, "
            f"evaluation_metrics={self.evaluation_metrics})"
        )


class ModelMetadataCollection:
    """Stores an ordered collection of models retrievable by index or name."""

    def __init__(self, models: List[ModelMetadata]):
        self._models = list(models)

    def __iter__(self):
        return iter(self._models)

    def __getitem__(self, key: Union[int, str]) -> ModelMetadata:
        if isinstance(key, str):
            for model in self._models:
                if model.name == key:
                    return model
            raise KeyError(f"No model found with name {key!r}.")
        return self._models[key]

    def __setitem__(self, key: int, model: ModelMetadata):
        # TODO: set with name?
        self._models[key] = model

    def __len__(self):
        return len(self._models)

    def append(self, model: ModelMetadata):
        self._models.append(model)

    def remove(self, model: ModelMetadata):
        self._models.remove(model)

    def __eq__(self, other):
        return self._models == other._models
