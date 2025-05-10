# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

from __future__ import annotations

import base64
import copy
import html
import json
import mimetypes
import os
import warnings
from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.sparse
from mapper.neighborgraph import np_from_bytes, np_to_bytes

from cobalt import schema
from cobalt.array_utils import array_dicts_equal
from cobalt.cobalt_types import ColumnDataType
from cobalt.config import debug_logger, settings
from cobalt.schema.embedding import (
    ArrayEmbedding,
    ColumnEmbedding,
    Embedding,
    EmbeddingManager,
)
from cobalt.schema.evaluation_metric import ColumnEvaluationMetric
from cobalt.schema.helpers import (
    indices_type_valid,
    raise_error_for_non_string_column_names,
    warn_if_index_not_unique,
)
from cobalt.schema.metadata import (
    DatasetMetadata,
    MediaInformationColumn,
    TextDataType,
    collect_columns_metadata,
    infer_column_type,
)
from cobalt.schema.model_metadata import (
    ModelMetadata,
    ModelMetadataCollection,
    ModelTask,
)
from cobalt.warnings_handler import WarningsHandler

# Typing for subsets
Indices = Union[np.ndarray, List[int]]

# Ignore warning invalid value encountered in scalar divide
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy.core._methods")

# Ignore warning scores.mean() for empty values
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, module="cobalt.schema.dataset"
)


# DatasetBase is currently messy. In general it represents an object that is the
# intersection of CobaltDataset and CobaltDataSubset: you can access data from
# it, get embeddings, compute performance metrics, and view the data.
class DatasetBase(ABC):
    ## METADATA ACCESSORS
    @property
    @abstractmethod
    def metadata(self) -> DatasetMetadata:
        """A DatasetMetadata object containing the metadata for this dataset."""

    # TODO: move model metadata out of Dataset
    @property
    @abstractmethod
    def models(self) -> ModelMetadataCollection:
        """The models associated with this dataset.

        Each ModelMetadata object represents potential outcome, prediction, and
        error columns.
        """

    @property
    def model(self):
        if self.models:
            return self.models[0]
        return None

    @property
    @abstractmethod
    def array_names(self) -> List[str]: ...

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def get_categorical_columns(self) -> List[str]: ...

    def _has_model_metadata(self):
        """Check if dataset is fully set up.

        If dataset is not properly set up some functionality are not available.
        Example: FailureGroups and metrics will not be available.
        Dataset is fully prepared if:

        - prediction_col is setup
        - result_col is setup
        - model metadata is setup
        """
        # ToDo: add more caseflows where dataset is not fully set up.
        # ToDo: inform customer which data is missing.
        model = self.model
        if model is None:
            return False
        if model.prediction_column is None:
            return False
        if model.outcome_column is None:
            return False
        return True

    ## DATA ACCESSORS
    @property
    @abstractmethod
    def df(self) -> pd.DataFrame: ...

    @abstractmethod
    def get_array(self, key: str) -> np.ndarray: ...

    @abstractmethod
    def select_col(self, key: str) -> pd.Series: ...

    ## CONSTRUCTORS

    @abstractmethod
    def subset(self, indices: Union[np.ndarray, List[int]]) -> CobaltDataSubset: ...

    def mask(self, m: npt.ArrayLike) -> CobaltDataSubset:
        """Return a CobaltDataSubset consisting of rows at indices where ``m`` is nonzero."""
        m = np.asanyarray(m)
        if m.shape[0] != len(self):
            raise ValueError(f"m has {m.shape[0]} but it should be {len(self)} ")
        indices = np.flatnonzero(m)
        return self.subset(indices)

    def filter(self, condition: str) -> CobaltDataSubset:
        """Returns subset where condition evaluates to True in the DataFrame.

        Args:
            condition: String predicate that is evaluated using the `pd.eval` function.

        Returns:
            Selected Subset of type CobaltDataSubset

        Example:
            >>> df = pd.DataFrame({'a': [1, 2, 3, 4]})
            >>> ds = cobalt.CobaltDataset(df)
            >>> subset = ds.filter('a > 2')
            >>> len(subset)
            2

        """
        subset = self.df.eval(condition)
        return self.mask(subset)

    def sample(
        self, max_samples: int, random_state: Optional[int] = None
    ) -> CobaltDataSubset:
        """Return a CobaltDataSubset containing up to `max_samples` sampled rows.

        Up to `max_samples` rows will be sampled without replacement and returned
        as a CobaltDataSubset. If fewer rows exist than `max_samples`, all rows are
        returned.

        Args:
            max_samples: An integer indicating the maximum number of samples to pull.
            random_state: An optional integer to be used as a seed for random sampling.

        Returns:
            A CobaltDataSubset representing up to `max_samples` randomly sampled
            datapoints.
        """
        N = len(self)
        if max_samples < N:
            r = np.random.default_rng(seed=random_state)
            samples = r.choice(N, max_samples, replace=False)
            return self.subset(samples)
        else:
            return self.subset(np.arange(N))

    ## STATISTICS COMPUTATION
    def get_summary_statistics(
        self, categorical_max_unique_count: int = 10
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Returns summary statistics for each feature in the dataset."""
        numerical_df = self.df.select_dtypes(
            include=["number", "datetime64", "datetimetz"]
        )

        non_numerical_df = self.df.select_dtypes(
            exclude=["number", "datetime64", "datetimetz"]
        )

        N = len(self)

        low_cardinality_columns = [
            col
            for col in numerical_df.columns
            if numerical_df[col].nunique() < categorical_max_unique_count
        ]

        numerical_as_categorical_df = numerical_df[low_cardinality_columns].copy()

        for col in numerical_as_categorical_df:
            numerical_as_categorical_df[col] = numerical_as_categorical_df[col].astype(
                "category"
            )

        categorical_summary = pd.DataFrame()
        if numerical_as_categorical_df.shape[1] > 0:
            categorical_description = numerical_as_categorical_df.describe(
                include="all"
            )
            categorical_summary = categorical_description.T

        if non_numerical_df.shape[1] > 0:
            non_numerical_summarization = non_numerical_df.describe(include="all").T
            categorical_summary = pd.concat(
                [non_numerical_summarization, categorical_summary]
            )

        if "count" in categorical_summary.columns:
            categorical_summary["count missing"] = N - categorical_summary["count"]

        numerical_summary = pd.DataFrame()
        if numerical_df.shape[1] > 0:
            numerical_summary = numerical_df.describe(include="all").T
            numerical_summary["count missing"] = N - numerical_summary["count"]

        return (numerical_summary, categorical_summary)

    ## EMBEDDINGS
    # TODO: move embedding metadata out of Dataset
    @property
    @abstractmethod
    def embedding_metadata(self) -> List[Embedding]: ...

    @property
    def embedding_arrays(self) -> List[np.ndarray]:
        """A list of the raw arrays for each embedding.

        Deprecated. Get an embedding object from
        CobaltDataset.embedding_metadata and use Embedding.get() instead.
        """
        # DEPRECATE
        return [emb.get(self) for emb in self.embedding_metadata]

    def get_embedding(self, index: Union[int, str] = 0) -> np.ndarray:
        """Return the embedding associated with this CobaltDataset."""
        if isinstance(index, str):
            return self.get_embedding_array_by_name(index)
        return self.embedding_metadata[index].get(self)

    @abstractmethod
    def embedding_names(self) -> List[str]:
        """Return the available embedding names."""

    @abstractmethod
    def get_embedding_metadata_by_name(self, name: str) -> Embedding: ...

    def get_embedding_array_by_name(self, name: str) -> np.ndarray:
        return self.get_embedding_metadata_by_name(name).get(self)

    ## MODEL PERFORMANCE
    # TODO: move model performance out of dataset
    def overall_model_performance_scores(
        self, model_index: Union[int, str]
    ) -> Dict[str, float]:
        """Computes performance score for each available metrics."""
        model = self.models[model_index]
        return model.overall_performance_metrics(self)

    def overall_model_performance_score(
        self, metric: str, model_index: Union[int, str]
    ) -> float:
        """Computes the mean model performance score."""
        model = self.models[model_index]
        return model.overall_performance_metric(metric, self)

    def get_model_performance_data(
        self, metric: str, model_index: Union[int, str]
    ) -> np.ndarray:
        """Returns computed performance metric."""
        model = self.models[model_index]
        return model.calculate_performance_metric(metric, self)

    def get_image_columns(self) -> List[str]:
        """Gets image columns."""
        md = self.metadata
        return [mc.column_name for mc in md.media_columns if mc.is_image_type()]

    def _write_image_columns_to_table(
        self,
        media_columns: List[MediaInformationColumn],
        table_df: pd.DataFrame,
        run_server: Optional[bool] = False,
    ):
        image_columns = []
        for media_column in media_columns:
            new_column_name = media_column.autoname_media_visualization_column()[
                "column_name"
            ]
            image_columns.append(
                {"column_name": new_column_name, "is_remote": media_column.is_remote}
            )
            if not media_column.is_remote:
                image_host_url = media_column.get_path_to_media(run_server)
                table_df[new_column_name] = table_df[media_column.column_name].apply(
                    lambda x: f"{image_host_url}/{x}"  # noqa: B023
                )
            else:
                # images from remote server
                table_df[new_column_name] = table_df[media_column.column_name]

        if not run_server:
            # Fetching images concurrently and update dataframe
            self._encode_images_base64(table_df, image_columns)
        return image_columns

    def create_rich_media_table(
        self,
        break_newlines: bool = True,
        highlight_terms: Optional[Dict[str, List[str]]] = None,
        run_server: Optional[bool] = False,
    ) -> pd.DataFrame:
        """Returns media table with images columns as HTML column."""
        if highlight_terms is None:
            highlight_terms = {}

        md = self.metadata

        # leave the dataframe the user passed in intact.
        table_df = self.df.copy()
        media_columns = md.media_columns

        if media_columns:
            # Updates dataframe
            self._write_image_columns_to_table(
                media_columns, table_df, run_server=run_server
            )

        html_columns = md.long_text_columns
        for col in html_columns:
            # escape HTML content so it doesn't get rendered
            # we could add an option to turn this off, but should be careful about it
            table_df[col] = table_df[col].apply(
                lambda x: html.escape(x) if isinstance(x, str) else x
            )

            if break_newlines:
                table_df[col] = table_df[col].str.replace("\n", r"<br />")

            terms_to_highlight = highlight_terms.get(col)
            if terms_to_highlight:
                for term in terms_to_highlight:
                    # TODO: highlight color based on theme styles
                    table_df[col] = table_df[col].str.replace(
                        term, f"<span style='background-color: #23d5db'>{term}</span>"
                    )

        table_df = table_df.drop(md.hidable_columns, axis=1)
        return table_df

    @staticmethod
    def _repr_format_column_list(columns_to_render) -> str:
        max_columns_to_render = 10
        if len(columns_to_render) > max_columns_to_render:
            return str(columns_to_render[:max_columns_to_render])[:-1] + ", ...]"
        else:
            return str(columns_to_render)

    @abstractmethod
    def _brief_repr(self) -> str: ...

    def _repr_html_(self):
        """Defines representation in IPython output cell."""
        additional_info = self._brief_repr()
        print(additional_info)
        df_to_render = self.df

        if self.metadata.hidable_columns:
            df_to_render = df_to_render.drop(columns=self.metadata.hidable_columns)
        return df_to_render._repr_html_()

    @staticmethod
    def _fetch_base64_image(path) -> str:
        """Fetch the image from local folder and encode it to base64."""
        if os.path.isfile(path):
            mime_type, _ = mimetypes.guess_type(path)
            if mime_type and mime_type.startswith("image"):
                try:
                    with open(path, "rb") as image_file:
                        image_data = image_file.read()
                        base64_encoded_image = base64.b64encode(image_data).decode()
                    return f"data:{mime_type};base64,{base64_encoded_image}"

                except Exception as e:
                    debug_logger.info(f"Failed to serve Base64 image: {e}")
            else:
                debug_logger.info(f"Requested file {path} is not an image.")
        else:
            debug_logger.info(f"File {path} not found.")
        return ""

    def _encode_images_base64(self, df: pd.DataFrame, image_columns: List[Dict]):
        """Encode image data to base64 with maximum total size.

        Will only encode images up to a maximum output size of
        cobalt.settings.MAX_BASE64_TOTAL_SIZE bytes. Everything after that will
        be encoded as an empty string.
        """
        total_out_bytes = 0
        for col in image_columns:
            if col["is_remote"]:
                continue
            base64_enc_images = []
            for path in df[col["column_name"]]:
                if total_out_bytes < settings.MAX_BASE64_TOTAL_SIZE:
                    enc_img = self._fetch_base64_image(path) if pd.notna(path) else ""
                    total_out_bytes += len(enc_img)
                    base64_enc_images.append(enc_img)
                else:
                    # TODO: explain why image is not available
                    base64_enc_images.append("")

            df[col["column_name"]] = base64_enc_images


class CobaltDataset(DatasetBase):
    """Foundational object for a Cobalt analysis.

    Encapsulates all necessary information regarding the data, metadata,
    and model outputs associated with an analysis.

    Attributes:
        name: Optional string for dataset name
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        metadata: Optional[DatasetMetadata] = None,
        models: Optional[List[ModelMetadata]] = None,
        embeddings: Optional[List[Embedding]] = None,
        name: Optional[str] = None,
        arrays: Optional[Dict[str, np.ndarray]] = None,
    ):
        # keep this as a private implementation detail.
        # access should be through the df property

        # MARK START: Validation
        dataset = warn_if_index_not_unique(dataset)
        raise_error_for_non_string_column_names(dataset)

        # MARK END: Validation
        self._dataset = dataset

        self.emb_mngr = EmbeddingManager(embeddings if embeddings else [])
        self._arrays = arrays if arrays else {}
        self._metadata = metadata if metadata is not None else DatasetMetadata()
        self._models = (
            ModelMetadataCollection(models)
            if models is not None
            else ModelMetadataCollection([])
        )
        self.name = name or "Unnamed Dataset"
        self.img_host_url = None

        if not self._metadata.data_types:
            # this is usually the case.
            self._metadata.data_types = collect_columns_metadata(self.df)
        if metadata is None:
            # set timestamp columns based on autodetected dtypes
            self._metadata.timestamp_columns = [
                c
                for c, c_metadata in self._metadata.data_types.items()
                if c_metadata.col_type == ColumnDataType.datetime
            ]

    ## METADATA ACCESSORS
    @property
    def metadata(self) -> DatasetMetadata:
        return self._metadata

    @property
    def models(self) -> ModelMetadataCollection:
        return self._models

    def get_categorical_columns(self) -> List[str]:
        categorical_columns = []
        for c in self.df.columns:
            metadata = self.metadata.data_types.get(c)
            if metadata is None:
                metadata = infer_column_type(self.df[c])
                self.metadata.data_types[c] = metadata
            if metadata.is_categorical:
                categorical_columns.append(c)
        return categorical_columns

    def set_column_text_type(self, column: str, input_type: TextDataType):
        """Set the type for a text column in the dataset.

        Options include "long_text", which means the data in the column will be
        subject to keyword analysis but will not be available for coloring, and
        "short_text", which prevents keyword analysis but allows categorical
        coloring.
        """
        if not self.metadata.data_types[column].col_type == ColumnDataType.text:
            raise ValueError(f"Column {column} is not of text type")
        input_type = TextDataType(input_type)
        self.metadata.data_types[column].text_type = input_type
        if input_type != TextDataType.short_text:
            self.metadata.data_types[column].is_categorical = False
            self.metadata.data_types[column].cat_values = None
        else:
            col = self.select_col(column)
            self.metadata.data_types[column].is_categorical = True
            cat_values = col.unique().tolist()
            self.metadata.data_types[column].cat_values = cat_values

        # TODO: update any UI state

    def _set_columns_types(self, types: Dict[str, TextDataType]):
        for key, input_type in types.items():
            self.metadata.data_types[key].text_type = input_type

    @property
    def array_names(self) -> List[str]:
        """Names of the arrays stored in this dataset."""
        return list(self._arrays.keys())

    def add_model(
        self,
        input_columns: Optional[Union[str, List[str]]] = None,
        target_column: Optional[Union[str, List[str]]] = None,
        prediction_column: Optional[Union[str, List[str]]] = None,
        task: Union[str, ModelTask] = "custom",
        performance_columns: Optional[List[Union[str, dict]]] = None,
        name: Optional[str] = None,
    ):
        """Add a new model.

        Args:
            input_columns: The column(s) in the dataset that the model takes as input.
            target_column: The column(s) in the dataset with the target values for
                the model outputs.
            prediction_column: The column(s) in the dataset with the model's outputs.
            task: The task the model performs. This determines which performance
                metrics are calculated automatically. The default is "custom", which
                does not compute any performance metrics. Other options are
                "regression" and "classification".
            performance_columns: Columns of the dataset containing pointwise
                model performance metrics. This can be used to add extra custom
                performance metrics for the model.
            name: An optional name for the model. If one is not provided, a
                unique id will be generated.
        """
        dataset_cols = set(self.df.columns)
        if isinstance(task, str):
            task = ModelTask(task)

        if input_columns is None:
            input_columns = []
        elif isinstance(input_columns, str):
            input_columns = [input_columns]

        if target_column is None:
            outcome_columns = []
        elif isinstance(target_column, str):
            outcome_columns = [target_column]
        else:
            outcome_columns = target_column

        if prediction_column is None:
            prediction_columns = []
        elif isinstance(prediction_column, str):
            prediction_columns = [prediction_column]
        else:
            prediction_columns = prediction_column

        if performance_columns is None:
            performance_columns = []

        for col in input_columns:
            if col not in dataset_cols:
                raise ValueError(f"Input column {col} is not a column in this dataset")
        for col in prediction_columns:
            if col not in dataset_cols:
                raise ValueError(
                    f"Prediction column {col} is not a column in this Dataset"
                )
        for col in outcome_columns:
            if col not in dataset_cols:
                raise ValueError(f"Target column {col} is not a column in this dataset")

        error_columns = []
        performance_metrics = []
        for metric in performance_columns:
            if isinstance(metric, str):
                if metric not in dataset_cols:
                    raise ValueError(
                        f"Performance column {metric} is not a column in this dataset"
                    )
                error_columns.append(metric)
            elif isinstance(metric, dict):
                if metric["column"] not in dataset_cols:
                    raise ValueError(
                        f"Performance column {metric['column']} is not a column in this dataset"
                    )
                performance_metrics.append(metric)

        md = ModelMetadata(
            outcome_columns=outcome_columns,
            prediction_columns=prediction_columns,
            task=task,
            error_columns=error_columns,
            evaluation_metrics=performance_metrics,
            name=name,
            input_columns=input_columns,
        )

        self.models.append(md)

    def _get_model_by_name(self, name: str) -> ModelMetadata:
        try:
            return next(m for m in self.models if m.name == name)
        except StopIteration:
            raise ValueError(f"No model with name {name} found.") from None

    ## DATA ACCESSORS
    @property
    def df(self) -> pd.DataFrame:
        """Returns a pd.DataFrame of the underlying data for this dataset."""
        return self._dataset

    def select_col(self, col: str) -> pd.Series:
        """Return the values for column `col` of this dataset."""
        return self._dataset[col]

    def set_column(
        self, key: str, data, is_categorical: Union[bool, Literal["auto"]] = "auto"
    ):
        """Add or replace a column in the dataset.

        Args:
            key: Name of the column to add.
            data: ArrayLike of values to store in the column. Must have length
                equal to the length of the dataset.
            is_categorical: Whether the column values should be treated as
                categorical. If "auto" (the default), will autodetect.
        """
        series = pd.Series(data, name=key)
        if not is_categorical and series.dtype == "categorical":
            raise ValueError("Categorical Series provided but is_categorical=False.")

        if is_categorical and (is_categorical != "auto"):
            series = series.astype("category")

        col_metadata = infer_column_type(series)
        if not is_categorical:
            col_metadata.is_categorical = False

        self._dataset[key] = series
        self.metadata.data_types[key] = col_metadata

    def add_media_column(
        self,
        paths: List[str],
        local_root_path: Optional[str] = None,
        column_name: Optional[str] = None,
    ):
        """Add a media column to the dataset.

        Args:
            paths: A list or other array-like object containing the paths to the
                media file for each data point in the dataset.
            local_root_path: A root path for all the paths in `paths`
            column_name: The name for the column in the dataset that should
                store the media file paths.
        """
        if len(paths) != len(self):
            raise Exception("image paths has incompatible length.")

        if type(self) == CobaltDataSubset:
            raise TypeError("Cannot add a media column to a CobaltDataSubset object.")

        first_file = paths[0]
        ext = first_file.split(os.extsep)[-1]
        column_name = (
            ("_image_path" if ext.lower() in ("jpg", "png", "jpeg") else "_media_path")
            if column_name is None
            else column_name
        )

        self.set_column(column_name, paths, is_categorical=False)

        is_remote = bool(
            all(
                path.startswith(
                    (
                        "http://",
                        "https://",
                    )
                )
                for path in paths
            )
        )
        if not is_remote and local_root_path is None:
            raise ValueError(f"{local_root_path} can not be empty for local media")
        media_column = schema.MediaInformationColumn(
            column_name, ext, local_root_path, is_remote=is_remote
        )
        self.metadata.media_columns.append(media_column)
        self.metadata.data_types[column_name].col_type = ColumnDataType.text
        self.metadata.data_types[column_name].text_type = TextDataType.image_path
        self.metadata.data_types[column_name].is_categorical = False

    def get_array(self, key: str) -> np.ndarray:
        """Get an array from the dataset."""
        # TODO: namespaces?
        return self._arrays[key]

    def add_array(self, key: str, array: Union[np.ndarray, scipy.sparse.csr_array]):
        """Add a new array to the dataset.

        Will raise an error if an array with the given name already exists.
        """
        if array.shape[0] != len(self):
            raise ValueError(
                "Wrong size for array. "
                f"Expected {len(self)} rows, got {array.shape[0]}."
            )
        # TODO: namespaces?
        if key in self._arrays:
            raise ValueError(f"Array with name {key!r} already exists.")
        self._arrays[key] = array

    ## EMBEDDINGS

    def add_embedding_array(
        self,
        embedding: Union[np.ndarray, Any],
        metric: str = "euclidean",
        name: Optional[str] = None,
    ):
        """Add an embedding to the dataset.

        Args:
            embedding: An array or arraylike object containing the embedding
                values. Should be two-dimensional and have the same number of rows
                as the dataset.
            metric: The preferred distance metric to use with this embedding.
                Defaults to "euclidean"; "cosine" is another useful option.
            name: An optional name for the embedding.
        """
        if not (isinstance(embedding, np.ndarray) or scipy.sparse.issparse(embedding)):
            WarningsHandler.convert_numpy(embedding_type=type(embedding))
            embedding = np.array(embedding)
        if scipy.sparse.issparse(embedding):
            embedding = scipy.sparse.csr_array(embedding)

        if not pd.api.types.is_numeric_dtype(embedding.dtype):
            raise ValueError(
                f"Unsupported dtype for embedding = {embedding.dtype}. "
                "Embeddings should be of numeric dtype."
            )

        if name is None:
            name = self.emb_mngr.get_new_embedding_name()

        self.add_array(name, embedding)
        embedding = ArrayEmbedding(
            array_name=name, dimension=embedding.shape[1], metric=metric, name=name
        )
        self.add_embedding(embedding)

    def add_embedding(self, embedding: Embedding):
        """Add an Embedding object."""
        self.emb_mngr.add_embedding(embedding)

    @property
    def embedding_metadata(self) -> List[Embedding]:
        """The Embedding objects associated with this dataset."""
        return self.emb_mngr.embedding_metadata

    def embedding_names(self) -> List[str]:
        """Return the available embedding names."""
        return self.emb_mngr.embedding_names()

    def get_embedding_metadata_by_name(self, name: str) -> Embedding:
        return self.emb_mngr.map_name_to_embedding(name)

    ## PERFORMANCE METRICS
    def _column_name_for_model_and_metric(
        self, model: ModelMetadata, metric_key: str
    ) -> str:
        """Choose a name for the column containing the specified metric for this model."""
        # TODO: should this check if the column name already exists?
        return f"{model.name}_{metric_key}"

    def compute_model_performance_metrics(self):
        """Compute the performance metrics for each model in `dataset`.

        Adds columns to the dataset storing the computed metrics, and updates
        the `ModelMetadata.error_column` attributes corerspondingly.
        """
        # This function mainly exists to make performance metrics available for coloring.
        # There are probably better ways to do this.
        # TODO: Refactor performance metric coloring so this function is unnecessary.

        for model in self.models:
            performance_metrics = model.performance_metrics()
            error_cols = []

            for metric in performance_metrics.values():
                if isinstance(metric, ColumnEvaluationMetric):
                    # in this case, the column is already in the dataset with a user-provided name
                    # so there's no need to compute or add it again
                    # we just add the column to the list of error columns for the model
                    error_cols.append(metric.column)
                else:
                    stats = metric.calculate(self)
                    for metric_key in stats:
                        new_column_name = self._column_name_for_model_and_metric(
                            model, metric_key
                        )
                        self.set_column(new_column_name, stats[metric_key])
                        error_cols.append(new_column_name)

                # this ensures that these columns appear in the correct section of the coloring menu
                model.error_columns = error_cols

    def time_range(
        self, start_time: pd.Timestamp, end_time: pd.Timestamp
    ) -> CobaltDataSubset:
        """Return a CobaltDataSubset within a time range.

        Args:
            start_time: A pd.Timestamp marking the start of the time window.
            end_time: A pd.Timestamp marking the end of the time window.

        Returns:
            A CobaltDataSubset consisting of datapoints within the range
            [`start_time`, `end_time`).
        """
        if not self.metadata.has_timestamp_column():
            raise ValueError(
                "No timestamp column set. "
                "Please add a column name to self.metadata.timestamp_columns."
            )
        return self.mask(
            (self.df[self.metadata.timestamp_column()] >= start_time)
            & (self.df[self.metadata.timestamp_column()] < end_time)
        )

    def subset(self, indices: npt.ArrayLike) -> CobaltDataSubset:
        """Returns a CobalDataSubset consisting of rows indexed by `indices`."""
        indices = np.asanyarray(indices)
        if not indices_type_valid(indices):
            raise TypeError("Indices should be one dimensional integers.")
        return CobaltDataSubset(
            self,
            indices,
        )

    def as_subset(self):
        """Returns all rows of this CobaltDataset as a CobaltDataSubset."""
        return self.subset(np.arange(len(self)))

    def sample(
        self, max_samples: int, random_state: Optional[int] = None
    ) -> CobaltDataSubset:
        """Return a CobaltDataSubset containing up to `max_samples` sampled rows.

        Up to `max_samples` rows will be sampled without replacement and returned
        as a CobaltDataSubset. If fewer rows exist than `max_samples`, all rows are
        returned.

        Args:
            max_samples: The maximum number of samples to pull.
            random_state: An optional integer to be used as a seed for random sampling.

        Returns:
            A CobaltDataSubset representing up to `max_samples` randomly sampled
            datapoints.
        """
        N = len(self)
        if max_samples < N:
            r = np.random.default_rng(seed=random_state)
            samples = r.choice(N, max_samples, replace=False)
            return self.subset(samples)
        else:
            return self.as_subset()

    def __len__(self):
        return self._dataset.shape[0]

    ## DISPLAY
    def __repr__(self):
        return (
            f"CobaltDataset(name='{self.name}',\n"
            f"columns={self._repr_format_column_list(list(self._dataset.columns))},\n"
            f"media_columns={self._repr_format_column_list(list(self.metadata.media_columns))},\n"
            f"timestamp_columns={self._repr_format_column_list(self.metadata.timestamp_columns)},\n"
            f"hidable_columns={self._repr_format_column_list(self.metadata.hidable_columns)},\n"
            f"other_metadata_columns={self._repr_format_column_list(self.metadata.other_metadata_columns)}\n)"
        )

    def _brief_repr(self):
        return f"CobaltDataset(name='{self.name}', n_points={len(self)})"

    def to_dict(self) -> dict:
        """Save all information in this dataset to a dict."""
        index = self.df.index.tolist()
        buffer = BytesIO()
        self.df.to_parquet(buffer)
        buffer.seek(0)
        dataset_df = base64.b64encode(buffer.read()).decode()

        metadata_dict = self.metadata.to_dict()
        models = [model.to_dict() for model in self.models]
        embeddings = [embedding.to_dict() for embedding in self.embedding_metadata]

        arrays = {}
        for k, v in self._arrays.items():
            if not isinstance(v, np.ndarray):
                raise ValueError(f"{k} is not a numpy array and can not be serialized.")
            arrays[k] = base64.b64encode(np_to_bytes(v)).decode()

        return {
            "dataset": dataset_df,
            "metadata": metadata_dict,
            "models": models,
            "embeddings": embeddings,
            "name": self.name,
            "arrays": arrays,
            "index": index,
        }

    def to_json(self) -> str:
        """Serialize this dataset to a JSON string."""
        data_dict = self.to_dict()

        def convert_timestamp(obj):
            if isinstance(obj, pd.Timestamp):
                return obj.isoformat()

        return json.dumps(data_dict, default=convert_timestamp)

    @classmethod
    def from_json(cls, serialized_data: str) -> CobaltDataset:
        """Deserialize a JSON string into a dataset."""
        data = json.loads(serialized_data)
        instance = cls.from_dict(data)
        return instance

    @staticmethod
    def _df_from_parquet(data: dict) -> pd.DataFrame:
        decoded_data = base64.b64decode(data["dataset"])
        buffer = BytesIO(decoded_data)
        new_df = pd.read_parquet(buffer)
        return new_df

    @staticmethod
    def _set_categorical_cols_from_serialized_metadata(metadata, df):
        data_types = metadata.data_types
        cat_list = [
            data_types[colname].name
            for colname in data_types
            if data_types[colname].explicit_categorical
        ]
        for col in cat_list:
            df[col] = pd.Categorical(df[col])

    @classmethod
    def from_dict(cls, data) -> CobaltDataset:
        """Instantiate a CobaltDataset from a dictionary representation."""
        metadata = DatasetMetadata.from_dict(data["metadata"])
        data["metadata"] = metadata
        new_df = cls._df_from_parquet(data)

        old_index = data.pop("index")
        new_df.index = pd.Index(old_index)

        cls._set_categorical_cols_from_serialized_metadata(metadata, new_df)
        data["dataset"] = new_df

        embeddings = []
        for serialized_emb in data["embeddings"]:
            if "columns" in serialized_emb:
                embeddings.append(ColumnEmbedding.from_dict(serialized_emb))
            else:
                embeddings.append(ArrayEmbedding.from_dict(serialized_emb))
        data["embeddings"] = embeddings
        models = [ModelMetadata.from_dict(model) for model in data["models"]]
        data["models"] = models

        arrays_serialized = data["arrays"]
        arrays = {
            k: np_from_bytes(base64.b64decode(v)) for k, v in arrays_serialized.items()
        }
        data["arrays"] = arrays

        instance = cls(**data)
        return instance

    def save(self, file_path: Union[str, os.PathLike]) -> str:
        """Write this dataset to a .json file.

        Returns the path written to.
        """
        if isinstance(file_path, Path):
            file_path = str(file_path)

        if not file_path.endswith(".json"):
            raise ValueError("The file path must end with .json")

        json_data = self.to_json()

        # Only create directories if a directory path is provided
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(file_path, "w") as f:
            json.dump(json_data, f)
        return file_path

    @classmethod
    def load(cls, file_path: str) -> CobaltDataset:
        """Load a saved dataset from a .json file."""
        with open(file_path) as json_file:
            data_loaded = json.load(json_file)
        instance = cls.from_json(data_loaded)
        return instance

    def __eq__(self, other: Union[CobaltDataset, object]):
        if not isinstance(other, CobaltDataset):
            return NotImplemented

        return all(
            [
                self.df.equals(other.df),
                array_dicts_equal(self._arrays, other._arrays),
                self.emb_mngr.embedding_metadata == other.emb_mngr.embedding_metadata,
                self.metadata == other.metadata,
                self.models == other.models,
            ]
        )

    def __hash__(self):
        return hash((len(self.df), tuple(self.df.columns)))


class CobaltDataSubset(DatasetBase):
    """Represents a subset of a CobaltDataset.

    Should in general be constructed by calling the subset() method (or other
    similar methods) on a `CobaltDataset` or `CobaltDataSubset`.

    In principle, this could have repeated data points, since there is no check
    for duplicates.

    Attributes:
        source_dataset: The CobaltDataset of which this is a subset.
        indices: np.ndarray of integer row indices defining the subset.
    """

    def __init__(
        self,
        source: CobaltDataset,
        indices: Union[np.ndarray, List[int]],
    ) -> None:
        if not isinstance(indices, np.ndarray):
            # TODO: should indices always be sorted?
            # need to be careful about this. what are reasonable expectations?
            # do we ever create a subset and expect it to be in the given order?
            indices = np.array(indices, dtype=np.int32)
        self.indices = indices
        self.source_dataset = source

    ## METADATA ACCESSORS
    @property
    def metadata(self) -> DatasetMetadata:
        return self.source_dataset.metadata

    @property
    def models(self) -> ModelMetadataCollection:
        return self.source_dataset.models

    @property
    def array_names(self) -> List[str]:
        return self.source_dataset.array_names

    def __len__(self):
        return len(self.indices)

    def get_categorical_columns(self) -> List[str]:
        return self.source_dataset.get_categorical_columns()

    ## DATA ACCESSORS
    @property
    def df(self) -> pd.DataFrame:
        """Returns a pd.DataFrame of the data represented by this data subset."""
        return self.source_dataset._dataset.iloc[self.indices]

    def select_col(self, col: str) -> pd.Series:
        """Return the pd.Series for column `col` of this data subset."""
        return self.source_dataset._dataset[col].iloc[self.indices]

    def get_array(self, key: str) -> np.ndarray:
        # TODO: namespaces?
        return self.source_dataset.get_array(key)[self.indices, :]

    ## CONSTRUCTORS
    def subset(self, indices: npt.ArrayLike) -> CobaltDataSubset:
        """Returns a subset obtained via indexing into self.df.

        Tracks the dependency on self.source_dataset.
        """
        indices = np.array(indices)
        if not indices_type_valid(indices):
            raise TypeError("Indices should be one dimensional integers.")
        return CobaltDataSubset(
            self.source_dataset,
            indices=self.indices[indices],
        )

    def concatenate(self, dataset: CobaltDataSubset) -> CobaltDataSubset:
        """Add another data subset to this one. Does not check for overlaps.

        Returns:
            A new CobaltDataSubset object containing points from `self` and the passed dataset.

        Raises:
            ValueError: if `self` and `dataset` have different parent datasets.
        """
        if dataset.source_dataset is not self.source_dataset:
            raise ValueError("Must be subsets of the same dataset")
        indices = np.concatenate([self.indices, dataset.indices])
        return self.source_dataset.subset(indices)

    def difference(self, dataset: CobaltDataSubset) -> CobaltDataSubset:
        """Returns the subset of `self` that is not contained in `dataset`.

        Raises:
            ValueError: if `self` and `dataset` have different parent datasets.
        """
        if self.source_dataset is not dataset.source_dataset:
            raise ValueError("Must be subsets of the same dataset")
        indices = np.setdiff1d(self.indices, dataset.indices)
        return CobaltDataSubset(
            self.source_dataset,
            indices,
        )

    def intersect(self, dataset: CobaltDataSubset) -> CobaltDataSubset:
        """Returns the intersection of `self` with `dataset`.

        Raises:
            ValueError: if `self` and `dataset` have different parent datasets.
        """
        if self.source_dataset is not dataset.source_dataset:
            raise ValueError("Must be subsets of the same dataset")
        indices = np.intersect1d(self.indices, dataset.indices)
        return CobaltDataSubset(
            self.source_dataset,
            indices,
        )

    def to_dataset(self) -> CobaltDataset:
        """Converts this subset to a standalone CobaltDataset.

        Returns:
            dataset (CobaltDataset): returns this object as a dataset.
        """
        embedding_manager = self.source_dataset.emb_mngr
        embedding_list = embedding_manager._embeddings

        arrays = self.source_dataset._arrays
        subsetted_arrays = {key: self.get_array(key) for key in arrays}

        return CobaltDataset(
            self.df,
            copy.deepcopy(self.metadata),
            copy.deepcopy(self.models),
            copy.deepcopy(embedding_list),
            name=f"{self.source_dataset.name} subset",
            arrays=copy.deepcopy(subsetted_arrays),
        )

    def intersection_size(self, dataset: CobaltDataSubset) -> int:
        """Returns the size of the intersection of `self` with `dataset`.

        Somewhat more efficient than `len(self.intersect(dataset))`.

        Raises:
            ValueError: if `self` and `dataset` have different parent datasets.
        """
        if self.source_dataset is not dataset.source_dataset:
            raise ValueError("Must be subsets of the same dataset")
        indices = np.intersect1d(self.indices, dataset.indices)
        return len(indices)

    def complement(self) -> CobaltDataSubset:
        """Returns the complement of this set in its source dataset."""
        return self.source_dataset.as_subset().difference(self)

    ## DESCRIPTORS
    def as_mask_on(self, base_subset: CobaltDataSubset) -> np.ndarray[np.bool_]:
        """Returns mask of self on another subset.

        Raises:
            ValueError: if self is not a subset of base_subset.
        """
        if not self.is_subset(base_subset):
            raise ValueError("base_subset is not a superset of this dataset.")

        full_mask = self.as_mask()
        mask = full_mask[base_subset.indices]
        return mask

    def as_mask(self) -> np.ndarray[np.bool_]:
        """Returns mask of self on self.source_dataset."""
        full_mask = np.zeros(len(self.source_dataset), dtype=np.bool_)
        full_mask[self.indices] = 1
        return full_mask

    def is_subset(self, other: CobaltDataSubset) -> bool:
        if other.source_dataset is not self.source_dataset:
            raise Exception("Datasets do not have the same source dataset.")
        indices = set(self.indices)
        other_indices = other.indices
        inter = indices.intersection(set(other_indices))
        return len(inter) == len(indices)

    ## EMBEDDINGS
    def embedding_names(self) -> List[str]:
        """Return the available embedding names."""
        return self.source_dataset.emb_mngr.embedding_names()

    @property
    def embedding_metadata(self) -> List[Embedding]:
        return self.source_dataset.embedding_metadata

    def get_embedding_metadata_by_name(self, name: str) -> Embedding:
        return self.source_dataset.get_embedding_metadata_by_name(name)

    def get_model_performance_metrics(self):
        """Retrieve and aggregate performance metrics for each model in the subset.

        This method iterates over each model and retrieves its overall performance scores.

        Returns:
            dict: A dictionary structured as {model_name: {metric_name: metric_value}},
                  where metric_value is the computed score for each metric.
        """
        performance_metric_results = {}

        for i, model in enumerate(self.models):
            model_name = model.name
            performance_metrics = self.overall_model_performance_scores(i)
            performance_metric_results[model_name] = performance_metrics

        return performance_metric_results

    def get_classifier(
        self,
        model_type: Union[
            Literal["svm", "knn", "rf"],
            Callable[[CobaltDataSubset, CobaltDataSubset, int], Classifier],
        ] = "knn",
        embedding_index: int = 0,
        global_set: Optional[CobaltDataSubset] = None,
        params: Optional[Dict] = None,
    ):
        """Build a Classifier to distinguish this subset from the rest of the data.

        The classifier takes data point embeddings as an input; the specific
        embedding to be used can be selected by the user.

        This is an experimental method and interfaces may change.

        Args:
            model_type: a string representing the type of model to be trained
            embedding_index: which embedding from self.embeddings to use as inputs
            global_set: the ambient dataset that the classifier should
                distinguish this subset from. If not provided, the classifier will
                distinguish self from self.source_dataset
            params: a dict of keyword arguments to be passed to the classifier constructor
        """
        if global_set is None:
            global_set = self.source_dataset.as_subset()

        if params is None:
            params = {}
        if model_type == "svm":
            from sklearn.svm import SVC

            classifier = SVC(random_state=4723, **params)
            return ScikitClassifier(self, global_set, embedding_index, classifier)
        elif model_type == "rf":
            from sklearn.ensemble import RandomForestClassifier

            classifier = RandomForestClassifier(random_state=876, **params)
            return ScikitClassifier(self, global_set, embedding_index, classifier)

        elif model_type == "knn":
            return KNNClassifier(
                self, global_set, embedding_index, random_state=9043, **params
            )

        elif callable(model_type):
            return model_type(self, global_set, embedding_index, **params)
        else:
            raise ValueError(f"Invalid classifier type {model_type}.")

    def get_graph_inputs(
        self, embedding: Union[int, str, schema.Embedding]
    ) -> Tuple[np.ndarray, schema.Embedding]:
        if isinstance(embedding, str):
            X = self.get_embedding_array_by_name(embedding)
            embedding_metadata = self.get_embedding_metadata_by_name(embedding)
        elif isinstance(embedding, int):
            X = self.get_embedding(embedding)
            embedding_metadata = self.embedding_metadata[embedding]
        elif isinstance(embedding, schema.Embedding):
            X = embedding.get(self)
            embedding_metadata = embedding
        else:
            raise TypeError(
                "Expected argument 'embedding' as type int, str, or Embedding, "
                f"but got {type(embedding)}"
            )
        return X, embedding_metadata

    def __eq__(self, other):
        if not isinstance(other, CobaltDataSubset):
            return False
        return (
            self.source_dataset is other.source_dataset
            and len(self) == len(other)
            and np.all(self.indices == other.indices)
        )

    def __hash__(self):
        return hash((len(self.df), tuple(self.df.columns), tuple(self.indices)))

    def __repr__(self):
        return (
            f"CobaltDataSubset(source_dataset='{self.source_dataset.name}',\n"
            f"columns={self._repr_format_column_list(list(self.source_dataset._dataset.columns))},\n"
            f"media_columns={self._repr_format_column_list(list(self.metadata.media_columns))},\n"
            f"timestamp_columns={self._repr_format_column_list(self.metadata.timestamp_columns)},\n"
            f"hidable_columns={self._repr_format_column_list(self.metadata.hidable_columns)},\n"
            f"other_metadata_columns={self._repr_format_column_list(self.metadata.other_metadata_columns)}\n)"
        )

    def _brief_repr(self):
        return (
            f"CobaltDataSubset(source_dataset='{self.source_dataset.name}', "
            f"n_points={len(self)})"
        )

    def to_json(self) -> str:
        raise NotImplementedError(
            "Subsets cannot be serialized. Convert to a dataset first."
        )

    @classmethod
    def from_json(cls, serialized_data) -> CobaltDataSubset:
        raise NotImplementedError


class Classifier(ABC):
    """A trainable classifier based on a CobaltDataSubset.

    Experimental; interface may change.
    """

    def __init__(
        self,
        subset: CobaltDataSubset,
        global_set: CobaltDataSubset,
        embedding_index: int,
    ):
        self.subset = subset
        self.global_set = global_set
        self.embedding_index = embedding_index

    def get_training_data(self):
        # TODO: train/val(/test?) split
        true_set = self.subset.get_embedding(self.embedding_index)
        false_set = self.global_set.difference(self.subset).get_embedding(
            self.embedding_index
        )
        y_true = np.ones(true_set.shape[0], dtype=np.bool_)
        y_false = np.zeros(false_set.shape[0], dtype=np.bool_)
        X = np.concatenate([true_set, false_set], axis=0)
        y = np.concatenate([y_true, y_false], axis=0)
        return X, y

    @abstractmethod
    def fit(self):
        """Fit the classifier."""

    @abstractmethod
    def apply(self, X):
        """Apply this classifier to a set of inputs."""

    def evaluate(self, X, y):
        """Compute precision and recall of this classifier."""
        y_pred = self.apply(X)
        # TODO: remove sklearn dependency
        from sklearn.metrics import precision_score, recall_score

        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        return precision, recall


class ScikitClassifier(Classifier):
    def __init__(
        self,
        subset: CobaltDataSubset,
        global_set: CobaltDataSubset,
        embedding_index: int,
        model,
    ):
        super().__init__(subset, global_set, embedding_index)
        self.model = model

    def fit(self, verbose=False):
        X, y = self.get_training_data()
        self.model.fit(X, y)

        if verbose:
            precision, recall = self.evaluate(X, y)
            print(f"Classifier trained. Precision={precision}, Recall={recall}")

    def apply(self, X):
        out = self.model.predict(X)
        return out


class KNNClassifier(Classifier):
    def __init__(
        self,
        subset: CobaltDataSubset,
        global_set: CobaltDataSubset,
        embedding_index: int,
        k: int = 10,
        **kwargs,
    ):
        super().__init__(subset, global_set, embedding_index)
        self.k = k
        self.metric = subset.embedding_metadata[embedding_index].default_distance_metric
        self.is_fitted = False
        self.kwargs = kwargs

    def fit(self, verbose=False):
        X, y = self.get_training_data()
        import pynndescent

        self.nn_index = pynndescent.NNDescent(X, metric=self.metric, **self.kwargs)
        self.y = y

        self.is_fitted = True
        if verbose:
            precision, recall = self.evaluate(X, y)
            print(f"Classifier trained. Precision={precision}, Recall={recall}")

    def apply(self, X):
        nbrs, _ = self.nn_index.query(X, k=self.k)
        nbr_vals = self.y[nbrs]
        nbr_fraction = np.mean(nbr_vals, axis=1)
        return nbr_fraction > 0.5


def fast_intersect_subset_with_indices(
    subset: CobaltDataSubset,
    indices: List[np.ndarray],
):
    """Determines which sets defined in ``indices`` intersect with ``subset``.

    Equivalent to
    ``[i for i, u in enumerate(indices)
        if len(subset.intersect(subset.source_dataset.subset(u))) > 0]``
    but much faster. Useful for finding which of a collection of groups
    (e.g. graph nodes) intersect with subset.
    """
    # Using a mask here is a performance optimization, since
    # subset.intersect() and the underlying np.intersect1d() do not scale
    # well with group size.
    # For small subsets and sets of indices, using a set is faster, but the mask
    # scales better. Since the performance for small subsets is acceptable, we
    # use the mask.
    subset_mask = subset.as_mask()
    selected = [i for i, u in enumerate(indices) if np.any(subset_mask[u])]
    return selected
