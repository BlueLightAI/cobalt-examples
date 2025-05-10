# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

from __future__ import annotations

import json
import os
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from cobalt.cobalt_types import ColumnDataType
from cobalt.repositories.server_repository import SERVER_REGISTRY
from cobalt.warnings_handler import WarningsHandler


class Column:
    """Represents a column in a pandas DataFrame.

    Attributes:
        column_name: String name of a pd.DataFrame column.
    """

    def __init__(self, column_name) -> None:
        self.column_name = column_name

    def __str__(self) -> str:
        return self.column_name

    def __repr__(self) -> str:
        return f"'{self.column_name}'"


class MediaInformationColumn(Column):
    """Represent a column containing information about media files.

    Attributes:
        column_name (str): Column Name in dataframe.
        file_type (str): A string indicating the file type, e.g. its extension.
        host_directory (str): Path or URL where the file is located.
        is_remote: Whether the file is remote.
    """

    def __init__(
        self, column_name: str, file_type: str, host_directory: str, is_remote=False
    ) -> None:
        super().__init__(column_name=column_name)
        self.file_type = file_type
        self.host_directory = host_directory
        self.is_remote = is_remote

        if self.file_type == "img":
            WarningsHandler.img_type_deprecated()

    def is_image_type(self):
        return self.file_type.lower() in ("jpg", "jpeg", "png", "img")

    def autoname_media_visualization_column(self) -> dict:
        """Autoname media column."""
        if self.is_image_type():
            return {
                "is_remote": self.is_remote,
                "column_name": f"{self.column_name}_img",
            }

        raise Exception("Other Media Types not supported in this version.")

    def __eq__(self, other) -> bool:
        if not isinstance(other, MediaInformationColumn):
            return NotImplemented

        return all(
            [
                self.column_name == other.column_name,
                self.file_type == other.file_type,
                self.host_directory == other.host_directory,
                self.is_remote == other.is_remote,
            ]
        )

    def to_dict(self) -> dict:
        return {
            "column_name": self.column_name,
            "file_type": self.file_type,
            "host_directory": self.host_directory,
            "is_remote": self.is_remote,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, serialized_data) -> MediaInformationColumn:
        data = json.loads(serialized_data)
        instance = cls(**data)
        return instance

    def get_path_to_media(self, run_server: bool):
        if run_server:
            servers = SERVER_REGISTRY.servers
            img_host_url = servers[self.host_directory].img_host_url
        else:
            img_host_url = os.path.abspath(os.path.join(self.host_directory))
        return img_host_url


class TextDataType(Enum):
    long_text = "long_text"
    short_text = "short_text"
    image_path = "image"


IMAGE_FILE_SUFFICES = (".png", ".jpg", ".jpeg")


# TODO: this could just be a Dataclass
class DatasetColumnMetadata:
    """Contains metadata about a column in the dataset.

    Attributes:
        name: The name of the column in the data table.
        col_type: The underlying storage type of data.
        is_categorical: Whether the column values can be treated as categorical.
        explicit_categorical: If is_categorical=True, whether this was set
            explicitly or autodetected.
        cat_values: The distinct categorical values taken, if is_categorical=True.
        text_type: A finer division of the type of text, set if col_type=text.
    """

    def __init__(
        self,
        name: str,
        explicit_categorical: bool,
        col_type: ColumnDataType,
        is_categorical: bool = False,
        cat_values: Optional[List] = None,
        text_type: Optional[TextDataType] = None,
    ) -> None:
        self.name = name
        self.col_type = col_type
        self.cat_values = cat_values if cat_values is not None else []
        self.is_categorical = is_categorical
        self.explicit_categorical = explicit_categorical
        self.text_type = text_type

    def to_dict(self):
        return {
            "name": self.name,
            "col_type": self.col_type.value,
            "cat_values": self.cat_values,
            "text_type": self.text_type.value if self.text_type else None,
            "is_categorical": self.is_categorical,
            "explicit_categorical": self.explicit_categorical,
        }

    def to_json(self):
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, data):
        data = json.loads(data)
        data["col_type"] = ColumnDataType(data["col_type"])
        data["text_type"] = (
            TextDataType(data["text_type"]) if data["text_type"] else None
        )
        return cls(**data)

    def __eq__(self, other):
        if not isinstance(other, DatasetColumnMetadata):
            return NotImplemented

        return all(
            [
                self.name == other.name,
                self.col_type == other.col_type,
                list_with_nans_equal(self.cat_values, other.cat_values),
                self.is_categorical == other.is_categorical,
                self.explicit_categorical == other.explicit_categorical,
                self.text_type == other.text_type,
            ]
        )

    def __repr__(self):
        return (
            f"DatasetColumnMetadata(name={self.name!r}, "
            f"explicit_categorical={self.explicit_categorical}, "
            f"col_type={self.col_type!r}, is_categorical={self.is_categorical!r}, "
            f"cat_values={self.cat_values!r}, text_type={self.text_type!r})"
        )


def list_with_nans_equal(list1, list2):
    if len(list1) != len(list2):
        return False
    for x, y in zip(list1, list2):
        if x == y or (
            isinstance(x, (float, np.floating))
            and isinstance(y, (float, np.floating))
            and np.isnan(x)
            and np.isnan(y)
        ):
            continue
        return False

    return True


def get_column_text_type(col: pd.Series) -> Optional[TextDataType]:
    val = col.iloc[0]
    if not isinstance(val, str):
        return None
    if val.lower().endswith(IMAGE_FILE_SUFFICES):
        return TextDataType.image_path
    if col.str.count(r"\s+").mean() >= 5:
        return TextDataType.long_text
    return TextDataType.short_text


def is_column_categorical(col: pd.Series, categorical_max_unique_count: int = 10):
    if isinstance(col.dtype, pd.api.types.CategoricalDtype):
        return True
    # This feels a bit overeager
    try:
        if pd.api.types.is_object_dtype(col) and not col.is_unique:
            return True
        if col.nunique() < categorical_max_unique_count:
            return True
    except TypeError:
        return False
    return False


def infer_column_type(
    col: pd.Series, categorical_max_unique_count: int = 10
) -> DatasetColumnMetadata:
    explicit_categorical = isinstance(col.dtype, pd.CategoricalDtype)
    column_value_dtype = col.cat.categories.dtype if explicit_categorical else col.dtype

    # assign column_type and text_type
    text_type = None
    if pd.api.types.is_numeric_dtype(column_value_dtype):
        column_type = ColumnDataType.numerical
    elif pd.api.types.is_datetime64_any_dtype(column_value_dtype):
        column_type = ColumnDataType.datetime
    elif pd.api.types.is_object_dtype(column_value_dtype):
        column_type = ColumnDataType.text
        text_type = get_column_text_type(col)
    else:
        column_type = ColumnDataType.other

    # assign is_categorical and cat_values
    if explicit_categorical:
        cat_values = col.cat.categories.values.tolist()
        is_categorical = True
    elif text_type == TextDataType.long_text:
        is_categorical = False
        cat_values = None
    else:
        try:
            is_categorical = is_column_categorical(col, categorical_max_unique_count)
            cat_values = col.unique().tolist() if is_categorical else None
        except TypeError:
            # non hashable column
            is_categorical = False
            cat_values = None

    return DatasetColumnMetadata(
        name=col.name,
        col_type=column_type,
        cat_values=cat_values,
        is_categorical=is_categorical,
        explicit_categorical=explicit_categorical,
        text_type=text_type,
    )


def collect_columns_metadata(
    df: pd.DataFrame, categorical_max_unique_count: int = 10
) -> Dict[str, DatasetColumnMetadata]:
    result = {
        col: infer_column_type(df[col], categorical_max_unique_count)
        for col in df.columns
    }
    return result


class DatasetMetadata:
    """Encapsulates various metadata about a CobaltDataset.

    Attributes:
        media_columns: Optional list of MediaInformationColumns.
        timestamp_columns: Optional list of timestamp column name strings.
        hidable_columns: Optional list of hidable column name strings.
        default_columns: Optional list containing the names of columns to
            display by default in an interactive data table.
        other_metadata_columns: Optional list of column name strings.
        data_types: Dict mapping column names to DatasetColumnMetadata objects.
    """

    def __init__(
        self,
        media_columns: Optional[List[MediaInformationColumn]] = None,
        timestamp_columns: Optional[List[str]] = None,
        hidable_columns: Optional[List[str]] = None,
        default_columns: Optional[List[str]] = None,
        other_metadata_columns: Optional[List[str]] = None,
        default_topic_column: Optional[str] = None,
    ) -> None:
        # TODO: Should change these to be timestamp_columns_names.

        self.media_columns: List[MediaInformationColumn] = (
            media_columns if media_columns else []
        )
        self.timestamp_columns: List[str] = (
            timestamp_columns if timestamp_columns else []
        )
        self.hidable_columns: List[str] = hidable_columns if hidable_columns else []
        self.default_columns: Optional[List[str]] = default_columns
        self.other_metadata_columns: List[str] = (
            other_metadata_columns if other_metadata_columns else []
        )
        # will generally be set by the Dataset constructor
        # can be set manually if you really know what you're doing
        self.data_types: Dict[str, DatasetColumnMetadata] = {}

        self._default_topic_column = default_topic_column

    def timestamp_column(self, index=0) -> str:
        """Return the (string) name of the indexth timestamp column."""
        return self.timestamp_columns[index]

    def has_timestamp_column(self) -> bool:
        return len(self.timestamp_columns) > 0

    @property
    def long_text_columns(self) -> List[str]:
        """Columns containing large amounts of text data.

        These are candidates for topic or keyword analysis.
        """
        long_text_columns = [
            col
            for col, col_metadata in self.data_types.items()
            if col_metadata.text_type == TextDataType.long_text
        ]
        return long_text_columns

    @property
    def default_topic_column(self) -> Optional[str]:
        """Default column to use for topic analysis.

        If len(self.long_text_columns) == 0, will always be None.
        """
        if self._default_topic_column:
            return self._default_topic_column
        else:
            if self.long_text_columns:
                return self.long_text_columns[0]
            else:
                return None

    @property
    def embeddings(self):
        raise DeprecationWarning(
            "The `embeddings` property is deprecated. Use `CobaltDataset.embedding_metadata`"
        )

    def add_embedding(self, *_):
        raise DeprecationWarning(
            "The `add_embedding` method is deprecated. Use `CobaltDataset.add_embedding` instead"
        )

    def __eq__(self, other):
        if not isinstance(other, DatasetMetadata):
            return NotImplemented

        return all(
            [
                self.media_columns == other.media_columns,
                self.timestamp_columns == other.timestamp_columns,
                self.hidable_columns == other.hidable_columns,
                self.default_columns == other.default_columns,
                self.other_metadata_columns == other.other_metadata_columns,
                self.data_types == other.data_types,
                self._default_topic_column == other._default_topic_column,
            ]
        )

    def to_dict(self) -> dict:
        data_types_serialized = {k: v.to_dict() for k, v in self.data_types.items()}
        return {
            "media_columns": self.media_columns,
            "timestamp_columns": self.timestamp_columns,
            "hidable_columns": self.hidable_columns,
            "default_columns": self.default_columns,
            "other_metadata_columns": self.other_metadata_columns,
            "default_topic_column": self._default_topic_column,
            "data_types": data_types_serialized,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, serialized_data) -> DatasetMetadata:
        data = json.loads(serialized_data)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> DatasetMetadata:
        data_types = {}
        if "data_types" in data:
            serialized_data_types = data.pop("data_types")
            data_types = {
                k: DatasetColumnMetadata(**v) for k, v in serialized_data_types.items()
            }
            for _, v in data_types.items():
                v.col_type = ColumnDataType(v.col_type)
                if v.text_type:
                    v.text_type = TextDataType(v.text_type)
        instance = cls(**data)
        instance.data_types = data_types
        return instance

    def __repr__(self):
        return (
            f"DatasetMetadata(media_columns={self.media_columns}, "
            f"timestamp_columns={self.timestamp_columns}, "
            f"hidable_columns={self.hidable_columns}, "
            f"other_metadata_columns={self.other_metadata_columns}, "
            f"data_types={self.data_types})"
        )
