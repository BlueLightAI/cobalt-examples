# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.
import os

os.environ["KMP_WARNINGS"] = "FALSE"

from mapper.multiresolution import (
    DisjointPartitionGraph,
    HierarchicalPartitionGraph,
    MultiResolutionGraph,
)

from cobalt import schema
from cobalt._version import __version__
from cobalt.api_wrapper import get_api_client, setup_api_client
from cobalt.build_graph import FilterSpec, GraphSpec
from cobalt.config import check_license, settings
from cobalt.generate_tab_embeddings import get_tabular_embeddings
from cobalt.problem_group.schema import ProblemGroup
from cobalt.repositories.run_repository import GroupResultsCollection
from cobalt.schema.dataset import CobaltDataset, CobaltDataSubset
from cobalt.schema.embedding import ArrayEmbedding, ColumnEmbedding, Embedding
from cobalt.schema.group_collection import GroupCollection
from cobalt.schema.metadata import (
    DatasetMetadata,
    MediaInformationColumn,
)
from cobalt.schema.model_metadata import ModelMetadata
from cobalt.schema.split import DatasetSplit
from cobalt.schema.subset_collection import SubsetCollection
from cobalt.set_license import check_license_type, register_license, setup_license
from cobalt.tabular import load_tabular_dataset
from cobalt.ui import UI
from cobalt.workspace import Workspace

__all__ = [
    "__version__",
    "schema",
    "Workspace",
    "UI",
    "CobaltDataset",
    "CobaltDataSubset",
    "ModelMetadata",
    "DatasetMetadata",
    "MediaInformationColumn",
    "Embedding",
    "ArrayEmbedding",
    "ColumnEmbedding",
    "DatasetSplit",
    "ProblemGroup",
    "GroupResultsCollection",
    "MultiResolutionGraph",
    "HierarchicalPartitionGraph",
    "DisjointPartitionGraph",
    "GraphSpec",
    "FilterSpec",
    "load_tabular_dataset",
    "get_tabular_embeddings",
    "check_license",
    "setup_api_client",
    "get_api_client",
    "setup_license",
    "register_license",
]

check_license_type()
