# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.
from __future__ import annotations

import warnings
from math import ceil, floor
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from uuid import uuid4

import mapper
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.stats
from mapper import HierarchicalPartitionGraph

from cobalt import schema
from cobalt.build_graph import GraphBuilder
from cobalt.cobalt_types import GroupType
from cobalt.config import handle_cb_exceptions, is_colab_environment
from cobalt.event_bus import EventBusController
from cobalt.feature_compare import feature_compare
from cobalt.multires_graph import create_multilandscape
from cobalt.problem_group import failure_group, group_labeling
from cobalt.problem_group.autogroup import autogroup_modularity
from cobalt.problem_group.schema import Cluster, ProblemGroup
from cobalt.repositories.run_repository import GroupResultsCollection, RunType
from cobalt.repositories.server_repository import SERVER_REGISTRY
from cobalt.schema import DatasetSplit, SplitDescriptor, evaluation_metric
from cobalt.schema.dataset import CobaltDataSubset
from cobalt.schema.group_collection import GroupCollection, GroupMetadata
from cobalt.schema.metadata import TextDataType
from cobalt.schema.model_metadata import ModelMetadata
from cobalt.state import State
from cobalt.table.table_select import TableSelector
from cobalt.ui import UI
from cobalt.visualization import GraphManager

# joblib (or maybe pynndescent/numba) uses attributes deprecated in Python 3.12
warnings.filterwarnings("ignore", category=DeprecationWarning, module="joblib")
# numba can use different threading backends but will warn if it can't use an old version of TBB
warnings.filterwarnings(
    "ignore", message="The TBB threading layer requires TBB version"
)


class Workspace:
    """Encapsulates analysis done with a dataset and models.

    Attributes:
        ui: A user interface that can be used to interact with the data, models,
            and other analysis.
        run_auto_group_analysis: Whether to automatically run a group analysis
            of the data and models when the UI is opened, if no analysis has yet
            been run.
    """

    def __init__(
        self,
        dataset: schema.CobaltDataset,
        split: Optional[Union[schema.DatasetSplit, SplitDescriptor]] = None,
        auto_graph: bool = True,
        run_server: Optional[bool] = None,
    ):
        """Initialize a Workspace.

        Args:
            dataset: The CobaltDataset to use for the analysis.
            split: A division of the dataset into predetermined groups, e.g. test/train.
            auto_graph: Whether to automatically run the graph creation.
            run_server: Whether to run a web server to host images. If None
                (default), will run a server unless a Colab environment is detected.

        The dataset split can be provided in a number of different ways.
        """
        self.workspace_id = uuid4()
        if isinstance(split, DatasetSplit):
            self.state = State(dataset, split, workspace_id=self.workspace_id)
        else:
            self.state = State(
                dataset, DatasetSplit(dataset, split), workspace_id=self.workspace_id
            )
        if run_server is None:
            run_server = not is_colab_environment()
        self.run_server = run_server

        # Collect metrics for initial dataset
        self.state.collect_metrics(field="all", dataset=dataset)
        self.run_event_bus = EventBusController.get_run_event_bus(self.workspace_id)

        self.split = split
        self.auto_graph = auto_graph

        media_columns = self.state.metadata.media_columns
        if media_columns:
            for mc in media_columns:
                if not mc.is_remote and self.run_server:
                    SERVER_REGISTRY.add_server(media_columns[0].host_directory)
        SERVER_REGISTRY.run_servers()

        self.graph_builder = GraphBuilder(self.state)
        self.run_auto_group_analysis = self.auto_graph
        self.ui = self._init_ui(dataset)

    @staticmethod
    def from_arrays(
        model_inputs: Union[List, np.ndarray, pd.DataFrame],
        model_predictions: np.ndarray,
        ground_truth: Optional[np.ndarray],
        task: str = "classification",
        embedding: Optional[np.ndarray] = None,
        embeddings: Optional[List[np.ndarray]] = None,
        embedding_metric: Optional[str] = None,
        embedding_metrics: Optional[List[str]] = None,
        split: Optional[Union[schema.DatasetSplit, SplitDescriptor]] = None,
    ):
        """Returns a Workspace object constructed from user-defined arrays.

        Args:
            model_inputs: the data evaluated by the model.
            model_predictions: the model's predictions corresponding to `model_inputs`.
            ground_truth: ground truths for `model_inputs`.
            task: model task, pass in "classification"
            embedding: embedding array to include.
            embeddings: list of embedding arrays to use.
            embedding_metric: embedding metric corresponding to embedding.
            embedding_metrics: list of metrics corresponding to embeddings.
            split: an optional dataset split.

        At most one of ``embedding`` or ``embeddings`` (and the corresponding
        ``embedding_metric`` or ``embedding_metrics``) should be provided.
        """
        # TODO: Weighing whether it should be a secondary step to provide groups.
        # What we could do is create a different analysis with different splits.

        if isinstance(model_inputs, pd.DataFrame):
            df = model_inputs
            input_columns = list(model_inputs.columns)
        else:
            if isinstance(model_inputs, np.ndarray) and len(model_inputs.shape) > 1:
                raise ValueError(
                    "model_inputs should be One Dimensional if an np.ndarray."
                )

            df = pd.DataFrame({"inputs": model_inputs})
            input_columns = "inputs"

        df["predictions"] = model_predictions

        if ground_truth is not None:
            df["target"] = ground_truth

        dataset = schema.CobaltDataset(df)

        dataset.add_model(
            input_columns=input_columns,
            target_column="target",
            prediction_column="predictions",
            task=task,
            name="model_1",
        )

        if embedding is not None:
            dataset.add_embedding_array(embedding, embedding_metric)

        if embeddings is not None:
            for e, m in zip(embeddings, embedding_metrics):
                dataset.add_embedding_array(e, m)

        return Workspace(dataset, split)

    def view_table(
        self,
        subset: Optional[Union[List[int], CobaltDataSubset]] = None,
        display_columns: Optional[List[str]] = None,
        max_rows: Optional[int] = None,
    ):
        """Returns a visualization of the dataset table."""
        if not isinstance(subset, CobaltDataSubset):
            subset: CobaltDataSubset = (
                self.state.dataset.subset(subset)
                if subset is not None
                else self.state.dataset.as_subset()
            )

        media_columns = subset.metadata.media_columns
        image_columns = [
            img_col.autoname_media_visualization_column() for img_col in media_columns
        ]
        html_columns = subset.metadata.long_text_columns

        if display_columns is None and subset.metadata.default_columns is not None:
            display_columns = subset.metadata.default_columns

        return TableSelector(
            subset,
            self.state,
            workspace_id=self.workspace_id,
            columns=display_columns,
            image_columns=image_columns,
            html_columns=html_columns,
            max_rows_to_display=max_rows,
        )

    def _init_ui(self, dataset):
        ui = UI(workspace=self, dataset=dataset)
        ui.workspace_id = self.workspace_id
        return ui

    @staticmethod
    def analyze(
        subset: CobaltDataSubset,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Compute numerical and categorical statistics for the given subset.

        Returns:
            A tuple (numerical_statistics, categorical statistics) giving
            summary statistics for numerical and categorical features in the
            dataset.
        """
        return subset.get_summary_statistics()

    @staticmethod
    def feature_compare(
        group_1: CobaltDataSubset,
        group_2: Union[CobaltDataSubset, Literal["all", "rest", "neighbors"]],
        numerical_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        numerical_test: Literal["t-test"] = "t-test",
        categorical_test: Literal["G-test"] = "G-test",
        include_nan: bool = False,
    ):
        """Compare the distributions of features between two subsets."""
        if group_2 == "neighbors":
            raise NotImplementedError(
                "'neighbors' comparison group is not yet implemented"
            )
        elif group_2 == "all":
            group_2 = group_1.source_dataset.as_subset()
        elif group_2 == "rest":
            group_2 = group_1.complement()

        return feature_compare(
            group_1,
            group_2,
            numerical_features,
            categorical_features,
            numerical_test,
            categorical_test,
            include_nan=include_nan,
        )

    def get_groups(self) -> GroupCollection:
        """Get a GroupCollection object with the currently saved groups.

        Returns:
            GroupCollection read-only object with groups.
            A group consists of a subset of data points together with some metadata
            about the subset.
        """
        raw_groups = self.state.get_groups().copy()
        if not raw_groups:
            return raw_groups
        g_metadata_list = []
        for name, subset in raw_groups.items():
            g_metadata_list.append(GroupMetadata(name=name, subset=subset))
        g_collection = GroupCollection.from_groups(g_metadata_list)
        return g_collection

    @property
    def saved_groups(self) -> GroupCollection:
        """An object that represents the currently saved groups.

        This does not include groups selected by algorithms like
        ``find_failure_groups()``, only groups saved manually in the UI or with
        ``Workspace.add_group()``.
        """
        # TODO: make this behave nicely with setters?
        return self.get_groups()

    def add_group(self, name: str, group: schema.CobaltDataSubset):
        """Add a group to the collection of saved groups."""
        self.state.add_group(group=group, group_name=name)
        self.state.collect_metrics(field="groups", group_id=name, dataset=group)

    @property
    def graphs(self) -> Dict[str, HierarchicalPartitionGraph]:
        """The graphs that have been created and saved."""
        graphs_dict = self.state.graph_repo.get_graphs()
        return {name: graph["graph"].graph for name, graph in graphs_dict.items()}

    def add_graph(
        self,
        name: str,
        graph: HierarchicalPartitionGraph,
        subset: schema.CobaltDataSubset,
        init_max_nodes: int = 500,
        init_max_degree: float = 15.0,
    ):
        """Add a graph to self.graphs.

        Args:
            name (str): A name for the graph.
            graph: The graph to add.
            subset: The subset of the self.dataset this graph is constructed from.
            init_max_nodes: The maximum number of nodes to show in the initial view
                of this graph.
            init_max_degree: The maximum average node degree for the initial view of this graph.
        """
        new_landscape_object = create_multilandscape(
            graph, subset, init_max_nodes, init_max_degree
        )

        self.state.add_graph(graph=new_landscape_object, graph_name=name)

    def new_graph(
        self,
        name: Optional[str] = None,
        subset: Optional[Union[str, schema.CobaltDataSubset]] = None,
        embedding: Union[int, str, schema.Embedding] = 0,
        metric: Optional[str] = None,
        init_max_nodes: int = 500,
        init_max_degree: float = 15.0,
        **kwargs,
    ) -> HierarchicalPartitionGraph:
        """Create a new graph from a specified subset.

        The resulting graph will be returned and added to the Workspace.

        Args:
            name: The name to give the graph in self.graphs. If None: Autoname it.
            subset: The subset of the dataset to include in the graph. If a string,
                will try to use a subset with that name from the dataset split or the
                saved groups (in that order). Otherwise, should be a `CobaltDataSubset`.
            embedding: The embedding to use to generate the graph. May be specified
                as an index into self.dataset.embeddings, the name of the embedding, or
                an `Embedding` object.
            metric: The distance metric to use when constructing the graph. If none
                is provided, will use the metric specified by the embedding.
            init_max_nodes: The maximum number of nodes to show in the initial view
                of this graph.
            init_max_degree: The maximum average node degree for the initial view of this graph.
            **kwargs: Any additional keyword parameters will be interpreted as parameters to
                construct a `GraphSpec` object.
        """
        if subset is None:
            subset = self.state.dataset.as_subset()

        g = self.graph_builder.new_graph(
            subset=subset, embedding=embedding, metric=metric, **kwargs
        )

        subset_ = self.graph_builder.get_subset(subset)

        if name is None:
            name = GraphManager.generate_new_graph_name(self.state)

        self.add_graph(
            name,
            g,
            subset_,
            init_max_nodes=init_max_nodes,
            init_max_degree=init_max_degree,
        )

        return g

    def get_graph_level(
        self,
        graph: Union[str, HierarchicalPartitionGraph],
        level: int,
        name: Optional[str] = None,
    ) -> GroupCollection:
        """Create a GroupCollection from a specified level of a graph.

        This method is experimental and its interface may be changed in the
        future.

        Args:
            graph: Name of the graph to use, or the graph object itself.
            level: The level of the graph to use for the groups. One group will
                be created for each node in the graph.
            name: An optional name for the GroupCollection.
        """
        if isinstance(graph, str):
            graph_data = self.state.graph_repo.get_graph(graph)
            landscape = graph_data["graph"]
            graph: HierarchicalPartitionGraph = landscape.graph
            source = graph_data["subset"]
        else:
            source = graph.cobalt_subset
        group_collection = GroupCollection.from_indices(
            source, graph.levels[level].nodes, name=name
        )
        return group_collection

    def get_graph_levels(
        self,
        graph: Union[str, HierarchicalPartitionGraph],
        min_level: int,
        max_level: int,
        name_prefix: Optional[str] = None,
    ) -> Dict[int, GroupCollection]:
        """Create GroupCollections for a range of levels of a graph.

        All levels between min_level and max_level will be used. The return
        value is a dict mapping levels to GroupCollections.

        This method is experimental and its interface may be changed in the
        future.

        Args:
            graph: Name of the graph to use, or the graph object itself.
            min_level: The lowest level of the graph to use for the groups.
            max_level: The highest level of the graph to use for the groups.
            name_prefix: If provided, the GroupCollection for level i will be
                named "{name_prefix}_{i}".
        """
        return {
            level: self.get_graph_level(
                graph, level, name=f"{name_prefix}_{level}" if name_prefix else None
            )
            for level in range(min_level, max_level + 1)
        }

    def add_evaluation_metric_values(
        self,
        name: str,
        metric_values: npt.ArrayLike,
        model: Union[int, str, ModelMetadata] = 0,
        lower_values_are_better: bool = True,
    ):
        """Add values for a custom evaluation metric.

        Args:
            name: A name for this evaluation metric. This will be used to name a
                column in the dataset where these values will be stored, as well as
                to name the metric itself.
            metric_values: An arraylike with one value for each data point in the dataset.
            model: The name or index of the model in self.dataset that this metric evaluates.
            lower_values_are_better: If True, Cobalt will interpret lower values
                of this metric as positive; otherwise, it will interpret higher
                values as positive.
        """
        dataset = self.state.dataset
        if len(metric_values) != len(dataset):
            raise ValueError("Not enough metric values for this dataset.")
        model_metadata = self._get_model(model)
        if name in model_metadata.evaluation_metrics:
            raise ValueError(f"Evaluation metric with name '{name}' already exists.")
        column_name = self._find_available_column_name(dataset.df.columns, name)
        dataset.set_column(column_name, metric_values)
        metric_metadata = evaluation_metric.ColumnEvaluationMetric(
            name, column_name, lower_values_are_better
        )
        model_metadata.evaluation_metrics[name] = metric_metadata

    def find_drifted_groups(
        self,
        reference_group: Union[str, CobaltDataSubset],
        comparison_group: Union[str, CobaltDataSubset],
        embedding: int = 0,
        relative_prevalence_threshold: float = 2,
        p_value_threshold: float = 0.05,
        min_size: int = 5,
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        manual: bool = True,
        visible: bool = True,
        generate_group_descriptions: bool = True,
        model: Union[int, str, ModelMetadata] = 0,
    ) -> GroupResultsCollection:
        """Return groups in the comparison group that are underrepresented in the reference group.

        Args:
            reference_group: The reference subset of the data, e.g. the training set.
            comparison_group: The subset of the data that may have regions that
                are not well represented in the reference set. This may be a test
                dataset or production data.
            embedding: The embedding to use for the analysis. If none is provided,
                will use the default dataset embedding. (If one does not exist, will
                raise an error.)
            relative_prevalence_threshold: How much more common points from
                comparison_group need to be in a group relative to the overall
                average for it to be considered drifted. This is computed by
                comparing the ratio of comparison points to reference points in a
                group, compared with the ratio in the overall dataset. If the
                overall balance of points is 1:1 from each group and
                relative_prevalence_threshold = 2, a drifted group will have
                at least a 2:1 balance in favor of data points from the comparison
                set. If the overall ratio of points is 1:2 comparison : reference,
                then a drifted group will need to have at least a 1:1 ratio.

                Choose this value based on what amount of overrepresentation of
                the comparison group would be meaningful to you. Under the
                default parameter of 2, the interpretation is roughly that for
                any returned group, points from the comparison subset are at
                least twice as common as they would be in a random sample of
                data points.
            p_value_threshold: Used in a significance test that the prevalence
                of points from the comparison group is at least as high as required
                based on the value of relative_prevalence_threshold.
            min_size: The minimum number of data points that need to be in the drifted
                region else, the drifted region is dropped from the result
            run_name: A name under which to store the results. If one is not
                provided, it will be chosen automatically.
            config: A dictionary containing further configuration parameters
                that will be passed to the underlying algorithm.
            manual: Used internally to signal whether the failure group analysis
                was created by the user.
            visible: Whether to show the results of this analysis in the UI.
            generate_group_descriptions: Whether to generate statistical and
                textual descriptions of returned groups. True by default, but
                consider setting to False for large datasets with many columns, as
                this process can be very time consuming.
            model: Index or name of the model whose error metric will be shown
                with the returned groups.

        Returns:
            A GroupResultsCollection object containing the discovered drifted
            groups and the parameters used by the algorithm.
        """
        if config is None:
            config = {}
        subset1 = self._get_subset(reference_group)
        subset2 = self._get_subset(comparison_group)

        if not subset1.embedding_metadata or not subset2.embedding_metadata:
            return None
        combined = subset1.concatenate(subset2)
        _, embedding = self._get_embedding(combined, embedding)
        drift_score = np.zeros(len(subset1) + len(subset2))
        drift_score[len(subset1) :] = 1

        comparison_prob = len(subset2) / len(combined)
        comparison_odds = len(subset2) / len(subset1)
        # we're looking at the odds ratio between the overall dataset and each individual group
        threshold_odds = comparison_odds * relative_prevalence_threshold
        threshold_prob = threshold_odds / (threshold_odds + 1)

        raw_groups, params = failure_group.find_failure_groups_superlevel_auto(
            combined,
            drift_score,
            embedding,
            # don't include nodes in a group if the ratio is _less_ than the overall ratio
            threshold=comparison_prob,
            use_merge_tree=True,
            **config,
        )

        params["prod"] = subset2

        # Remove drifted groups which have fewer data points than min_size.
        raw_groups = [group for group in raw_groups if len(group) >= min_size]

        # remove drifted groups which don't pass the hypothesis test for being above threshold_prob
        group_sizes = [len(g) for g in raw_groups]
        comparison_counts = [subset2.intersection_size(g) for g in raw_groups]
        p_vals = [
            scipy.stats.binomtest(
                count, size, p=threshold_prob, alternative="greater"
            ).pvalue
            for count, size in zip(comparison_counts, group_sizes)
        ]
        raw_groups = [
            group
            for group, p_val in zip(raw_groups, p_vals)
            if p_val < p_value_threshold
        ]

        if run_name:
            group_name_prefix = run_name
        else:
            run_name = self.state.run_repo.new_run_name()
            group_name_prefix = f"{run_name}_drift"

        if "full_graph" in params:
            self.add_graph(
                f"{group_name_prefix}_graph",
                params["full_graph"],
                combined,
            )

        error_metric = None
        if self.state.dataset.models:
            model_metadata = self._get_model(model)
            if "error" in model_metadata.performance_metric_keys():
                error_metric = model_metadata.get_performance_metric_for("error")

        problem_groups: List[ProblemGroup] = []
        for group in raw_groups:
            prod_fraction = group.intersection_size(subset2) / len(group)
            metrics = {"production fraction": prod_fraction}
            if error_metric is None:
                severity = prod_fraction
                problem_description = (
                    f"Drifted group | {len(group)} points "
                    f"| {100 * prod_fraction :.1f}% new"
                )
            else:
                error_rate = error_metric.overall_score(group)
                metrics.update(error_rate)
                severity = error_rate["error"]
                problem_description = (
                    f"Drifted group | {len(group)} points "
                    f"| {100 * prod_fraction :.1f}% new "
                    f"| {100 * error_rate['error'] :.1f}% error "
                )

            problem_group = ProblemGroup(
                name="",
                subset=group,
                severity=severity,
                problem_description=problem_description,
                metrics=metrics,
                group_type=GroupType.drift,
            )
            problem_groups.append(problem_group)

        problem_groups.sort(
            key=lambda g: (round(g.severity, 3), len(g.subset)),
            reverse=True,
        )
        for i, g in enumerate(problem_groups):
            g.name = f"{group_name_prefix}/{i + 1}"
        # TODO: model performance metrics? histograms?

        if generate_group_descriptions:
            group_labeling.add_feature_group_descriptions(
                problem_groups, omit_columns=subset1.metadata.hidable_columns
            )

            group_labeling.add_text_keyword_group_descriptions(problem_groups)

        run: GroupResultsCollection = self.state.run_repo.build_run(
            run_type=RunType.MANUAL if manual else RunType.AUTOMATIC,
            groups=problem_groups,
            run_name=run_name,
            algorithm="drift",
            group_type=GroupType.drift,
            source_data=combined,
            params=params,
            visible=visible,
        )

        if visible:
            self.run_event_bus.run_all()

        return run

    @property
    def drifted_groups(self) -> Dict[str, GroupResultsCollection]:
        """The collection of all drifted group analysis results."""
        return self.state.run_repo.get_runs(group_type=GroupType.drift)

    # TODO: better way of handling user-specified graph
    def find_failure_groups(
        self,
        method: Literal["superlevel"] = "superlevel",
        subset: Optional[Union[schema.CobaltDataSubset, str]] = None,
        model: Union[int, str, ModelMetadata] = 0,
        embedding: Union[int, str, schema.Embedding] = 0,
        failure_metric: Optional[Union[str, pd.Series]] = None,
        min_size: int = 1,
        max_size: Union[int, float] = np.inf,
        min_failures: int = 3,
        config: Optional[Dict[str, Dict]] = None,
        run_name: Optional[str] = None,
        manual: bool = True,
        visible: bool = True,
        generate_group_descriptions: bool = True,
    ) -> GroupResultsCollection:
        """Run an analysis to find failure groups in the dataset.

        Saves the results in self.failure_groups under `run_name`.

        Args:
            method: Algorithm to use for finding failure groups. Currently only
                "superlevel" is supported.
            subset: The subset of the data on which to perform the analysis. If none
                is provided, will use the entire dataset.
            model: Index or name of the model for which failure groups should be found.
            embedding: The embedding to use for the analysis. If none is provided,
                will use the default dataset embedding. (If one does not exist, will
                raise an error.)
            failure_metric: The performance metric to use. If a string, will use the
                model performance metric with that name; otherwise, must be a Pandas
                Series, with length either equal to the length of the specified subset,
                or the whole dataset. If a Series is passed, it will be added to the
                dataset as a model evaluation metric.
            min_size: The minimum size for a returned failure group. Smaller groups
                will be discarded.
            max_size: The maximum size for a returned failure group. Larger groups
                will be split into smaller groups by applying a clustering algorithm.
            min_failures: The minimum number of failure for a returned failure groups.
                Smaller groups will be discarded. Default is set to 3 to allow DS to spot
                failure patterns. This is only for classification tasks.
            config: A dictionary containing further configuration parameters
                that will be passed to the underlying algorithm.
            run_name: A name under which to store the results. If one is not
                provided, it will be chosen automatically.
            manual: Used internally to signal whether the failure group analysis
                was created by the user.
            visible: Whether to show the results of this analysis in the UI.
            generate_group_descriptions: Whether to generate statistical and
                textual descriptions of returned groups. True by default, but
                consider setting to False for large datasets with many columns, as
                this process can be very time consuming.

        Returns:
            A GroupResultsCollection object containing the discovered failure
            groups and the parameters used by the algorithm.
        """
        if method != "superlevel":
            raise NotImplementedError(f"method {method} is not implemented")

        if subset is None:
            subset = self.state.dataset.as_subset()
        subset = self._get_subset(subset)
        if not subset.embedding_metadata:
            return None
        _, embedding = self._get_embedding(subset, embedding)

        model_metadata = self._get_model(model)

        if failure_metric is None:
            failure_metric = model_metadata.performance_metric_keys()[0]
        if isinstance(failure_metric, str):
            failure_metric_name = failure_metric
        else:
            if len(failure_metric) < len(self.state.dataset):
                warnings.warn(
                    (
                        "Not enough failure values provided for dataset. "
                        "Attempting to assign to specified subset."
                        "Data points outside the subset will have missing "
                        "values for this metric."
                    ),
                    stacklevel=2,
                )
                failure_values = np.full(len(self.state.dataset), np.nan)
                failure_values[subset.indices] = failure_metric
            else:
                failure_values = failure_metric
            self.add_evaluation_metric_values(
                failure_metric.name, failure_values, model=model
            )
            failure_metric_name = str(failure_metric.name)

        failure_values = subset.get_model_performance_data(failure_metric_name, model)
        metric_metadata = model_metadata.get_performance_metric_for(failure_metric_name)
        if not metric_metadata.lower_values_are_better:
            failure_values = -np.array(failure_values, dtype=float)

        # TODO: use already-generated graph if it exists

        if config is None:
            config = {}
        config = config.get(method, config)
        raw_failure_groups, params = failure_group.find_failure_groups_superlevel_auto(
            subset, failure_values, embedding, **config
        )

        params["model"] = model_metadata

        if "full_graph" in params:

            def split_group(gp):
                return failure_group.split_group(
                    subset,
                    gp,
                    params["full_graph"],
                    max_size,
                    params["graph_level"],
                )

        else:

            def split_group(gp):
                return failure_group.split_group_new_graph(gp, embedding, max_size)

        filtered_failure_groups = failure_group.split_groups_max_size(
            raw_failure_groups,
            max_size,
            split_group,
        )

        # if we've split some groups, there may be some that don't exceed the threshold now
        filtered_failure_groups = failure_group.filter_groups_metric_threshold(
            filtered_failure_groups, metric_metadata, params["threshold"]
        )

        filtered_failure_groups = failure_group.filter_groups_min_errors(
            filtered_failure_groups, model_metadata, min_failures
        )
        filtered_failure_groups = [
            gp for gp in filtered_failure_groups if len(gp) >= min_size
        ]

        failure_groups = [
            group_labeling.create_failure_group_metadata(
                fg,
                model_metadata,
                failure_metric_name,
                severity_is_inverse=not metric_metadata.lower_values_are_better,
            )
            for fg in filtered_failure_groups
        ]

        group_labeling.add_model_data_histograms(failure_groups, model_metadata)

        if generate_group_descriptions:
            group_labeling.add_feature_group_descriptions(
                failure_groups,
                omit_columns=[
                    *subset.metadata.hidable_columns,
                    *model_metadata.error_columns,
                ],
            )

            topic_column = subset.metadata.default_topic_column
            # override the default to be the model input column if it's text
            if (
                model_metadata.input_columns is not None
                and len(model_metadata.input_columns) == 1
            ):
                input_column = model_metadata.input_columns[0]
                if (
                    subset.metadata.data_types[input_column].text_type
                    == TextDataType.long_text
                ):
                    topic_column = input_column

            group_labeling.add_text_keyword_group_descriptions(
                failure_groups, summary_column=topic_column
            )

        # sort by rounded metric value, then by size
        # most of the time the sort by size won't be meaningful
        failure_groups.sort(
            key=lambda g: (round(g.severity, 3), len(g.subset)),
            reverse=True,
        )

        if run_name:
            group_name_prefix = run_name
        else:
            run_name = self.state.run_repo.new_run_name()
            group_name_prefix = f"{run_name}_fg"

        for i, gp in enumerate(failure_groups):
            gp.name = f"{group_name_prefix}/{i + 1}"

        # BuildRun and automatically and it to the repos
        run: GroupResultsCollection = self.state.run_repo.build_run(
            run_type=RunType.MANUAL if manual else RunType.AUTOMATIC,
            groups=failure_groups,
            run_name=run_name,
            algorithm=method,
            group_type=GroupType.failure,
            source_data=subset,
            params=params,
            visible=visible,
        )
        self.state.collect_metrics_for_current_run(run)

        if "full_graph" in params:
            self.add_graph(
                f"{run_name}_graph",
                params["full_graph"],
                subset,
            )

        if visible:
            self.run_event_bus.run_all()

        return run

    @handle_cb_exceptions
    def _get_or_create_displayed_groups(self) -> List[ProblemGroup]:
        runs = self.state.run_repo.get_runs(run_visible=True)

        if not runs and (
            not self.run_auto_group_analysis
            or not self.state.dataset.embedding_metadata
        ):
            return []

        if not runs and self.state.dataset._has_model_metadata():
            # can run failure group analysis
            run = self.find_failure_groups(
                run_name="auto_fg", manual=False, visible=True
            )
            runs["auto_fg"] = run

        # run autogrouping if we couldn't do failure groups
        if not runs:
            autogroup_run = self.find_clusters(
                run_name="auto_cluster", manual=False, visible=True
            )
            if autogroup_run:
                runs["auto_cluster"] = autogroup_run

        groups = []
        for run in runs.values():
            groups.extend(run.groups)

        return groups

    def _get_displayed_groups(self) -> List[ProblemGroup]:
        runs = self.state.run_repo.get_runs(run_visible=True)

        groups = []
        for run in runs.values():
            groups.extend(run.groups)

        return groups

    @property
    def failure_groups(self) -> Dict[str, GroupResultsCollection]:
        """The collection of all failure group analysis results."""
        return self.state.run_repo.get_runs(group_type=GroupType.failure)

    def find_clusters(
        self,
        method: Literal["modularity"] = "modularity",
        subset: Optional[Union[schema.CobaltDataSubset, str]] = None,
        graph: Optional[mapper.HierarchicalPartitionGraph] = None,
        embedding: Union[int, str, schema.Embedding] = 0,
        min_group_size: Union[int, float] = 1,
        max_group_size: Union[int, float] = np.inf,
        max_n_groups: int = 10000,
        min_n_groups: int = 1,
        config: Optional[Dict[str, Any]] = None,
        run_name: Optional[str] = None,
        manual: bool = True,
        visible: bool = True,
        generate_group_descriptions: bool = True,
    ) -> GroupResultsCollection:
        """Run an analysis to find natural clusters in the dataset.

        Saves the results in self.clustering_results under `run_name`.

        Args:
            method: Algorithm to use for finding clusters. Currently only
                "modularity" is supported.
            subset: The subset of the data on which to perform the analysis. If none
                is provided, will use the entire dataset.
            graph: A graph to use for the clustering. If none is provided, will
                create a new graph based on the specified embedding. Note that
                if a graph is provided, it must be built on the subset specified
                by the ``subset`` parameter.
            embedding: The embedding to use to create a graph if none is
                provided. If none is provided, will use the default dataset
                embedding. (If one does not exist, will raise an error.)
            min_group_size: The minimum size for a returned cluster. If a value
                between 0 and 1 is provided, it will be interpreted as a fraction of
                the size of the subset of data being clustered.
            max_group_size: The maximum size for a returned cluster. If a value
                between 0 and 1 is provided, it will be interpreted as a fraction of
                the size of the subset of data being clustered.
            max_n_groups: The maximum number of clusters to return.
            min_n_groups: The minimum number of clusters to return.
            config: A dictionary containing further configuration parameters
                that will be passed to the underlying algorithm.
            run_name: A name under which to store the results. If one is not
                provided, it will be chosen automatically.
            manual: Used internally to signal whether the clustering analysis
                was created by the user.
            visible: Whether to show the results of this analysis in the UI.
            generate_group_descriptions: Whether to generate statistical and
                textual descriptions of returned clusters. True by default, but
                consider setting to False for large datasets with many columns, or
                when a large number of clusters is desired, as this process can be
                very time consuming.


        Returns:
            A GroupResultsCollection object containing the discovered clusters
            and the parameters used by the algorithm.
        """
        if method != "modularity":
            raise NotImplementedError(f"method {method} is not implemented")

        if subset is None:
            subset = self.state.dataset.as_subset()
        subset = self._get_subset(subset)
        if not subset.embedding_metadata and not graph:
            return None

        if graph is None:
            graph = self.new_graph("autogroup_graph", subset, embedding)

        if min_group_size < 1:
            min_group_size = ceil(len(subset) * min_group_size)
        if max_group_size < 1:
            max_group_size = floor(len(subset) * max_group_size)

        if min_group_size > max_group_size:
            raise ValueError("min_group_size must not be greater than max_group_size")
        if min_n_groups > max_n_groups:
            raise ValueError("min_n_groups must not be greater than max_n_groups")

        raw_groups, params = autogroup_modularity(
            subset,
            graph=graph,
            min_group_size=min_group_size,
            max_group_size=max_group_size,
            max_n_groups=max_n_groups,
            min_n_groups=min_n_groups,
        )

        if run_name:
            group_name_prefix = run_name
        else:
            run_name = self.state.run_repo.new_run_name()
            group_name_prefix = f"{run_name}_cluster"

        raw_groups.sort(key=lambda g: len(g), reverse=True)

        groups = [
            Cluster(
                name=f"{group_name_prefix}/{i + 1}",
                subset=subset,
                problem_description=f"{len(subset)} points",
            )
            for i, subset in enumerate(raw_groups)
        ]

        if self.state.dataset.model:
            group_labeling.add_model_data_histograms(groups, self.state.dataset.model)

        if generate_group_descriptions:
            group_labeling.add_feature_group_descriptions(
                groups, omit_columns=subset.metadata.hidable_columns
            )

            group_labeling.add_text_keyword_group_descriptions(groups)

        run: GroupResultsCollection = self.state.run_repo.build_run(
            run_type=RunType.MANUAL if manual else RunType.AUTOMATIC,
            groups=groups,
            run_name=run_name,
            algorithm=method,
            group_type=GroupType.cluster,
            source_data=subset,
            params=params,
            visible=visible,
        )

        if visible:
            self.run_event_bus.run_all()

        return run

    @property
    def clustering_results(self) -> Dict[str, GroupResultsCollection]:
        """Results from all previous runs of the clustering algorithm."""
        return self.state.run_repo.get_runs(group_type=GroupType.cluster)

    def _get_subset(self, subset: Union[str, CobaltDataSubset]) -> CobaltDataSubset:
        return self.graph_builder.get_subset(subset)

    def _get_embedding(
        self, subset: CobaltDataSubset, embedding: Union[int, str, schema.Embedding] = 0
    ) -> Tuple[np.ndarray, schema.Embedding]:
        return self.graph_builder.get_embedding(subset, embedding)

    def _get_model(self, model: Union[int, str, ModelMetadata]) -> ModelMetadata:
        if isinstance(model, ModelMetadata):
            return model
        return self.state.dataset.models[model]

    @staticmethod
    def _find_available_column_name(columns: List[str], desired_name: str) -> str:
        name = desired_name
        i = 0
        while name in columns:
            name = f"{desired_name}_{i}"
            i += 1
        return name

    def export_groups_as_dataframe(self) -> pd.DataFrame:
        """Exports saved groups as a DataFrame.

        The columns of the resulting DataFrame are named after the saved groups, and
        the column for each group contains a boolean mask indicating which data
        points in the dataset belong to that group.
        """
        groups = self.get_groups()
        d = {}
        for group in groups.metadata:
            indicator_function = group.subset.as_mask()
            g_name = group.name
            d[g_name] = indicator_function
        return pd.DataFrame(d)

    def import_groups_from_dataframe(self, df: pd.DataFrame):
        """Imports groups from a DataFrame with one column for each group.

        The name of each column will be used as the name for the group,
        and the entries in the column will be interpreted as boolean values
        indicating the membership of each data point in that group.
        """
        groups = [
            {"group_name": col, "group": self.state.dataset.mask(df[col])}
            for col in df.columns
        ]
        self.state.add_groups(groups)

    def _get_summary(self):
        summary_data = {"splits": {}, "models": []}

        for split_name, split_data in self.state.split.items():
            summary_data["splits"][split_name] = {
                "points_number": len(split_data.indices)
            }

        for model in self.state.dataset.models:
            summary_data["models"].append(
                {
                    "model_name": model.name,
                    "model_metadata": {
                        "task": model.task.name,
                        "input_columns": model.input_columns,
                        "outcome_columns": model.outcome_columns,
                        "prediction_columns": model.prediction_columns,
                    },
                }
            )

        return summary_data

    def auto_analysis(
        self,
        ref: Union[str, CobaltDataSubset],
        cmp: Union[str, CobaltDataSubset],
        model: Union[int, str, ModelMetadata] = 0,
        embedding: Union[int, str, schema.Embedding] = 0,
        failure_metric: Optional[Union[str, pd.Series]] = None,
        min_size: int = 3,
        min_failures: int = 3,
        config: Optional[Dict[str, Dict]] = None,
        run_name: Optional[str] = None,
        manual: bool = True,
        visible: bool = True,
    ):
        """Returns an analysis of errors and warnings with the data and model.

        Args:
            ref: The subset of the data on which to do the reference analysis.
                Users should typically pass in the training dataset.
            cmp: The subset of the data on which to do the comparison analysis.
                Users may pass in a test dataset, or a production dataset.
            model: The index or name of the model object you want to consider.
            embedding: The embedding to use to create a graph if none is
                provided. If none is provided, will use the default dataset
                embedding. (If one does not exist, will raise an error.)
            failure_metric: The failure metric to use to find error patterns based on.
            min_size: The minimum size of a returned group.
            min_failures: The minimum number of failures in a failure group, for a
                classification task.
            config: A dictionary containing further configuration parameters
                that will be passed to the underlying algorithm.
            run_name: A name under which to store the results. If one is not
                provided, it will be chosen automatically.
            manual: Used internally to signal whether the clustering analysis
                was created by the user.
            visible: Whether to show the results of this analysis in the UI.

        Returns:
            a dictionary with keys "summaries" and "groups"

            Under "summaries" is a tuple of two DataFrames. The first is a table
            summarizing the discovered error groups; the second is a table
            summarizing the discovered warning groups.

            Under "groups" is a tuple of two lists of CobaltDataSubsets, the
            first listing the error groups, and the second listing the warning
            groups.
        """
        # TODO: Add other features to "Explain" our failure modes and make them
        # actionable and prioritizable in a variety of use cases.

        # Examples will include
        # - TFIDF
        # - GPT
        # - External Feature Stores
        # - etc.

        baseline = self._get_subset(ref)
        prod = self._get_subset(cmp)
        combo = baseline.concatenate(prod)

        ### We create failure clusters, then we score them with drift amount.
        ### These are communicated to the data scientist as "Errors".

        fg_run: GroupResultsCollection = self.find_failure_groups(
            subset=combo,
            method="superlevel",
            model=model,
            embedding=embedding,
            min_size=min_size,
            min_failures=min_failures,
            failure_metric=failure_metric,
            config=config,
            run_name=run_name,
            manual=manual,
            visible=visible,
        )

        if fg_run:
            fgs = fg_run.groups
            error_groups_table = fg_run.summary(production_subset=prod)
        else:
            fgs = []
            error_groups_table = []

        ### Now we compute drifted groups, score each by error.
        ### And present these to the customer as "warnings".

        drift_run = self.find_drifted_groups(
            reference_group=baseline,
            comparison_group=prod,
            embedding=embedding,
            config=config,
            min_size=min_size,
        )
        if drift_run:
            drift_groups = drift_run.groups
            warning_groups_table = drift_run.summary(production_subset=prod)
        else:
            drift_groups = []
            warning_groups_table = []

        return {
            "summaries": (error_groups_table, warning_groups_table),
            "groups": (fgs, drift_groups),
        }

    def add_column(
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
        self.state.dataset.set_column(key, data, is_categorical)
        self.state.dataset_event_bus.run_all()
