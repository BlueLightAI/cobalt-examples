from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from mapper import HierarchicalPartitionGraph

from cobalt import CobaltDataset, Workspace
from cobalt.embedding_models import SentenceTransformerEmbeddingModel
from cobalt.graph_utils import aggregate_scores_multi
from cobalt.lab.neighbors import generate_cluster_df_at_levels
from cobalt.text.ctfidf import top_keywords_per_level_per_subset


# TODO: Make this configurable.
def percentile_95(a: np.ndarray):
    """Return 95th percentile of array."""
    return np.percentile(a, 95)


def percentile_5(a: np.ndarray):
    """Return 5th percentile of array."""
    return np.percentile(a, 5)


def mask_count_less(a: np.ndarray):
    """Return count of elements less than or equal to a certain threshold."""
    threshold = 0
    return (a <= threshold).sum()


def mask_count_greater(a: np.ndarray):
    """Return count of elements greater than or equal to a certain threshold."""
    threshold = 0
    return (a >= threshold).sum()


def generate_keyword_summary_table(
    ds: CobaltDataset,
    text_column_name: str,
    n_gram_range: Union[str, Tuple],
    min_level: int = 0,
    max_level: Optional[int] = None,
    max_keywords: int = 3,
    max_neighbors: int = 3,
) -> Tuple[pd.DataFrame, Workspace, Dict[int, Dict[int, str]]]:
    """Returns a summary of groups in a set of texts.

    This builds a multiresolution graph from the embeddings provided in the
    input dataset, and for a range of coarseness levels, computes a keyword
    description of the text contained in each node, and returns this information
    in a DataFrame.

    Optionally also returns a Workspace object that can be used to access the
    graph and explore the results further.

    Args:
        ds (CobaltDataset): Dataset (containing an embedding of the text data)
        text_column_name (str): Column containing text data for keyword analysis.
        n_gram_range (Union[str, Tuple]): Whether to analyze keywords with
            unigrams, bigrams, or a combination.
        min_level (int): Minumum graph level to output cluster labels for.
        max_level (int): Maximum graph level to output cluster labels for.
        max_keywords (int): Maximum number of keywords to find for each cluster.
        max_neighbors (int): Maximum number of neighbors to return in table.
        return_intermediates (bool): Whether to return intermediate results.

    Returns:
        A tuple consisting of a pd.DataFrame per level with the labels for each cluster,
        a Workspace object and the raw labels per level per node.


    """
    w = Workspace(ds)
    g = w.new_graph()

    labels = top_keywords_per_level_per_subset(
        ds, g, text_column_name, n_keywords=max_keywords, n_gram_range=n_gram_range
    )

    if max_level is None:
        max_level = len(g.levels)

    summary_df = generate_cluster_df_at_levels(
        g, labels, (min_level, max_level), max_neighbors=max_neighbors
    )

    return (summary_df, w, labels)


def _concatenate_results(aggregated_scores_by_level: List[List[Dict[str, float]]]):
    dfs = [
        pd.DataFrame.from_records(level_summary)
        for level_summary in aggregated_scores_by_level
    ]

    # Concatenate Vertically
    scores_df = pd.concat(dfs).reset_index(drop=True)
    return scores_df


def column_namer_just_name(col: str, meth: str):
    return col


def column_namer_name_with_aggregation_method(col: str, meth: str):
    return f"{col}_{meth}"


def score_all_hierarchical_graph_subsets(
    graph: HierarchicalPartitionGraph,
    ds: CobaltDataset,
    aggregation_columns: List[str],
    aggregation_methods: List[Callable],
    min_level: int,
    max_level: int,
):
    """Return aggregated scores for every subset at at every level of the graph.

    Args:
        graph (HierarchicalPartitionGraph): graph
        ds (CobaltDataset): dataset
        aggregation_columns (List[str]): columns from within `ds` to aggregate.
        aggregation_methods (List[Callable]): methods to aggregate columns by.
        min_level (int): Minimum Level to compute scores at.
        max_level (int): Maximum level to compute scores at.

    Returns:
        df (pd.DataFrame): Aggregated scores dataframe.

    """
    method = None

    # Custom Logic: If len(aggregation_methods) == 1,
    # then don't include what method it is in the dataframe.

    # This is for backward compatibility for certain externally released
    # notebooks.
    if len(aggregation_methods) == 1:
        method = column_namer_just_name
    else:
        method = column_namer_name_with_aggregation_method

    aggregated_scores_by_level = [
        aggregate_scores_multi(
            graph.levels[i].nodes, ds, aggregation_columns, aggregation_methods, method
        )
        for i in range(min_level, max_level)
    ]

    return _concatenate_results(aggregated_scores_by_level)


def describe_groups_multiresolution(
    ds: CobaltDataset,
    text_column_name: str,
    n_gram_range: Union[str, Tuple],
    aggregation_columns: Optional[List[str]] = None,
    min_level: int = 0,
    max_level: Optional[int] = None,
    max_keywords: int = 3,
    aggregation_method: Union[Literal["all", "mean"], List[Callable]] = "mean",
    return_intermediates: bool = False,
) -> Tuple[pd.DataFrame, Workspace, Dict[int, Dict[int, str]]]:
    """Returns a summary of groups in a set of texts.

    This builds a multiresolution graph from the embeddings provided in the
    input dataset, and for a range of coarseness levels, computes a keyword
    description of the text contained in each node, and returns this information
    in a DataFrame.

    Optionally also returns a Workspace object that can be used to access the
    graph and explore the results further.

    Args:
        ds (CobaltDataset): Dataset (containing an embedding of the text data)
        text_column_name (str): Column containing text data for keyword analysis.
        n_gram_range (Union[str, Tuple]): Whether to analyze keywords with
            unigrams, bigrams, or a combination.
        aggregation_columns: Columns in ds to aggregate.
        min_level (int): Minumum graph level to output cluster labels for.
        max_level (int): Maximum graph level to output cluster labels for.
        max_keywords (int): Maximum number of keywords to find for each cluster.
        max_neighbors (int): Maximum number of neighbors to return in table.
        aggregation_method: Method(s) to aggregate columns by.
        return_intermediates (bool): Whether to return intermediate results.

    Returns:
        A tuple consisting of a pd.DataFrame per level with the labels for each cluster,
        a Workspace object and the raw labels per level per node.


    """
    if aggregation_columns is None:
        aggregation_columns = ds.df.select_dtypes(include=np.number).columns.tolist()

    max_neighbors = 3
    summary_df, w, labels = generate_keyword_summary_table(
        ds,
        text_column_name,
        n_gram_range,
        min_level,
        max_level,
        max_keywords,
        max_neighbors=max_neighbors,
    )

    graph = w.graphs["New Graph"]
    if max_level is None:
        max_level = len(graph.levels)

    assert not summary_df.level.isna().any()

    if isinstance(aggregation_method, str):
        if aggregation_method == "mean":
            aggregation_methods = [np.mean]
        elif aggregation_method == "all":
            aggregation_methods: List[Callable] = [
                np.mean,
                np.std,
                np.min,
                np.max,
                np.median,
                percentile_95,
                percentile_5,
                mask_count_less,
                mask_count_greater,
            ]
        else:
            raise ValueError("Invalid Aggregation Method passed in.")
    else:
        aggregation_methods = aggregation_method

    if isinstance(aggregation_columns, str):
        aggregation_columns = [aggregation_columns]

    # Really: we want a organizational structure like the following:
    # The user may do something like
    # `w.generate_labels_by(text_column: str)
    #   .to_dataframe()`

    # `generate_labeled_clusters_by` can return something like
    #  an `AnnotatedHierarchicalGraph`.

    # then convert to a `pd.DataFrame` with `to_dataframe`.

    # to generate summary_df: `pd.DataFrame`.

    # That's the proposed flow to generate an annotated dataframe.
    # Now given that annotated dataframe which underlyingly includes groups,
    # we can run an aggregate scores method.

    # `w.graph_groups.aggregate_scores(
    #   aggregation_columns,
    #   aggregation_methods
    # )` to return a scores `pd.DataFrame`.

    scores_df = score_all_hierarchical_graph_subsets(
        graph, ds, aggregation_columns, aggregation_methods, min_level, max_level
    )

    # What information does the user pass in to compute best_model
    # for each subgroup?

    # Say `aggregation_columns` is `[model_1,  ..., model_50]`
    # Naive:
    # find_best = True is another parameter
    # if find_best is True, then compute argmax over aggregation_columns
    # and write the best model for each group in a new column.

    # Then shift to the left.

    # Post Processing Step
    # Take a Results Dataframe + a set of columns
    # Choose the best model for each

    assert len(scores_df) == len(summary_df)
    assert not summary_df.level.isna().any()
    assert "level_0" not in set(summary_df.columns)

    # Concatenate keyword and group dataframes horizontally.
    summary_df = pd.concat([summary_df, scores_df], axis=1)

    assert not summary_df.level.isna().any()

    no_neighbors = True

    if no_neighbors:
        neighbor_columns = [f"Neighbor Label {i}" for i in range(max_neighbors)]
        summary_df = summary_df.drop(neighbor_columns, axis=1)
    summary_df = summary_df.rename(columns={"Node Size": "query_count"})

    # The (w, labels) are useful for visualization.
    if return_intermediates:
        return summary_df, w, labels
    return summary_df


def compute_arg_maximal_score(
    results: pd.DataFrame, score_columns: List[str]
) -> pd.Series:
    """Return maximal column over a list of columns given a dataframe.

    Args:
        results (pd.DataFrame): Results Dataframe.
        score_columns (List[str]): Columns in results to compare over.

    Returns:
        out (pd.Series): Columns with the best performance for each row in results.

    Notes for post-processing
    Columns: "Label", "query_count", "level", "a_mean",  ...
    Take dataframe + ['a_mean', 'b_mean', 'c_mean', 'd_mean']
    Returns a pd.Series that corresponds to the best model
    for each group.
    """
    cols = results[score_columns]

    # TODO: What about duplicates?
    indxs = cols.idxmax(axis=1)

    return indxs


def raw_group_description_multiresolution(
    texts: List[str],
    numerical_data: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    device: str = "mps",
    sentence_transformer_model_id: str = "all-MiniLM-L6-v2",
    min_level: int = 0,
    max_level: Optional[int] = None,
    max_keywords: int = 3,
    return_intermediates: bool = False,
) -> Tuple[pd.DataFrame, Workspace, Dict[int, Dict[int, str]]]:
    """Returns summary of groups in a set of texts, constructing a workspace along the way.

    Args:
        texts (List[str]): the texts to embed and analyze
        numerical_data : scoring column(s) as np.ndarray or dataframe.
        device (str): the device to run the embedding computation on
        sentence_transformer_model_id (str): SBERT ID
        min_level (int): Minumum Graph Level to output cluster labels for.
        max_level (int): Maximum Graph Level to output cluster labels for.
        max_keywords (int): Max Keywords to find in TFIDF algorithm.
        return_intermediates (bool): whether to return intermediate results or not.

    Returns:
        tuple (Tuple): consisting of a pd.DataFrame per level, the labels for each cluster.
            a workspace object and the raw labels per level per node.

    Notes:
        `numerical_data` really could be a dataframe but since all it needs to be right now
          is a np.ndarray,
        I implemented it as simply as possible. Lower-level, there is capability for it to
        use different aggregation methods. But that's not really necessary right now.

        The device is defaulted to be `mps` because that's what we're using internally and I didn't
        want to break the order of the API.

    """
    df = pd.DataFrame({"text": texts})

    if numerical_data is not None:
        if isinstance(numerical_data, pd.DataFrame):
            df = pd.concat([df, numerical_data], axis=1)
        else:
            df["score"] = numerical_data

    ds = CobaltDataset(df)
    m = SentenceTransformerEmbeddingModel(sentence_transformer_model_id)
    embedding = m.embed(texts, device=device)
    ds.add_embedding_array(embedding, metric="cosine", name="sbert")
    return describe_groups_multiresolution(
        ds,
        "text",
        "unigrams",
        aggregation_columns=None,
        min_level=min_level,
        max_level=max_level,
        max_keywords=max_keywords,
        return_intermediates=return_intermediates,
    )


def get_interpretable_groups(
    ds: CobaltDataset,
    text_column_name: str,
    n_gram_range: Union[str, Tuple],
    aggregation_columns: Optional[List[str]] = None,
    min_level: int = 0,
    max_level: Optional[int] = None,
    max_keywords: int = 3,
    aggregation_method: Union[Literal["all", "mean"], List[Callable]] = "mean",
    return_intermediates: bool = False,
) -> Tuple[pd.DataFrame, Workspace, Dict[int, Dict[int, str]]]:
    """Returns summary of groups in a set of texts, constructing a workspace along the way.

    Args:
        ds (CobaltDataset): Dataset (with one embedding)
        text_column_name (str): COlumn containing text to construct keywords out of.
        aggregation_columns: Columns in ds to aggregate.
        n_gram_range (Union[str, Tuple]): Whether to do unigrams, bigrams, or a combination.
        min_level (int): Minumum Graph Level to output cluster labels for.
        max_level (int): Maximum Graph Level to output cluster labels for.
        max_keywords (int): Max Keywords to find in TFIDF algorithm.
        aggregation_method: Method(s) to aggregate columns by.
        return_intermediates (bool): whether to return intermediate results or not.

    Returns:
        tuple (Tuple): consisting of a pd.DataFrame per level, the labels for each cluster.
            a workspace object and the raw labels per level per node.


    """
    # This function name is a compatibility shim to support external-facing notebook.
    return describe_groups_multiresolution(
        ds,
        text_column_name,
        n_gram_range,
        aggregation_columns,
        min_level,
        max_level,
        max_keywords,
        aggregation_method,
        return_intermediates,
    )
