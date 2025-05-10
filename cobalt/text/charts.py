from typing import List, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mapper import DisjointPartitionGraph

from cobalt import CobaltDataset
from cobalt.text.ctfidf import CTFIDFKeywordAnalysis


def to_chart_data(labels_frame: pd.DataFrame, graph_subsets: List[List[int]]):
    """Given keyword dataframe and data point subsets, compute labels & frequencies."""
    subset_lengths = [len(indices) for indices in graph_subsets]
    keyword_names = labels_frame.agg(", ".join, axis=1).tolist()
    return keyword_names, subset_lengths


def generate_chart_data(
    graph: DisjointPartitionGraph,
    dataset: CobaltDataset,
    text_column_name: str,
    n_keywords=3,
):
    keyword_analyzer = CTFIDFKeywordAnalysis(dataset, text_column_name)
    result_selection, _, _ = keyword_analyzer.get_keywords_dataframe(
        subsets=[dataset.subset(indices) for indices in graph.nodes],
        n_keywords=n_keywords,
    )

    return to_chart_data(result_selection, graph.nodes)


def create_chart(
    keywords,
    frequencies,
    max_keywords: int = 10,
    chart_type: Literal["pie", "bar"] = "pie",
):
    heights = frequencies
    labels = keywords
    indices = np.argsort(heights)[::-1]
    num_slices = max_keywords

    other_amount = np.sum(heights) - np.array(heights)[indices[:num_slices]].sum()
    heights = list(np.array(heights)[indices[:num_slices]])
    heights.append(other_amount)

    labels = list(np.array(labels)[indices[:num_slices]])
    labels.append("Other")

    if chart_type == "pie":
        plt.pie(heights, labels=labels)
    else:
        plt.bar(labels, heights)
        plt.xticks(rotation=90)
