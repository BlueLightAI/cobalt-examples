# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
from mapper import HierarchicalPartitionGraph
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import normalize

from cobalt.array_utils import (
    get_sorted_max_row_values_and_indices,
    right_pad_array,
    sum_sparse_array_rows,
)
from cobalt.schema.dataset import CobaltDataset, CobaltDataSubset


class CTFIDFKeywordAnalysis:
    def __init__(
        self,
        dataset: CobaltDataset,
        text_column: str,
        n_gram_range: Union[
            Literal["up_to_bigrams", "unigrams", "bigrams"], Tuple[int, int]
        ] = "up_to_bigrams",
        stop_words: Optional[Union[str, List[str]]] = "english",
    ):
        self.dataset = dataset
        self.text_column = text_column
        if isinstance(n_gram_range, str):
            if n_gram_range == "unigrams":
                n_gram_range = (1, 1)
            elif n_gram_range == "up_to_bigrams":
                n_gram_range = (1, 2)
            elif n_gram_range == "bigrams":
                n_gram_range = (2, 2)
        self.n_gram_range = n_gram_range
        self.stop_words = stop_words
        self.count_vectorizer = CountVectorizer(
            ngram_range=self.n_gram_range, stop_words=self.stop_words
        )
        self._count_matrix: Optional[sp.csr_array] = None

    def _get_count_matrix(self) -> sp.csr_array:
        text = self.dataset.select_col(self.text_column)
        return sp.csr_array(self.count_vectorizer.fit_transform(text.fillna("")))

    @property
    def term_count_matrix(self) -> sp.spmatrix:
        if self._count_matrix is None:
            self._count_matrix = self._get_count_matrix()
        return self._count_matrix

    def aggregated_term_count_matrix(
        self, subsets: List[CobaltDataSubset]
    ) -> sp.spmatrix:
        indices = [subset.indices for subset in subsets]
        return sp.csr_matrix(sum_sparse_array_rows(self.term_count_matrix, indices))

    def get_keywords(
        self, subsets: List[CobaltDataSubset], n_keywords: int = 3
    ) -> Tuple[List[Sequence[str]], List[Sequence[float]], List[Sequence[float]]]:
        """Find keywords/terms for each subset.

        The keywords will be selected from self.text_column.

        Returns:
            keywords: A list of lists; the list in entry i of the outer list
                contains up to n_keywords keywords for subset i.
            scores: A list of arrays; the array in entry i of the outer list
                contains scores for the corresponding keywords.
            match_rates: A list of arrays; the array in entry i of the outer list
                provides the fraction of documents in subset i that contain each
                keyword.
        """
        subset_term_counts = self.aggregated_term_count_matrix(subsets)
        ctfidf_scores = (
            CTFIDFVectorizer().fit(subset_term_counts).transform(subset_term_counts)
        )

        words = self.count_vectorizer.get_feature_names_out()
        terms_per_subset = []
        scores_per_subset = []
        keyword_match_rates = []
        for i, subset in enumerate(subsets):
            top_values_sorted, sorted_term_ids = get_sorted_max_row_values_and_indices(
                ctfidf_scores, i, n_keywords
            )
            top_words: List[str] = [words[index] for index in sorted_term_ids]
            per_doc_term_counts = self.term_count_matrix[subset.indices, :][
                :, sorted_term_ids
            ].todense()
            frac_docs_with_keywords = np.mean((per_doc_term_counts > 0), axis=0)
            # if there aren't enough keywords, fill the remaining slots with empty strings/0
            terms_per_subset.append(top_words)
            scores_per_subset.append(top_values_sorted)
            keyword_match_rates.append(frac_docs_with_keywords)
        # TODO: best output format? list of lists of dicts? list of dataframes? 3d array?
        return (terms_per_subset, scores_per_subset, keyword_match_rates)

    def get_keywords_dataframe(
        self, subsets: List[CobaltDataSubset], n_keywords: int = 3
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        keywords, scores, match_rates = self.get_keywords(subsets, n_keywords)
        # ensure that all keyword lists are the same length
        keywords = [
            [*kws, *("" for _ in range(n_keywords - len(kws)))] for kws in keywords
        ]
        scores = [right_pad_array(sc, n_keywords, 0) for sc in scores]
        match_rates = [right_pad_array(mr, n_keywords, 0) for mr in match_rates]
        keywords_df = pd.DataFrame(
            keywords,
            columns=[f"Top Keyword {i}" for i in range(n_keywords)],
        )
        keywords_df.index.name = "Subset Index"
        scores_per_node_df = pd.DataFrame(
            scores,
            columns=[f"Top Keyword {i}" for i in range(n_keywords)],
        )
        scores_per_node_df.index.name = "Subset Index"
        match_rate_per_node_df = pd.DataFrame(
            match_rates,
            columns=[f"Top Keyword {i}" for i in range(n_keywords)],
        )
        match_rate_per_node_df.index.name = "Subset Index"
        return keywords_df, scores_per_node_df, match_rate_per_node_df


class CTFIDFVectorizer(TfidfTransformer):
    """This class based interpretation of TF-IDF is from BERTopic."""

    def __init__(self, *args, **kwargs):
        super(CTFIDFVectorizer).__init__(*args, **kwargs)

    def fit(self, X: sp.csr_matrix):
        """Learn the idf vector (global term weights)."""
        _, n_features = X.shape
        df = np.squeeze(np.asarray(X.sum(axis=0)))
        avg_nr_samples = int(X.sum(axis=1).mean())
        idf = np.log((avg_nr_samples / df) + 1)
        self._idf_diag = sp.diags(
            idf,
            offsets=0,
            shape=(n_features, n_features),
            format="csr",
            dtype=np.float64,
        )
        return self

    def transform(self, X: sp.csr_matrix) -> sp.csr_matrix:
        """Transform a count-based matrix to c-TF-IDF."""
        X = X * self._idf_diag
        X = normalize(X, axis=1, norm="l1", copy=False)
        return X


class CTFIDF:

    @staticmethod
    def tfidf_on_text_clusters(
        list_text_clusters: List[str],
        top_n: int,
        n_gram_range: Union[
            Literal["up_to_bigrams", "unigrams", "bigrams"], Tuple[int, int]
        ],
        stop_words: Optional[Union[str, List[str]]] = "english",
    ) -> Dict:
        """Returns the most frequent terms for each element in `list_text_clusters`.

        Args:
            list_text_clusters (List[str]): Each element is a string.
            top_n (int): The number of terms to return per string.
            stop_words: The stop words to pass into the CountVectorizer.
            n_gram_range: Can specify unigrams or bigrams or up_to_bigrams.

        Notes:
            TF-IDF is a classical algorithm, here we used C-TFIDF
            for because each node is concatenating into one single
            "document"/"class" so that TF-IDF can then be run.

            Inspired by https://maartengr.github.io/BERTopic/algorithm/algorithm.html and
            https://www.maartengrootendorst.com/blog/ctfidf/

        """
        if isinstance(n_gram_range, str):
            if n_gram_range == "unigrams":
                n_gram_range = (1, 1)
            elif n_gram_range == "up_to_bigrams":
                n_gram_range = (1, 2)
            elif n_gram_range == "bigrams":
                n_gram_range = (2, 2)

        count_vectorizer = CountVectorizer(
            stop_words=stop_words, ngram_range=n_gram_range
        ).fit(list_text_clusters)
        count = count_vectorizer.transform(list_text_clusters)
        words = count_vectorizer.get_feature_names_out()

        ctfidf = CTFIDFVectorizer().fit_transform(count).toarray()

        words_per_node = {}
        scores_per_node = {}
        for i in range(len(list_text_clusters)):
            partially_sorted_indices = np.argpartition(ctfidf[i], -top_n)[-top_n:]

            top_values = ctfidf[i][partially_sorted_indices]

            sorted_indices = partially_sorted_indices[np.argsort(top_values)[::-1]]
            top_values_sorted = ctfidf[i][sorted_indices]
            top_words = [words[index] for index in sorted_indices]
            words_per_node[i] = top_words
            scores_per_node[i] = top_values_sorted
        return (words_per_node, scores_per_node)

    @staticmethod
    def extract_and_concatenate_text(
        subsets: List[CobaltDataSubset],
        text_column_name: str,
    ) -> List[str]:
        """Returns concatenated texts along `text_column_name` column per subset in `subsets`.

        Args:
            subsets (List[CobaltDataSubset]): subsets or clusters to extract texts from.
            text_column_name (str): column in each subset that pertains to text column.

        Returns:
            texts (List[str]): \
                where `texts[i]` contains the documents in `subsets[i].df[text_column_name]`,\
                      joined together by spaces.

        Notes:
            This is central to the use of "Class" TF-IDF (C-TFIDF) from the BERTopic library/paper.
        """
        texts = []
        for subset in subsets:
            texts.append(subset.df[text_column_name].str.cat(sep=" "))
        return texts

    @staticmethod
    def get_top_keywords(
        texts: List[str],
        n_gram_range: Union[
            Literal["up_to_bigrams", "unigrams", "bigrams"], Tuple[int, int]
        ],
        top_n_keywords: int = 3,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Return most frequent keywords per string, based on CTFIDF keyword analysis.

        Args:
            texts (List[str]): Texts, can represent many data points.
            top_n_keywords (int): Number of Keywords to return.
            n_gram_range (str): Can specify unigrams or bigrams or up_to_bigrams.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: A dataframe of keywords as columns per
            CobaltDataSubset row, and a matching dataframe of the ctfidf scores per cell.
        """
        text_to_keywords, scores_per_node = CTFIDF.tfidf_on_text_clusters(
            texts, top_n=top_n_keywords, n_gram_range=n_gram_range
        )
        keywords_df = pd.DataFrame.from_dict(
            text_to_keywords,
            orient="index",
            columns=[f"Top Keyword {i}" for i in range(top_n_keywords)],
        )
        keywords_df.index.name = "Node Index"
        scores_per_node_df = pd.DataFrame.from_dict(
            scores_per_node,
            orient="index",
            columns=[f"Top Keyword {i}" for i in range(top_n_keywords)],
        )
        scores_per_node_df.index.name = "Node Index"
        return (keywords_df, scores_per_node_df)

    @staticmethod
    def get_labels(
        subsets: List[CobaltDataSubset],
        text_column_name: str,
        n_gram_range: Union[
            Literal["up_to_bigrams", "unigrams", "bigrams"], Tuple[int, int]
        ],
        top_n_keywords: int = 3,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """This runs CTFIDF on a list of subsets.

        One usage is from a Mapper graph of clusters extracted
        from a single resolution level in the Hierarchical Partition Graph.

        Args:
            subsets: Contain Nodes that have text.
            text_column_name: str: Dataframe text column name.
            top_n_keywords: int = 3: Number of ctfidf keywords to compute.
            n_gram_range (str): Can specify unigrams or bigrams or up_to_bigrams.

        Returns:
            top_keywords: A dataframe with one row per subset,
                whose rows contain the top n keywords from each subset.
            ctfidf_scores: A dataframe of the same shape containing the
                corresponding ctfidf scores for each keyword.
        """
        texts = CTFIDF.extract_and_concatenate_text(subsets, text_column_name)
        labelled_topics_df, scores_per_node_df = CTFIDF.get_top_keywords(
            texts, top_n_keywords=top_n_keywords, n_gram_range=n_gram_range
        )
        return (labelled_topics_df, scores_per_node_df)


def top_keywords_per_level_per_subset(
    dataset: CobaltDataset,
    hierarchical_graph: HierarchicalPartitionGraph,
    text_column_name: str,
    n_gram_range: Union[
        Literal["up_to_bigrams", "unigrams", "bigrams"], Tuple[int, int]
    ],
    n_keywords: int = 3,
) -> Dict[int, Dict[int, str]]:
    """Computes node labels per level in the Multi Resolution Graph.

    Args:
        dataset (CobaltDataset): The dataset upon which to extract text.
        hierarchical_graph (HierarchicalPartitionGraph): the multi-level graph.
        text_column_name (str): The text column to extract from.
        n_keywords (int): Number of Keywords to include per node.
        n_gram_range (str): Can specify unigrams or bigrams or up_to_bigrams.

    Returns:
        labels (Dict[int, Dict[int, str]]): Nested Dictionary keyed by level,
          then node_id of node labels.

    """
    level_map = {}

    keyword_generator = CTFIDFKeywordAnalysis(dataset, text_column_name, n_gram_range)
    for level in range(len(hierarchical_graph.levels) - 1, -1, -1):
        single_graph = hierarchical_graph.levels[level]
        subsets = [dataset.subset(indices) for indices in single_graph.nodes]
        result_selection, *_ = keyword_generator.get_keywords_dataframe(
            subsets, n_keywords
        )
        keyword_names = result_selection.agg(
            lambda terms: ", ".join(t for t in terms if t), axis=1
        ).tolist()
        node_labels = dict(enumerate(keyword_names))
        level_map[level] = node_labels
    return level_map


def top_keywords_per_level_per_datapoint(
    dataset: CobaltDataset,
    hg: HierarchicalPartitionGraph,
    text_column_name: str,
    n_gram_range: Union[
        Literal["up_to_bigrams", "unigrams", "bigrams"], Tuple[int, int]
    ],
    top_n_keywords: int = 3,
) -> Dict[int, List[str]]:
    """Computes the Top Keywords Per Level Per Datapoint.

    Args:
        dataset (CobaltDataset): dataset
        hg (HierarchicalPartitionGraph): hierarchical graph with levels
        text_column_name (str): column in dataset to extract keywords from.
        top_n_keywords (int): number of keywords
        n_gram_range (str): Can specify unigrams or bigrams or up_to_bigrams.

    Returns:
        datapoint_labels (Dict[int, List[str]]): labels per data point at each level.

    """
    level_map = {}

    keyword_generator = CTFIDFKeywordAnalysis(dataset, text_column_name, n_gram_range)
    for level in range(len(hg.levels) - 1, -1, -1):
        single_graph = hg.levels[level]

        result_selection, *_ = keyword_generator.get_keywords_dataframe(
            [dataset.subset(indices) for indices in single_graph.nodes], top_n_keywords
        )

        keyword_names = result_selection.agg(
            lambda terms: ", ".join(t for t in terms if t), axis=1
        ).tolist()
        node_labels = dict(enumerate(keyword_names))
        dp_nodes = single_graph.node_membership
        topic_names = [node_labels[node_id] for node_id in dp_nodes]
        level_map[level] = topic_names
    return level_map
