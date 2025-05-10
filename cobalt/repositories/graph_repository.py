# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict

from cobalt.multires_graph import MultiResolutionLandscape
from cobalt.schema.dataset import CobaltDataSubset


@dataclass
class GraphRepositoryEntry:
    subset: CobaltDataSubset
    graph: MultiResolutionLandscape


class AbstractGraphRepository:
    @abstractmethod
    def add_graph(self, graph: dict, graph_name: str):
        raise NotImplementedError

    @abstractmethod
    def update_graph(self, graph, graph_name):
        raise NotImplementedError

    @abstractmethod
    def get_graph(self, graph_name):
        raise NotImplementedError

    @abstractmethod
    def get_graphs(self):
        raise NotImplementedError

    @abstractmethod
    def get_graphs_with_source_data(self, source_data: CobaltDataSubset):
        raise NotImplementedError


class GraphRepository(AbstractGraphRepository):
    def __init__(self) -> None:
        self.graphs: Dict[str, GraphRepositoryEntry] = {}

    def add_graph(self, graph: dict, graph_name: str):
        self.graphs[graph_name] = GraphRepositoryEntry(**graph)

    def update_graph(self, graph: Dict, graph_name: str):
        self.graphs[graph_name] = GraphRepositoryEntry(**graph)

    def get_graph(self, graph_name) -> Dict:
        graph = self.graphs[graph_name]
        return {"subset": graph.subset, "graph": graph.graph}

    def get_graphs(self) -> Dict[str, Dict]:
        return {name: self.get_graph(name) for name in self.graphs}

    def get_graphs_with_source_data(self, source_data: CobaltDataSubset):
        return {
            name: self.get_graph(name)
            for name in self.graphs
            if self.graphs[name].subset == source_data
        }
