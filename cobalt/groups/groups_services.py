# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

from mapper.nerve import partition_vec_to_cover


def get_autogroupings(
    max_group_count: int,
    multires_graph,
):
    graph = multires_graph

    hierarchical_clustering = graph.hierarchical_partition
    best_partition = hierarchical_clustering.best_modularity_partition(
        max_clusters=max_group_count
    )

    groups = partition_vec_to_cover(best_partition)

    return groups
