from collections import Counter
from itertools import chain
import os
import random
from typing import (
    Any,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
    Union,
)

import numpy as np
from transformers import (
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
    AddedToken,
)

from xlm.datamodule import SimpleSpaceTokenizer
from xlm.utils.rank_zero import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)
ranked_logger = RankedLogger(__name__, rank_zero_only=False)

########################################################
# region: Types


class Graph(TypedDict):
    edge_list: List[Tuple[int, int]]
    source: int
    goal: int
    path: List[int]


class GraphBatch(TypedDict):
    edge_list: List[List[Tuple[int, int]]]
    source: List[int]
    goal: List[int]
    path: List[List[int]]


class GraphExample(TypedDict):
    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: List[int]


# endregion: Types
########################################################

########################################################
# region: Graph Dataset Generation


def simple_star_graph(  # noqa: C901
    degree: int = 3, pathlength: int = 5, vocab_size: int = 20
) -> Graph:
    """
    Generate a asymmetric star graph.


    # Adapted from : https://github.com/gregorbachmann/Next-Token-Failures/blob/main/data/graphs.py

    Properties:
        - The source is always the center of the star.
        - The goal is always the end of one of the arms of the star.
        - No nodes are repeated in the graph.
            So pathlength * degree + 1  should be greater than or equal to vocab_size otherwise the
            code will loop indefinitely.

    Args:
        degree: int, number of edges from source (center of the start) to other nodes
        pathlength: int, length of the path (in terms of number of nodes)
        vocab_size: int, number of nodes in the graph
    Returns:
        path: list, path from source to goal
        edge_list: list, list of edges in the graph
    """
    if vocab_size < pathlength * degree + 1:
        raise ValueError(
            "vocab_size must be greater than or equal to pathlength * degree + 1"
        )
    source = np.random.randint(0, vocab_size, 1)[0]
    goal = np.random.randint(0, vocab_size, 1)[0]
    while goal == source:
        goal = np.random.randint(0, vocab_size, 1)[0]

    path = [source]
    edge_list = []

    # Choose random nodes along the path
    for _ in range(pathlength - 2):
        node = np.random.randint(0, vocab_size, 1)[0]
        while node in path or node == goal:  # should not be in path or goal
            node = np.random.randint(0, vocab_size, 1)[0]
        path.append(node)

    path.append(goal)
    # Connect the path
    for i in range(len(path) - 1):
        edge_list.append([path[i], path[i + 1]])

    remaining_nodes = []
    for i in range(vocab_size):
        if i not in path:
            remaining_nodes.append(i)

    i = 0
    deg_nodes = set()
    while i < degree - 1:
        # Add neighbour to source
        node = source
        next_node = np.random.randint(0, vocab_size, 1)[0]
        l = 1
        while l < pathlength:
            if next_node not in deg_nodes and next_node not in path:
                edge_list.append([node, next_node])
                deg_nodes.add(next_node)
                node = next_node
                l += 1
            next_node = np.random.randint(0, vocab_size, 1)[0]

        i += 1

    random.shuffle(edge_list)
    # return path, edge_list, source, goal
    return {
        "path": path,
        "edge_list": edge_list,
        "source": source,
        "goal": goal,
    }


def asymmetric_variable_armlength_star_graph(
    degree: int = 3,
    min_pathlength: int = 3,
    max_pathlength: int = 6,
    vocab_size: int = 20,
    min_plan_length: int = 2,
) -> Graph:
    # generate 'degree' number of paths
    if vocab_size < max_pathlength * degree - degree + 1:
        raise ValueError(
            f"vocab_size must be greater than or equal to max_pathlength * degree - degree + 1: {vocab_size} < ({max_pathlength}) * ({degree}) - {degree} + 1"
        )

    if min_plan_length > min_pathlength:
        raise ValueError(
            "min_plan_length must be less than or equal to min_pathlength"
        )

    path_lengths = np.random.randint(
        min_pathlength, max_pathlength + 1, degree
    )
    # sample join locations, one per path
    join_locations = np.full(degree, -1)
    join_locations[0] = np.random.randint(path_lengths[0] - min_plan_length)
    for i in range(1, degree):
        join_locations[i] = np.random.randint(
            path_lengths[i] - min_plan_length
        )

    # generate unique vocab id for each node
    num_nodes = sum(path_lengths)
    unique_node_ids = np.random.choice(
        vocab_size, num_nodes - degree + 1, replace=False
    )
    center_node = unique_node_ids[0]
    edge_list = []
    j = 1
    # CLEANUP: Remove debug prints
    # print(f"len(unique_node_ids): {len(unique_node_ids)}")
    # print(f"path_lengths: {path_lengths}")
    # print(f"join_locations: {join_locations}")
    for path_num in range(degree):
        prev_node = None
        for node_num in range(path_lengths[path_num]):
            if prev_node is None:
                if node_num == join_locations[path_num]:
                    prev_node = center_node
                    continue
                prev_node = unique_node_ids[j]
                j += 1
                continue
            assert prev_node is not None
            if node_num == join_locations[path_num]:
                # print(
                #    f"path_num: {path_num}, node_num: {node_num}, join_locations: {join_locations[path_num]}"
                # )
                # print(f"prev_node: {prev_node}, center_node: {center_node}")
                edge_list.append([prev_node, center_node])
                prev_node = center_node
                continue
            # print(
            #    f"path_num: {path_num}, node_num: {node_num}, prev_node: {prev_node}, unique_node_ids[j]: {unique_node_ids[j]}"
            # )
            # print(
            #    f"prev_node: {prev_node}, unique_node_ids[j]: {unique_node_ids[j]}"
            # )
            edge_list.append([prev_node, unique_node_ids[j]])
            prev_node = unique_node_ids[j]
            j += 1
    # read out source and goal
    source = edge_list[0][0]
    goal = edge_list[path_lengths[0] - 2][1]
    path = [edge_list[0][0]] + [
        edge_list[k][1] for k in range(path_lengths[0] - 1)
    ]
    random.shuffle(edge_list)
    return {
        "edge_list": edge_list,
        "source": source,
        "goal": goal,
        "path": path,
    }


def plot_graph(edge_list, path=None, source=None, goal=None):
    """

    Example usage:
    graph = simple_star_graph(4, 4, 20)
    plot_graph(graph["edge_list"], graph["path"], graph["source"], graph["goal"])
    """
    import matplotlib.pyplot as plt
    import networkx as nx

    G = nx.DiGraph()
    G.add_edges_from(edge_list)

    # pos = nx.spring_layout(G, k=2)  # positions for all nodes
    # pos = nx.nx_agraph.graphviz_layout(G, prog='dot', args='-Grankdir=TB')
    pos = nx.nx_agraph.graphviz_layout(G, prog="neato")
    # pos = nx.spectral_layout(G, scale=10)

    # Draw the nodes
    nx.draw_networkx_nodes(G, pos, node_size=500)

    # Draw the edges
    nx.draw_networkx_edges(G, pos, edgelist=edge_list, width=2)

    # Draw the labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")

    # Highlight the path if provided
    if path:
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(
            G, pos, edgelist=path_edges, width=2, edge_color="r"
        )

    # Highlight source and goal if provided
    if source is not None:
        nx.draw_networkx_nodes(
            G, pos, nodelist=[source], node_color="g", node_size=500
        )
    if goal is not None:
        nx.draw_networkx_nodes(
            G, pos, nodelist=[goal], node_color="b", node_size=500
        )

    plt.axis("off")  # Turn off the axis
    plt.show()


def identify_junction_node(edge_list: List[Tuple[int, int]]) -> int:
    node_counts = Counter(chain(*edge_list))
    junction = None
    for node, count in node_counts.items():
        if count > 2:
            if junction is None:
                junction = node
            elif junction is not None:
                raise ValueError(
                    f"Multiple junction nodes found: {junction} and {node}\n node_counts: {node_counts}"
                )
    if junction is None:
        raise ValueError("No junction node found")
    return junction  # type: ignore


class StarGraphGenerator:
    def __init__(self, count: int = 50000):
        self.count = count

    def graphs(
        self, count: Optional[int] = None
    ) -> Generator[Graph, None, None]:
        raise NotImplementedError

    def unique_str(self) -> str:
        raise NotImplementedError

    @property
    def max_pathlength(self) -> int:
        raise NotImplementedError

    @property
    def max_paths(self) -> int:
        raise NotImplementedError

    @property
    def max_num_edges(self) -> int:
        raise NotImplementedError


class SimpleStarGraphGenerator(StarGraphGenerator):
    def __init__(
        self,
        count: int = 50000,
        degree: int = 3,
        pathlength: int = 5,
        vocab_size: int = 20,
    ):
        super().__init__(count)
        self.degree = degree
        self.pathlength = pathlength
        self.vocab_size = vocab_size

    @property
    def max_pathlength(self) -> int:
        return self.pathlength

    @property
    def max_paths(self) -> int:
        return self.degree

    @property
    def max_num_edges(self) -> int:
        return self.degree * (self.pathlength - 1)

    def unique_str(self):
        return f"simple_star_graph_degree={self.degree}_pathlength={self.pathlength}_vocab_size={self.vocab_size}_count={self.count}"

    def graphs(
        self, count: Optional[int] = None
    ) -> Generator[Graph, None, None]:
        if count is None:
            count = self.count
        for _ in range(count):
            graph = simple_star_graph(
                self.degree, self.pathlength, self.vocab_size
            )
            yield graph


class AsymmetricVariableArmlengthStarGraphGenerator(StarGraphGenerator):
    def __init__(
        self,
        count: int = 50000,
        degree: int = 3,
        min_pathlength: int = 3,
        max_pathlength: int = 6,
        vocab_size: int = 20,
        min_plan_length: int = 2,
    ):
        super().__init__(count)
        self.degree = degree
        self.min_pathlength = min_pathlength
        self._max_pathlength = max_pathlength
        self.vocab_size = vocab_size
        self.min_plan_length = min_plan_length

    @property
    def max_pathlength(self) -> int:
        return self._max_pathlength

    @property
    def max_paths(self) -> int:
        return self.degree

    @property
    def max_num_edges(self) -> int:
        return self.degree * (self.max_pathlength - 1)

    def unique_str(self):
        return f"asymmetric_variable_armlength_star_graph_degree={self.degree}_min_pathlength={self.min_pathlength}_max_pathlength={self._max_pathlength}_vocab_size={self.vocab_size}_min_plan_length={self.min_plan_length}_count={self.count}"

    def graphs(
        self, count: Optional[int] = None
    ) -> Generator[Graph, None, None]:
        if count is None:
            count = self.count
        for _ in range(count):
            graph = asymmetric_variable_armlength_star_graph(
                self.degree,
                self.min_pathlength,
                self._max_pathlength,
                self.vocab_size,
                self.min_plan_length,
            )
            yield graph


# endregion: Graph Generation
########################################################


########################################################
# region: Tokenizer


# endregion: Tokenizer
########################################################


def preprocess_fn(
    example: Dict[str, Any], tokenizer: SimpleSpaceTokenizer
) -> Dict[str, Any]:
    edge_list = example["edge_list"]
    source = example["source"]
    goal = example["goal"]
    path = example["path"]
    path_edges: List[Tuple[int, int]] = list(zip(path, path[1:]))
    prompt_ids = [
        tokenizer._convert_token_to_id(str(ch))
        for ch in list(chain(*edge_list)) + [source, goal]
    ]
    input_ids = [
        tokenizer._convert_token_to_id(str(ch))
        for ch in list(chain(*path_edges))
    ]
    example["input_token_ids"] = input_ids
    example["prompt_token_ids"] = prompt_ids
    return example
