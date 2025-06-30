import csv
import os
import random
from itertools import chain, repeat
from pathlib import Path
from typing import (
    Generator,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    TypedDict,
    Union,
    cast,
)

import numpy as np
import torch
from jaxtyping import Float, Integer
from numpy import ndarray as NA
from torch import Tensor as TT
from torch.utils.data import DataLoader
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
from transformers import BatchEncoding

from pcdd import flags
from pcdd.datamodule.xlnet.datamodule_xlnet import (
    XLNetPredictionBatch,
    XLNetTrainingBatch,
    print_batch_xlnet,
    xlnet_prediction_collator,
    xlnet_training_collator,
)

from xlm.datamodule import (
    BaseCollatorInput,
    BaseDataModule,
    Collator,
    DataLoaderKwargs,
    DefaultIDLMCollator,
    DefaultIDLMCollatorForPrediction,
    DefaultILMCollator,
    DefaultILMCollatorForPrediction,
    DefaultILMWithLengthClassificationCollator,
    DefaultILMWithLengthClassificationCollatorForPrediction,
    DefaultITCollator,
    DefaultITCollatorForPrediction,
    DefaultMDLMCollator,
    DefaultMDLMCollatorForPrediction,
    DefaultARLMCollator,
    DefaultARLMCollatorForPrediction,
    DefaultMLMCollator,
    DefaultMLMCollatorForPrediction,
    IDLMBatch,
    ILMBatch,
    ITBatch,
    MDLMBatch,
    MLMBatch,
    ARLMBatch,
    Tokenizer,
    pad_truncate,
    pad_left_truncate,
    pad_prefix_suffix,
    print_batch_idlm,
    print_batch_ilm,
    print_batch_it,
    print_batch_mdlm,
    print_batch_arlm,
    print_batch_mlm,
)
from pcdd.noise_schedule.noise_schedule import NoiseSchedule
from pcdd.utils.rank_zero import RankedLogger, rank_zero_only, warn_once

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
# region: Graph Generation


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
# region: Tokenization


def _insert_sep(ids: Iterable[int], sep_id: int) -> List[int]:
    """
    Example:
    >>> _insert_sep([1, 2, 3], 0)
    [1, 0, 2, 0, 3, 0]
    """
    return list(chain.from_iterable(zip(ids, repeat(sep_id))))


class _SimpleSpaceTokenizer(Tokenizer):
    """
    Override `graph_to_hf_example` to implement different input formats to the model.
    """

    def __init__(
        self,
        vocab_size: int,
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        cls_token: str = "<cls>",
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        mask_token: str = "<mask>",
        edge_sep_token: str = "<es>",
        source_goal_sep_token: str = "<sts>",
    ):

        self.main_vocab = {str(i): i for i in range(vocab_size)}
        self.special_tokens = [
            cls_token,
            bos_token,
            eos_token,
            unk_token,
            pad_token,
            mask_token,
            edge_sep_token,
            source_goal_sep_token,
        ]

        # Merge vocab
        self.special_tokens_map = {
            token: idx + len(self.main_vocab)
            for idx, token in enumerate(self.special_tokens)
        }
        # this is different from pre-trained huggingface tokenizers,
        # which keep a separate vocab for the special tokens if the special tokens are added post-pretraining.
        # https://stackoverflow.com/questions/67412925/what-is-the-difference-between-lentokenizer-and-tokenizer-vocab-size
        self.vocab = dict(self.main_vocab)
        self.vocab.update(self.special_tokens_map)

        # Reverse lookup
        self.id_to_token = {v: k for k, v in self.vocab.items()}

        # Register important tokens
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.mask_token = mask_token
        self.edge_sep_token = edge_sep_token
        self.source_goal_sep_token = source_goal_sep_token
        self.cls_token = cls_token
        self.bos_token_id = self.vocab[self.bos_token]
        self.eos_token_id = self.vocab[self.eos_token]
        self.cls_token_id = self.vocab[self.cls_token]
        self.unk_token_id = self.vocab[self.unk_token]
        self.pad_token_id = self.vocab[self.pad_token]
        self.mask_token_id = self.vocab[self.mask_token]
        self.edge_sep_token_id = self.vocab[self.edge_sep_token]
        self.source_goal_sep_token_id = self.vocab[self.source_goal_sep_token]

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def full_vocab_size(self):
        return len(self.vocab)

    def __len__(self):
        return len(self.vocab)

    def __call__(
        self, text: Union[str, List[str]], add_special_tokens: bool = False
    ) -> BatchEncoding:
        raise NotImplementedError(
            "You should not need to perform tokenization for Star Graph."
        )

    def tokenize(self, text: str) -> List[str]:
        # Split on whitespace
        return text.split()

    def convert_tokens_to_ids(
        self, tokens: Union[str, List[str]]
    ) -> Union[int, List[int]]:
        if isinstance(tokens, str):
            return self.vocab.get(tokens, self.unk_token_id)
        else:
            return [self.vocab.get(t, self.unk_token_id) for t in tokens]

    def convert_ids_to_tokens(
        self, ids: Union[int, List[int]]
    ) -> Union[str, List[str]]:
        if isinstance(ids, int):
            return self.id_to_token.get(ids, self.unk_token)
        else:
            return [self.id_to_token.get(i, self.unk_token) for i in ids]

    def decode(
        self,
        token_ids: Union[List[int], Integer[TT, " seq_len"]],
        skip_special_tokens: bool = False,
    ) -> str:
        _token_ids: List[int] = (
            token_ids.tolist()  # type: ignore
            if isinstance(token_ids, torch.Tensor)
            else token_ids
        )
        tokens = self.convert_ids_to_tokens(_token_ids)
        # remove special tokens
        if skip_special_tokens:
            tokens = [t for t in tokens if t not in self.special_tokens_map]
        return " ".join(tokens)

    def batch_decode(
        self,
        token_ids: Union[List[List[int]], Integer[TT, " batch seq_len"]],
        skip_special_tokens: bool = False,
    ) -> List[str]:
        if skip_special_tokens:
            warn_once(
                logger,
                "skip_special_tokens is not supported for SimpleSpaceTokenizer",
            )
        _sequences: List[List[int]] = (
            token_ids.tolist()  # type: ignore
            if isinstance(token_ids, torch.Tensor)
            else token_ids
        )
        return [
            self.decode(seq, skip_special_tokens=skip_special_tokens)
            for seq in _sequences
        ]

    def graph_to_hf_example(self, graph: Graph) -> GraphExample:
        edge_list, source, goal, path = (
            graph["edge_list"],
            graph["source"],
            graph["goal"],
            graph["path"],
        )
        edges = _insert_sep(chain(*edge_list), self.edge_sep_token_id)
        input_ids = (
            edges
            + [
                source,
                self.source_goal_sep_token_id,
                goal,
                self.bos_token_id,
            ]
            + path
        )
        attention_mask = [1] * len(input_ids)
        type_ids = [0] * (len(edges) + 4) + [1] * len(path)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": type_ids,
        }


class SimpleSpaceTokenizer(_SimpleSpaceTokenizer):
    def graph_to_hf_example(self, graph: Graph) -> GraphExample:
        edges, source, goal, path = (
            graph["edge_list"],
            graph["source"],
            graph["goal"],
            graph["path"],
        )
        path_edges: List[Tuple[int, int]] = list(zip(path, path[1:]))
        input_ids: List[int] = (
            list(chain(*edges))
            + [
                source,
                goal,
                self.bos_token_id,
            ]
            + list(chain(*path_edges))
        )
        attention_mask: List[int] = [1] * len(input_ids)
        # CLS=0, BOS+REST>=1, prefix=1, non_prefix=2
        type_ids: List[int] = [1] * (2 * len(edges) + 3) + [2] * len(
            2 * path_edges
        )
        assert len(input_ids) == len(attention_mask) == len(type_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": type_ids,
        }


class StarGraphTokenizerForIDLM(SimpleSpaceTokenizer):
    def decode(
        self,
        token_ids: Union[List[int], Integer[TT, " seq_len"]],
        skip_special_tokens: bool = False,
    ) -> str:
        _token_ids: List[int] = (
            token_ids.tolist()  # type: ignore
            if isinstance(token_ids, torch.Tensor)
            else token_ids
        )
        tokens = self.convert_ids_to_tokens(_token_ids)
        # remove special tokens
        if skip_special_tokens:
            # only skip mask and pad tokens for readability
            to_skip = {
                self.mask_token,
                self.pad_token,
            }
            tokens = [t for t in tokens if t not in to_skip]
        return " ".join(tokens)

    def graph_to_hf_example(self, graph: Graph) -> GraphExample:
        edges, source, goal, path = (
            graph["edge_list"],
            graph["source"],
            graph["goal"],
            graph["path"],
        )
        path_edges: List[Tuple[int, int]] = list(zip(path, path[1:]))
        input_ids: List[int] = (
            [self.cls_token_id]
            + list(chain(*edges))
            + [
                source,
                goal,
                self.bos_token_id,
            ]
            + list(chain(*path_edges))
        )
        attention_mask: List[int] = [1] * len(input_ids)
        # CLS=0, BOS+REST>=1, prefix=1, non_prefix=2
        type_ids: List[int] = (
            [0] + [1] * (2 * len(edges) + 3) + [2] * len(2 * path_edges)
        )
        assert len(input_ids) == len(attention_mask) == len(type_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": type_ids,
        }


class StarGraphTokenizerForARLM(SimpleSpaceTokenizer):
    def decode(
        self,
        token_ids: Union[List[int], Integer[TT, " seq_len"]],
        skip_special_tokens: bool = False,
    ) -> str:
        _token_ids: List[int] = (
            token_ids.tolist()  # type: ignore
            if isinstance(token_ids, torch.Tensor)
            else token_ids
        )
        tokens = self.convert_ids_to_tokens(_token_ids)
        # remove special tokens
        if skip_special_tokens:
            # only skip mask and pad tokens for readability
            to_skip = {
                self.mask_token,
                self.pad_token,
            }
            tokens = [t for t in tokens if t not in to_skip]
        return " ".join(tokens)

    def graph_to_hf_example(self, graph: Graph) -> GraphExample:
        edges, source, goal, path = (
            graph["edge_list"],
            graph["source"],
            graph["goal"],
            graph["path"],
        )
        path_edges: List[Tuple[int, int]] = list(zip(path, path[1:]))
        input_ids: List[int] = (
            list(chain(*edges))
            + [
                source,
                goal,
                self.bos_token_id,
            ]
            + list(chain(*path_edges))
            + [self.eos_token_id]
        )
        attention_mask: List[int] = [1] * len(input_ids)
        # CLS=0, BOS+REST>=1, prefix=1, non_prefix=2
        type_ids: List[int] = (
            [1] * (2 * len(edges) + 2) + [1] + [2] * len(2 * path_edges) + [2]
        )
        assert len(input_ids) == len(attention_mask) == len(type_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": type_ids,
        }


StarGraphTokenizerForXLNet = StarGraphTokenizerForARLM


StarGraphTokenizer = StarGraphTokenizerForIDLM


class StarGraphTokenizerForMDLM(SimpleSpaceTokenizer):
    def decode(
        self,
        token_ids: Union[List[int], Integer[TT, " seq_len"]],
        skip_special_tokens: bool = False,
    ) -> str:
        _token_ids: List[int] = (
            token_ids.tolist()  # type: ignore
            if isinstance(token_ids, torch.Tensor)
            else token_ids
        )
        tokens = self.convert_ids_to_tokens(_token_ids)
        # remove special tokens
        if skip_special_tokens:
            # only skip mask and pad tokens for readability
            to_skip = {
                self.mask_token,
                self.pad_token,
            }
            tokens = [t for t in tokens if t not in to_skip]
        return " ".join(tokens)

    def graph_to_hf_example(self, graph: Graph) -> GraphExample:
        edges, source, goal, path = (
            graph["edge_list"],
            graph["source"],
            graph["goal"],
            graph["path"],
        )
        path_edges: List[Tuple[int, int]] = list(zip(path, path[1:]))
        input_ids: List[int] = (
            list(chain(*edges))
            + [
                source,
                goal,
                self.bos_token_id,
            ]
            + list(chain(*path_edges))
        )  # edges | source | goal | BOS
        attention_mask: List[int] = [1] * len(input_ids)
        # CLS=0, BOS+REST>=1, prefix=1, non_prefix=2
        type_ids: List[int] = [1] * (2 * len(edges) + 3) + [2] * len(
            2 * path_edges
        )
        assert len(input_ids) == len(attention_mask) == len(type_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": type_ids,
        }


class StarGraphTokenizerForMDLMV2(SimpleSpaceTokenizer):
    def decode(
        self,
        token_ids: Union[List[int], Integer[TT, " seq_len"]],
        skip_special_tokens: bool = False,
    ) -> str:
        _token_ids: List[int] = (
            token_ids.tolist()  # type: ignore
            if isinstance(token_ids, torch.Tensor)
            else token_ids
        )
        tokens = self.convert_ids_to_tokens(_token_ids)
        # remove special tokens
        if skip_special_tokens:
            # only skip mask and pad tokens for readability
            to_skip = {
                self.mask_token,
                self.pad_token,
            }
            tokens = [t for t in tokens if t not in to_skip]
        return " ".join(tokens)

    def graph_to_hf_example(self, graph: Graph) -> GraphExample:
        edges, source, goal, path = (
            graph["edge_list"],
            graph["source"],
            graph["goal"],
            graph["path"],
        )
        path_edges: List[Tuple[int, int]] = list(zip(path, path[1:]))
        input_ids: List[int] = (
            list(chain(*edges))
            + [
                source,
                goal,
                self.bos_token_id,
            ]
            + list(chain(*path_edges))
            + [self.eos_token_id]
        )  # edges | source | goal | BOS
        attention_mask: List[int] = [1] * len(input_ids)
        # CLS=0, BOS+REST>=1, prefix=1, non_prefix=2
        type_ids: List[int] = (
            [1] * (2 * len(edges) + 3) + [2] * len(2 * path_edges) + [2]
        )
        assert len(input_ids) == len(attention_mask) == len(type_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": type_ids,
        }


class StarGraphTokenizerForMLM(SimpleSpaceTokenizer):
    def decode(
        self,
        token_ids: Union[List[int], Integer[TT, " seq_len"]],
        skip_special_tokens: bool = False,
    ) -> str:
        _token_ids: List[int] = (
            token_ids.tolist()  # type: ignore
            if isinstance(token_ids, torch.Tensor)
            else token_ids
        )
        tokens = self.convert_ids_to_tokens(_token_ids)
        # remove special tokens
        if skip_special_tokens:
            # only skip mask and pad tokens for readability
            to_skip = {
                self.mask_token,
                self.pad_token,
            }
            tokens = [t for t in tokens if t not in to_skip]
        return " ".join(tokens)

    def graph_to_hf_example(self, graph: Graph) -> GraphExample:
        edges, source, goal, path = (
            graph["edge_list"],
            graph["source"],
            graph["goal"],
            graph["path"],
        )
        path_edges: List[Tuple[int, int]] = list(zip(path, path[1:]))
        input_ids: List[int] = (
            list(chain(*edges))
            + [
                source,
                goal,
                self.bos_token_id,
            ]
            + list(chain(*path_edges))
            + [self.eos_token_id]
        )  # edges | source | goal | BOS
        attention_mask: List[int] = [1] * len(input_ids)
        # CLS=0, BOS+REST>=1, prefix=1, non_prefix=2
        type_ids: List[int] = (
            [1] * (2 * len(edges) + 3) + [2] * len(2 * path_edges) + [2]
        )
        assert len(input_ids) == len(attention_mask) == len(type_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": type_ids,
        }


# endregion: Tokenization
########################################################


########################################################
# region: Dataset


def serialize_edge_list(edge_list: List[Tuple[int, int]]) -> str:
    return "|".join([",".join(map(str, edge)) for edge in edge_list])


def serialize_path(lst: List[int]) -> str:
    return ",".join(map(str, lst))


def deserialize_edge_list(serialized_str: str) -> List[Tuple[int, int]]:
    return [
        tuple(map(int, edge.split(","))) for edge in serialized_str.split("|")
    ]  # type: ignore


def deserialize_path(serialized_str: str) -> List[int]:
    return [int(x) for x in serialized_str.split(",")]


class StarGraphDataset(torch.utils.data.Dataset[GraphExample]):
    def __init__(
        self,
        data_file: Path,
        tokenizer: SimpleSpaceTokenizer,
        max_examples: Optional[int] = None,
    ):
        self.max_examples = max_examples
        self.data_file = data_file
        self.tokenizer = tokenizer
        self.data: Optional[List[Graph]] = None

    def load_dataset(self, keep_ids: Optional[List[int]] = None):
        if keep_ids is not None:
            _keep_ids = set(keep_ids)
        if self.data is not None:
            ranked_logger.info(
                f"Dataset {self.data_file} already loaded. Skipping."
            )
        self.data = []
        with open(self.data_file, "r", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for i, row in enumerate(reader):
                if keep_ids is not None and i not in _keep_ids:
                    continue
                graph: Graph = {
                    "edge_list": deserialize_edge_list(row["edge_list"]),
                    "source": int(row["source"]),
                    "goal": int(row["goal"]),
                    "path": deserialize_path(row["path"]),
                }
                self.data.append(graph)
        if self.max_examples is not None:
            self.data = self.data[: self.max_examples]

    def __len__(self) -> int:
        if self.data is None:
            raise ValueError(
                "Dataset not loaded. Call load_dataset() before querying the length."
            )
        return len(self.data)

    def __getitem__(self, index: int) -> GraphExample:
        if self.data is None:
            raise ValueError(
                "Dataset not loaded. Call load_dataset() before querying the length."
            )
        graph = self.data[index]
        return self.tokenizer.graph_to_hf_example(graph)


# endregion: Dataset
########################################################


########################################################
# region: DataModule


class StarGraphIDLMCollator(DefaultIDLMCollator):
    """Adds the `constraint`  using the `token_type_ids`"""

    def __call__(self, examples: List[BaseCollatorInput]) -> IDLMBatch:
        batch: IDLMBatch = super().__call__(examples)
        # CLS=0, BOS+REST>=1, prefix=1, non_prefix=2
        # shift the end of prefix to left by 1, by roll and fill
        batch["constraint"] = torch.roll(
            batch["token_type_ids"] <= 1, shifts=-1, dims=-1
        )
        batch["constraint"][:, -1] = 0

        return batch


class StarGraphILMCollator(DefaultILMCollator):
    def __call__(self, examples: List[BaseCollatorInput]) -> ILMBatch:
        batch: ILMBatch = super().__call__(examples)
        # CLS=0, BOS+REST>=1, prefix=1, non_prefix=2
        # shift the end of prefix to left by 1, by roll and fill
        batch["constraint"] = torch.roll(
            batch["token_type_ids"] <= 1, shifts=-1, dims=-1
        )
        batch["constraint"][:, -1] = 0

        return batch


class StarGraphITCollator(DefaultITCollator):
    def __call__(self, examples: List[BaseCollatorInput]) -> ITBatch:
        batch: ITBatch = super().__call__(examples)
        # CLS=0, BOS+REST>=1, prefix=1, non_prefix=2
        # shift the end of prefix to left by 1, by roll and fill
        batch["constraint"] = torch.roll(
            batch["token_type_ids"] <= 1, shifts=-1, dims=-1
        )
        batch["constraint"][:, -1] = 0

        return batch


class StarGraphILMWithLengthClassificationCollator(
    DefaultILMWithLengthClassificationCollator
):
    def __call__(self, examples: List[BaseCollatorInput]) -> ILMBatch:
        batch: ILMBatch = super().__call__(examples)
        batch["constraint"] = torch.roll(
            batch["token_type_ids"] <= 1, shifts=-1, dims=-1
        )
        batch["constraint"][:, -1] = 0
        return batch


class StarGraphIDLMCollatorForPrediction(DefaultIDLMCollatorForPrediction):
    def __call__(self, examples: List[BaseCollatorInput]) -> IDLMBatch:
        batch: IDLMBatch = super().__call__(examples)
        # CLS=0, BOS+REST>=1, prefix=1, non_prefix=2
        batch["constraint"] = torch.roll(
            batch["token_type_ids"] <= 1, shifts=-1, dims=-1
        )
        batch["constraint"][:, -1] = 0
        return batch


class StarGraphILMCollatorForPrediction(DefaultILMCollatorForPrediction):
    def __call__(self, examples: List[BaseCollatorInput]) -> ILMBatch:
        batch: ILMBatch = super().__call__(examples)
        # CLS=0, BOS+REST>=1, prefix=1, non_prefix=2
        batch["constraint"] = torch.roll(
            batch["token_type_ids"] <= 1, shifts=-1, dims=-1
        )
        batch["constraint"][:, -1] = 0
        return batch


class StarGraphITCollatorForPrediction(DefaultITCollatorForPrediction):
    def __call__(self, examples: List[BaseCollatorInput]) -> ITBatch:
        batch: ITBatch = super().__call__(examples)
        batch["constraint"] = torch.roll(
            batch["token_type_ids"] <= 1, shifts=-1, dims=-1
        )
        batch["constraint"][:, -1] = 0
        return batch


class StarGraphILMWithLengthClassificationCollatorForPredictionDebug(
    DefaultILMWithLengthClassificationCollator
):
    def __call__(self, examples: List[BaseCollatorInput]) -> ILMBatch:
        batch: ILMBatch = super().__call__(examples)
        # CLS=0, BOS+REST>=1, prefix=1, non_prefix=2
        batch["constraint"] = torch.roll(
            batch["token_type_ids"] <= 1, shifts=-1, dims=-1
        )
        batch["constraint"][:, -1] = 0
        return batch

    def sample_n_drops(self, seq_len: int) -> int:
        return 0


class StarGraphILMWithLengthClassificationCollatorForPrediction(
    DefaultILMWithLengthClassificationCollatorForPrediction
):
    def __call__(self, examples: List[BaseCollatorInput]) -> ILMBatch:
        batch: ILMBatch = super().__call__(examples)
        batch["constraint"] = torch.roll(
            batch["token_type_ids"] <= 1, shifts=-1, dims=-1
        )
        batch["constraint"][:, -1] = 0
        return batch


# -----------------------------
class StarGraphARLMCollator(Collator):
    def __init__(
        self,
        tokenizer: Tokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ):
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.noise_schedule = noise_schedule

    @property
    def vocab_size(self) -> int:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not set")
        return len(self.tokenizer)

    def get_max_len(self, batch: List[BaseCollatorInput]) -> int:
        return self.block_size

    def __call__(
        self,
        examples: List[BaseCollatorInput],
    ) -> ARLMBatch:
        max_seq_len = self.get_max_len(examples)
        batch = pad_left_truncate(
            examples, max_seq_len, self.tokenizer.pad_token_id
        )
        constraint = (
            (batch["token_type_ids"] == 1)
            | (batch["input_ids"] == self.tokenizer.pad_token_id)
        ).long()
        drop = (
            (batch["token_type_ids"] == 1)
            | (batch["input_ids"] == self.tokenizer.pad_token_id)
        ).long()
        return {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"].bool(),
            "token_type_ids": batch["token_type_ids"],
            "drop": drop,
            "constraint": constraint,
        }


class StarGraphARLMCollatorForPrediction(Collator):
    def __init__(
        self,
        tokenizer: Tokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ):
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.noise_schedule = noise_schedule

    @property
    def vocab_size(self) -> int:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not set")
        return len(self.tokenizer)

    def get_max_len(self, batch: List[BaseCollatorInput]) -> int:
        return self.block_size

    def __call__(
        self,
        examples: List[BaseCollatorInput],
    ) -> ARLMBatch:
        max_seq_len = self.get_max_len(examples)
        batch_size = len(examples)
        base_batch = pad_prefix_suffix(self.tokenizer, examples, max_seq_len)
        base_batch["constraint"] = (
            (base_batch["token_type_ids"] == 1)
            | (base_batch["input_ids"] == self.tokenizer.pad_token_id)
        ).long()
        base_batch["drop"] = (1 - base_batch["constraint"]).long()
        base_batch = cast(ARLMBatch, base_batch)
        return base_batch


class StarGraphXLNetCollator(Collator):
    def __init__(
        self,
        tokenizer: Tokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ):
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.noise_schedule = noise_schedule

    @property
    def vocab_size(self) -> int:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not set")
        return len(self.tokenizer)

    def get_max_len(self, batch: List[BaseCollatorInput]) -> int:
        return self.block_size

    def __call__(
        self,
        examples: List[BaseCollatorInput],
    ) -> XLNetTrainingBatch:
        max_seq_len = self.get_max_len(examples)
        batch = xlnet_training_collator(
            examples,
            max_seq_len,
            pad_token_id=self.tokenizer.pad_token_id,
            prefix_attention_type="causal",
        )

        return {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"].bool(),
            "token_type_ids": batch["token_type_ids"],
            "perm_mask": batch["perm_mask"],
            "target_mapping": batch["target_mapping"],
            "labels": batch["labels"],
        }


class StarGraphXLNetCollatorForPrediction(Collator):
    def __init__(
        self,
        tokenizer: Tokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ):
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.noise_schedule = noise_schedule

    @property
    def vocab_size(self) -> int:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not set")
        return len(self.tokenizer)

    def get_max_len(self, batch: List[BaseCollatorInput]) -> int:
        return self.block_size

    def __call__(
        self,
        examples: List[BaseCollatorInput],
    ) -> XLNetPredictionBatch:
        max_seq_len = self.get_max_len(examples)
        batch = xlnet_prediction_collator(
            examples,
            max_seq_len,
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        return {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"].bool(),
            "drop": batch["drop"].long(),
        }


if flags.DEBUG_STOPPING:  # CLEANUP
    StarGraphILMWithLengthClassificationCollatorForPrediction = (
        StarGraphILMWithLengthClassificationCollatorForPredictionDebug
    )


def _get_default_dataloader_kwargs(
    type: Literal["train", "val", "test", "predict"],
) -> DataLoaderKwargs:
    return {
        "batch_size": 64,
        "num_workers": 4,
        "shuffle": True if type == "train" else False,
        "pin_memory": True,
    }


class StarGraphDataModule(BaseDataModule):
    prepare_data_per_node: bool = False

    def get_default_collator(
        self,
        tokenizer: SimpleSpaceTokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> Collator:
        raise NotImplementedError("Not implemented")

    def get_default_prediction_collator(
        self,
        tokenizer: SimpleSpaceTokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> Collator:
        raise NotImplementedError("Not implemented")

    def __init__(
        self,
        manual_cache_dir: str,
        train_generator: StarGraphGenerator,
        val_generator: StarGraphGenerator,
        test_generator: StarGraphGenerator,
        tokenizer: SimpleSpaceTokenizer,
        noise_schedule: NoiseSchedule,
        train_dataloader_kwargs: Optional[DataLoaderKwargs] = None,
        val_dataloader_kwargs: Optional[DataLoaderKwargs] = None,
        test_dataloader_kwargs: Optional[DataLoaderKwargs] = None,
        predict_dataloader_kwargs: Optional[DataLoaderKwargs] = None,
        rewrite_manual_cache: bool = False,
        global_batch_size: Optional[int] = None,
        block_size: int = 128,
        num_dataset_workers: int = 1,
        num_unconditional_samples: Optional[int] = None,
        train_filter_file: Optional[
            str
        ] = None,  # file will the list of ids to keep
        val_filter_file: Optional[str] = None,
        test_filter_file: Optional[str] = None,
        predict_filter_file: Optional[str] = None,
        # data_format: Literal["standard"] = "standard",
    ):
        super().__init__()
        self.manual_cache_dir = manual_cache_dir
        self.train_generator = train_generator
        self.val_generator = val_generator
        self.test_generator = test_generator
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None
        self.rewrite_manual_cache = rewrite_manual_cache
        self.tokenizer = tokenizer
        self.noise_schedule = noise_schedule
        self.train_dataloader_kwargs: DataLoaderKwargs = (
            train_dataloader_kwargs or _get_default_dataloader_kwargs("train")
        )
        self.val_dataloader_kwargs: DataLoaderKwargs = (
            val_dataloader_kwargs or _get_default_dataloader_kwargs("val")
        )
        self.test_dataloader_kwargs: DataLoaderKwargs = (
            test_dataloader_kwargs or _get_default_dataloader_kwargs("test")
        )
        self.predict_dataloader_kwargs: DataLoaderKwargs = (
            predict_dataloader_kwargs
            or _get_default_dataloader_kwargs("predict")
        )
        self.collator = self.get_default_collator(
            tokenizer, block_size, noise_schedule
        )
        self.prediction_collator = self.get_default_prediction_collator(
            tokenizer, block_size, noise_schedule
        )
        self.train_dataloader_names = {
            0: "lm",
            1: "prediction",
        }
        self.val_dataloader_names = {
            0: "lm",
            1: "prediction",
        }
        self.test_dataloader_names = {
            0: "lm",
            1: "prediction",
        }
        self.predict_dataloader_names = {
            0: "prediction",
        }
        # approximate_expected_max_length = (
        #    2 * 2 * self.train_generator.max_num_edges
        #    + 4
        #    + self.train_generator.max_pathlength
        # )
        approximate_expected_max_length = (
            1
            + 2 * self.train_generator.max_num_edges
            + 3
            + 2 * self.train_generator.max_pathlength
            - 2
        )
        if block_size < approximate_expected_max_length:
            raise ValueError(
                f"block_size must be greater than or equal to {approximate_expected_max_length}"
            )
        self.block_size = block_size
        if global_batch_size is not None:
            logger.warning(
                "Global batch size will be ignore. We don't support DDP for StarGraph."
            )
        self.num_unconditional_samples = num_unconditional_samples
        self.train_filter_file = train_filter_file
        self.val_filter_file = val_filter_file
        self.test_filter_file = test_filter_file
        self.predict_filter_file = predict_filter_file

    def _get_cache_file(self, split: Literal["train", "val", "test"]):
        generator = getattr(self, f"{split}_generator")
        return (
            Path(self.manual_cache_dir)
            / generator.unique_str()
            / f"{split}.csv"
        )
        # filter_file = getattr(self, f"{split}_filter_file")
        # if filter_file is None:
        #    return (
        #        Path(self.manual_cache_dir)
        #        / generator.unique_str()
        #        / f"{split}.csv"
        #    )
        # else:
        #    return (
        #        Path(self.manual_cache_dir)
        #        / generator.unique_str()
        #        / f"{os.path.basename(filter_file)}_{split}.csv"
        #    )

    def _write_to_cache(self, cache_file: Path, graphs: Iterator[Graph]):
        try:
            with open(cache_file, "w", newline="") as csvfile:
                fieldnames = ["edge_list", "source", "goal", "path"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                for graph in tqdm(graphs, desc=f"Writing graphs to cache"):
                    writer.writerow(
                        {
                            "edge_list": serialize_edge_list(
                                graph["edge_list"]
                            ),
                            "source": graph["source"],
                            "goal": graph["goal"],
                            "path": serialize_path(graph["path"]),
                        }
                    )
        except Exception as e:
            # remove the cache file if it is corrupted and reraise
            cache_file.unlink()
            raise e

    def _prepare_data_split(self, split: Literal["train", "val", "test"]):
        """Generate and cache data if not already cached or if manual_cache_dir is None."""
        # check if cache dir exists
        cache = self._get_cache_file(split)
        cache.parent.mkdir(parents=True, exist_ok=True)
        if cache.exists():
            if not self.rewrite_manual_cache:
                ranked_logger.info(
                    f"Cache file {cache} for {split} already exists. Nothing to preprocess."
                )
                return
            else:
                ranked_logger.info(
                    f"Cache file {cache} for {split} already exists. Overwriting."
                )
        ranked_logger.info(f"Preprocessing data for {split}")
        generator = getattr(self, f"{split}_generator")
        ranked_logger.info(f"Writing data to {cache}")
        self._write_to_cache(cache, generator.graphs())

    def set_epoch(self, epoch: int):
        pass  # nothing to do here

    def prepare_data(self):
        splits: List[Literal["train", "val", "test"]] = [
            "train",
            "val",
            "test",
        ]
        for split in splits:
            self._prepare_data_split(split)

    def _load_keep_ids(self, filter_file: str):
        with open(filter_file, "r") as f:
            return [int(line.strip()) for line in f]

    def setup(self, stage: Optional[str] = None):
        if (stage == "fit" or stage is None) and self.train_dataset is None:
            self.train_dataset = StarGraphDataset(
                self._get_cache_file("train"), self.tokenizer
            )
            keep_ids = (
                self._load_keep_ids(self.train_filter_file)
                if self.train_filter_file
                else None
            )

            self.train_dataset.load_dataset(keep_ids=keep_ids)
        if (
            stage == "fit" or stage == "validate"
        ) and self.val_dataset is None:
            self.val_dataset = StarGraphDataset(
                (
                    self._get_cache_file("val")
                    if not flags.DEBUG_STOPPING
                    else self._get_cache_file("train")
                ),
                # self._get_cache_file("test"),
                self.tokenizer,  # just use test because we are not doing tuning
            )
            keep_ids = (
                self._load_keep_ids(self.val_filter_file)
                if self.val_filter_file
                else None
            )
            self.val_dataset.load_dataset(keep_ids=keep_ids)
        if stage == "test" and self.test_dataset is None:
            self.test_dataset = StarGraphDataset(
                self._get_cache_file("test"), self.tokenizer
            )
            keep_ids = (
                self._load_keep_ids(self.test_filter_file)
                if self.test_filter_file
                else None
            )
            self.test_dataset.load_dataset(keep_ids=keep_ids)
        if stage == "predict" and self.predict_dataset is None:
            self.predict_dataset = StarGraphDataset(
                self._get_cache_file("test"),
                self.tokenizer,
                max_examples=self.num_unconditional_samples,
            )
            keep_ids = (
                self._load_keep_ids(self.predict_filter_file)
                if self.predict_filter_file
                else None
            )
            self.predict_dataset.load_dataset(keep_ids=keep_ids)

    def train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Train dataset not loaded. Call setup() first.")
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        generator = torch.Generator().manual_seed(seed)
        dataloader = StatefulDataLoader(
            self.train_dataset,
            generator=generator,
            collate_fn=self.collator,
            **self.train_dataloader_kwargs,
        )
        return dataloader

    def val_dataloader(self):
        if self.val_dataset is None:
            raise ValueError("Val dataset not loaded. Call setup() first.")
        lm_dataloader = DataLoader(
            (
                self.val_dataset
                if not flags.DEBUG_OVERFIT
                else self.train_dataset
            ),
            collate_fn=self.collator,
            **self.val_dataloader_kwargs,
        )
        prediction_dataloader = DataLoader(
            (
                self.val_dataset
                if not flags.DEBUG_OVERFIT
                else self.train_dataset
            ),
            collate_fn=self.prediction_collator,
            **self.val_dataloader_kwargs,
        )
        return [lm_dataloader, prediction_dataloader]

    def test_dataloader(self):
        if self.test_dataset is None:
            raise ValueError("Test dataset not loaded. Call setup() first.")
        lm_dataloader = DataLoader(
            self.test_dataset,
            collate_fn=self.collator,
            **self.test_dataloader_kwargs,
        )
        prediction_dataloader = DataLoader(
            self.test_dataset,
            collate_fn=self.prediction_collator,
            **self.test_dataloader_kwargs,
        )
        return [lm_dataloader, prediction_dataloader]

    def predict_dataloader(self):
        prediction_dataloader = DataLoader(
            self.predict_dataset,
            collate_fn=self.prediction_collator,
            **self.predict_dataloader_kwargs,
        )
        return prediction_dataloader


class StarGraphDataModuleForMDLM(StarGraphDataModule):
    def get_default_collator(
        self,
        tokenizer: SimpleSpaceTokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> Collator:
        return DefaultMDLMCollator(
            tokenizer, block_size, noise_schedule, loss_on_padding=True
        )

    def get_default_prediction_collator(
        self,
        tokenizer: SimpleSpaceTokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> Collator:
        return DefaultMDLMCollatorForPrediction(
            tokenizer, block_size, noise_schedule
        )

    @rank_zero_only
    def print_batch(
        self,
        batch: IDLMBatch,
        split: Literal["train", "val", "test"],
        dataloader_idx: Optional[int] = None,
    ):
        print_batch_mdlm(self, batch, split, dataloader_idx)


class StarGraphDataModuleForIDLM(StarGraphDataModule):
    def get_default_collator(
        self,
        tokenizer: SimpleSpaceTokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> Collator:
        return StarGraphIDLMCollator(tokenizer, block_size, noise_schedule)

    def get_default_prediction_collator(
        self,
        tokenizer: SimpleSpaceTokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> Collator:
        return StarGraphIDLMCollatorForPrediction(
            tokenizer, block_size, noise_schedule
        )

    @rank_zero_only
    def print_batch(
        self,
        batch: IDLMBatch,
        split: Literal["train", "val", "test"],
        dataloader_idx: Optional[int] = None,
    ):
        print_batch_idlm(self, batch, split, dataloader_idx)


class StarGraphDataModuleForILM(StarGraphDataModule):
    def get_default_collator(
        self,
        tokenizer: SimpleSpaceTokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> Collator:
        return StarGraphILMCollator(tokenizer, block_size, noise_schedule)

    def get_default_prediction_collator(
        self,
        tokenizer: SimpleSpaceTokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> Collator:
        return StarGraphILMCollatorForPrediction(
            tokenizer, block_size, noise_schedule
        )

    @rank_zero_only
    def print_batch(
        self,
        batch: ILMBatch,
        split: Literal["train", "val", "test"],
        dataloader_idx: Optional[int] = None,
    ):
        print_batch_ilm(self, batch, split, dataloader_idx)


class StarGraphDataModuleForIT(StarGraphDataModule):
    def get_default_collator(
        self,
        tokenizer: SimpleSpaceTokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> Collator:
        return StarGraphITCollator(tokenizer, block_size, noise_schedule)

    def get_default_prediction_collator(
        self,
        tokenizer: SimpleSpaceTokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> Collator:
        return StarGraphITCollatorForPrediction(
            tokenizer, block_size, noise_schedule
        )

    @rank_zero_only
    def print_batch(
        self,
        batch: ITBatch,
        split: Literal["train", "val", "test"],
        dataloader_idx: Optional[int] = None,
    ):
        print_batch_it(self, batch, split, dataloader_idx)


class StarGraphDataModuleForILMWithLengthClassification(
    StarGraphDataModuleForILM
):
    def get_default_collator(
        self,
        tokenizer: SimpleSpaceTokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> Collator:
        return StarGraphILMWithLengthClassificationCollator(
            tokenizer, block_size, noise_schedule
        )

    def get_default_prediction_collator(
        self,
        tokenizer: SimpleSpaceTokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> Collator:
        return StarGraphILMWithLengthClassificationCollatorForPrediction(
            tokenizer, block_size, noise_schedule
        )


class StarGraphDataModuleForARLM(StarGraphDataModule):
    def get_default_collator(
        self,
        tokenizer: StarGraphTokenizerForARLM,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> Collator:
        return StarGraphARLMCollator(tokenizer, block_size, noise_schedule)

    def get_default_prediction_collator(
        self,
        tokenizer: StarGraphTokenizerForARLM,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> Collator:
        return StarGraphARLMCollatorForPrediction(
            tokenizer, block_size, noise_schedule
        )

    @rank_zero_only
    def print_batch(
        self,
        batch: ARLMBatch,
        split: Literal["train", "val", "test"],
        dataloader_idx: Optional[int] = None,
    ):
        print_batch_arlm(self, batch, split, dataloader_idx)


class StarGraphDataModuleForMLM(StarGraphDataModule):
    def get_default_collator(
        self,
        tokenizer: SimpleSpaceTokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> Collator:
        return DefaultMLMCollator(
            tokenizer, block_size, noise_schedule, loss_on_padding=True
        )

    def get_default_prediction_collator(
        self,
        tokenizer: SimpleSpaceTokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> Collator:
        return DefaultMLMCollatorForPrediction(
            tokenizer, block_size, noise_schedule
        )

    @rank_zero_only
    def print_batch(
        self,
        batch: MLMBatch,
        split: Literal["train", "val", "test"],
        dataloader_idx: Optional[int] = None,
    ):
        print_batch_mlm(self, batch, split, dataloader_idx)


class StarGraphDataModuleForXLNet(StarGraphDataModule):
    def get_default_collator(
        self,
        tokenizer: SimpleSpaceTokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> Collator:
        return StarGraphXLNetCollator(tokenizer, block_size, noise_schedule)

    def get_default_prediction_collator(
        self,
        tokenizer: SimpleSpaceTokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ) -> Collator:
        return StarGraphXLNetCollatorForPrediction(
            tokenizer, block_size, noise_schedule
        )

    @rank_zero_only
    def print_batch(
        self,
        batch: XLNetPredictionBatch,
        split: Literal["train", "val", "test"],
        dataloader_idx: Optional[int] = None,
    ):
        print_batch_xlnet(self, batch, split, dataloader_idx)


# endregion: DataModule
########################################################
