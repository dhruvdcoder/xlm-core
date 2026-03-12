from itertools import chain
from typing import (
    Any,
    Dict,
    List,
    Tuple,
)
import nltk
import torch

from xlm.datamodule import SimpleSpaceTokenizer
from xlm.utils.rank_zero import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)

try:
    from nltk.parse import EarleyChartParser
except ImportError:
    logger.warning("NLTK is not installed. CFG parser and tasks will not be available.")
    EarleyChartParser = None


G1_str = """
S -> NP VP | NP Vindirect C S
NP -> NProp | Det N | Det A N | NP PP | NP Poss NP
PP -> P NP
VP -> Vintr | Vtr NP | VP PP
A -> 'big' | 'small' | 'happy' | 'sad'
C -> 'that'
Det -> 'the' | 'my' | 'your'
N -> 'man' | 'telescope' | 'elephant'
NProp -> 'John' | 'Mary' | 'Alice'
Vintr -> 'sneezed' | 'laughed'
Vtr -> 'chased' | 'bought'
Vindirect -> 'said' | 'thought'
P -> 'with' | 'under'
Poss -> '`s'
"""


class SerializableParser:
    def __init__(self, grammar_str: str):
        self.grammar = nltk.CFG.fromstring(grammar_str)
        self._parser = None

    @property
    def parser(self):
        if self._parser is None:
            self._parser = EarleyChartParser(self.grammar)
        return self._parser

    def parse(self, tokens: List[str]):
        return self.parser.parse(tokens)

    def __getstate__(self):
        # Don't pickle the parser instance
        return {"grammar": self.grammar}

    def __setstate__(self, state):
        # Recreate grammar but not parser
        self.grammar = state["grammar"]
        self._parser = None


def g1_parser():
    return SerializableParser(G1_str)


def preprocess_fn(
    example: Dict[str, Any], tokenizer: SimpleSpaceTokenizer
) -> Dict[str, Any]:
    text = example["text"]
    token_ids = tokenizer(text, add_special_tokens=False)
    return {
        "token_ids": token_ids["input_ids"],
    }


def parsable(batch, loss_dict, tokenizer, parser=None):
    if parser is None:
        raise ValueError("Parser is required")
    ignore = {
        tokenizer.pad_token_id,
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
        tokenizer.unk_token_id,
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
        tokenizer.mask_token_id,
    }

    ids_batch = loss_dict["ids"].cpu().tolist()
    tokens_batch = [
        tokenizer.convert_ids_to_tokens(
            [int(token_id) for token_id in ids if token_id not in ignore]
        )
        for ids in ids_batch
    ]
    values = []
    for token_seq in tokens_batch:
        try:
            parsed = list(parser.parse(token_seq))
            if parsed:
                values.append(1.0)
            else:
                values.append(0.0)
        except Exception:
            values.append(0.0)
    return {"value": torch.tensor(values, device=loss_dict["ids"].device)}


def create_tokenizer_cfg_small(**kwargs):  # **kwargs for Hydra config merge compatibility
    # fmt: off
    vocab = ['small', 'happy', '`s', 'John', 'Alice', 'said', 'Mary', 'thought', 'laughed', 'bought', 'your', 'man', 'my', 'telescope', 'chased', 'elephant', 'that', 'sad', 'big', 'the', 'sneezed']
    # fmt: on
    tokenizer = SimpleSpaceTokenizer(vocab)
    return tokenizer