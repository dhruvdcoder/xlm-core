from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
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


# ----------------------------
# Helpers for syntactic metrics (production keys, parse trees)
# ----------------------------


def _parse_tokens(
    parser: SerializableParser,
    tokens: Sequence[str],
    max_parses: int = 50,
) -> List[Any]:
    """Return up to max_parses parse trees for a token sequence.

    Shared by parsable() and compute_syntactic_metrics.
    Returns [] if tokens are not covered by the grammar or parsing fails.
    """
    trees: List[Any] = []
    try:
        for t in parser.parse(list(tokens)):
            trees.append(t)
            if len(trees) >= max_parses:
                break
    except (ValueError, Exception):
        pass
    return trees


def _prod_to_key(
    prod: Any,
    *,
    strip_terminals: bool,
    include_lexical: bool,
) -> Optional[str]:
    """Convert an NLTK Production into a string key."""
    lhs = str(prod.lhs())
    rhs = prod.rhs()

    is_lexical = all(isinstance(sym, str) for sym in rhs)
    if is_lexical and not include_lexical:
        return None

    rhs_parts = []
    for sym in rhs:
        if isinstance(sym, str):
            rhs_parts.append("<T>" if strip_terminals else repr(sym))
        else:
            rhs_parts.append(str(sym))

    return f"{lhs} -> {' '.join(rhs_parts)}"


def _tree_production_keys(
    tree: Any,
    *,
    strip_terminals: bool,
    include_lexical: bool,
) -> List[str]:
    """Extract production keys from a single parse tree."""
    keys: List[str] = []
    for prod in tree.productions():
        k = _prod_to_key(
            prod,
            strip_terminals=strip_terminals,
            include_lexical=include_lexical,
        )
        if k is not None:
            keys.append(k)
    return keys


def _tree_template_signature(
    tree: Any,
    *,
    include_lexical: bool = False,
) -> str:
    """Template signature capturing syntax, not word choice."""
    keys = _tree_production_keys(
        tree,
        strip_terminals=True,
        include_lexical=include_lexical,
    )
    keys_sorted = sorted(keys)
    return " | ".join(keys_sorted)


# ----------------------------
# SyntacticMetrics dataclass and compute function
# ----------------------------


@dataclass
class SyntacticMetrics:
    num_total: int
    num_parsable: int
    parse_rate: float
    rule_coverage: float
    rule_entropy_nats: float
    rule_entropy_bits: float
    template_diversity: float
    avg_num_parses: float
    frac_ambiguous: float


def compute_syntactic_metrics(
    sentences: Sequence[str],
    grammar_str: str,
    *,
    tokenizer: Callable[[str], List[str]] = str.split,
    max_parses_per_sent: int = 50,
    parse_mode: str = "first",
    include_lexical_rules: bool = False,
    strip_terminals_for_rules: bool = True,
    return_per_sample: bool = False,
) -> SyntacticMetrics | Tuple[SyntacticMetrics, List[Tuple[bool, Optional[str]]]]:
    """Compute syntactic metrics (rule coverage, entropy, template diversity).

    parse_mode: "first" uses only first parse per sentence; "all" aggregates
    productions across all parses. include_lexical_rules=False focuses on syntax.
    """
    if parse_mode not in {"first", "all"}:
        raise ValueError("parse_mode must be 'first' or 'all'")

    grammar = nltk.CFG.fromstring(grammar_str)
    sp = SerializableParser(grammar_str)

    grammar_prod_keys = []
    for prod in grammar.productions():
        k = _prod_to_key(
            prod,
            strip_terminals=strip_terminals_for_rules,
            include_lexical=include_lexical_rules,
        )
        if k is not None:
            grammar_prod_keys.append(k)
    grammar_prod_set = set(grammar_prod_keys)
    if not grammar_prod_set:
        raise ValueError(
            "No grammar productions left after filtering; check include_lexical_rules."
        )

    used_prod_set: set = set()
    prod_counter: Counter = Counter()
    template_set: set = set()
    per_sample: List[Tuple[bool, Optional[str]]] = [] if return_per_sample else []

    num_total = len(sentences)
    num_parsable = 0
    total_num_parses = 0
    num_ambiguous = 0

    for s in sentences:
        tokens = tokenizer(s)
        trees = _parse_tokens(sp, tokens, max_parses=max_parses_per_sent)

        if not trees:
            if return_per_sample:
                per_sample.append((False, None))
            continue

        num_parsable += 1
        if return_per_sample:
            sig = _tree_template_signature(
                trees[0], include_lexical=include_lexical_rules
            )
            per_sample.append((True, sig))
        total_num_parses += len(trees)
        if len(trees) > 1:
            num_ambiguous += 1

        template_set.add(
            _tree_template_signature(trees[0], include_lexical=include_lexical_rules)
        )

        trees_to_use = [trees[0]] if parse_mode == "first" else trees
        for t in trees_to_use:
            keys = _tree_production_keys(
                t,
                strip_terminals=strip_terminals_for_rules,
                include_lexical=include_lexical_rules,
            )
            for k in keys:
                used_prod_set.add(k)
                prod_counter[k] += 1

    parse_rate = (num_parsable / num_total) if num_total else 0.0
    rule_coverage = len(used_prod_set) / len(grammar_prod_set)

    total_rule_uses = sum(prod_counter.values())
    if total_rule_uses == 0:
        rule_entropy_nats = 0.0
    else:
        rule_entropy_nats = 0.0
        for c in prod_counter.values():
            p = c / total_rule_uses
            rule_entropy_nats -= p * math.log(p)
    rule_entropy_bits = (
        rule_entropy_nats / math.log(2.0) if rule_entropy_nats > 0 else 0.0
    )

    template_diversity = (len(template_set) / num_parsable) if num_parsable else 0.0
    avg_num_parses = (total_num_parses / num_parsable) if num_parsable else 0.0
    frac_ambiguous = (num_ambiguous / num_parsable) if num_parsable else 0.0

    metrics = SyntacticMetrics(
        num_total=num_total,
        num_parsable=num_parsable,
        parse_rate=parse_rate,
        rule_coverage=rule_coverage,
        rule_entropy_nats=rule_entropy_nats,
        rule_entropy_bits=rule_entropy_bits,
        template_diversity=template_diversity,
        avg_num_parses=avg_num_parses,
        frac_ambiguous=frac_ambiguous,
    )
    if return_per_sample:
        return metrics, per_sample
    return metrics


# ----------------------------
# Text preprocessing for post-hoc evaluator
# ----------------------------

# Common special token strings to strip from decoded text
_DEFAULT_SPECIAL_TOKENS = [
    "[PAD]",
    "[BOS]",
    "[EOS]",
    "[MASK]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "<|pad|>",
    "<|endoftext|>",
]


def _strip_special_tokens(text: str, tokenizer: Any = None) -> str:
    """Remove special token strings from decoded text for CFG parsing."""
    to_strip = list(_DEFAULT_SPECIAL_TOKENS)
    if tokenizer is not None:
        for attr in ("pad_token", "bos_token", "eos_token", "mask_token", "unk_token", "cls_token", "sep_token"):
            t = getattr(tokenizer, attr, None)
            if t is not None and t not in to_strip:
                to_strip.append(t)
    result = text
    for tok in to_strip:
        if tok:
            result = result.replace(tok, " ")
    return " ".join(result.split())


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
            trees = _parse_tokens(parser, token_seq, max_parses=1)
            values.append(1.0 if trees else 0.0)
        except Exception:
            values.append(0.0)
    return {"value": torch.tensor(values, device=loss_dict["ids"].device)}


def create_tokenizer_cfg_small(**kwargs):  # **kwargs for Hydra config merge compatibility
    # fmt: off
    vocab = ['small', 'happy', '`s', 'John', 'Alice', 'said', 'Mary', 'thought', 'laughed', 'bought', 'your', 'man', 'my', 'telescope', 'chased', 'elephant', 'that', 'sad', 'big', 'the', 'sneezed']
    # fmt: on
    tokenizer = SimpleSpaceTokenizer(vocab)
    return tokenizer


# ----------------------------
# Post-hoc evaluator for syntactic metrics
# ----------------------------


class SyntacticMetricsEvaluator:
    """Post-hoc evaluator for CFG syntactic metrics on logged predictions.

    Computes parse_rate, rule_coverage, rule_entropy, template_diversity, etc.
    on decoded text from JSONL. 
    """

    def __init__(
        self,
        grammar_str: Optional[str] = None,
        max_parses_per_sent: int = 50,
        parse_mode: str = "first",
        include_lexical_rules: bool = False,
        strip_terminals_for_rules: bool = True,
        text_field: str = "text",
    ):
        self.grammar_str = grammar_str if grammar_str is not None else G1_str
        self.max_parses_per_sent = max_parses_per_sent
        self.parse_mode = parse_mode
        self.include_lexical_rules = include_lexical_rules
        self.strip_terminals_for_rules = strip_terminals_for_rules
        self.text_field = text_field

    def eval(
        self,
        predictions: List[Dict[str, Any]],
        tokenizer: Any = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Evaluate predictions and return updated predictions + aggregated metrics."""
        if not predictions:
            return predictions, {}

        # 1. Extract and preprocess sentences (strip special tokens)
        sentences = []
        for pred in predictions:
            text = pred.get(self.text_field, "")
            cleaned = _strip_special_tokens(text, tokenizer)
            sentences.append(cleaned)

        def tokenize_fn(text: str) -> List[str]:
            return text.split() if text else []

        # 2. Compute metrics with per-sample data
        result = compute_syntactic_metrics(
            sentences,
            self.grammar_str,
            tokenizer=tokenize_fn,
            max_parses_per_sent=self.max_parses_per_sent,
            parse_mode=self.parse_mode,
            include_lexical_rules=self.include_lexical_rules,
            strip_terminals_for_rules=self.strip_terminals_for_rules,
            return_per_sample=True,
        )
        metrics, per_sample = result

        # 3. Add per-sample fields to predictions
        for pred, (parsable_val, template_sig) in zip(predictions, per_sample):
            pred["parsable"] = parsable_val
            pred["template_signature"] = template_sig

        # 4. Build aggregated metrics dict with syntactic/ prefix
        aggregated_metrics = {
            "syntactic/parse_rate": metrics.parse_rate,
            "syntactic/rule_coverage": metrics.rule_coverage,
            "syntactic/rule_entropy_nats": metrics.rule_entropy_nats,
            "syntactic/rule_entropy_bits": metrics.rule_entropy_bits,
            "syntactic/template_diversity": metrics.template_diversity,
            "syntactic/avg_num_parses": metrics.avg_num_parses,
            "syntactic/frac_ambiguous": metrics.frac_ambiguous,
        }

        return predictions, aggregated_metrics