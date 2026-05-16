"""Preprocessing for brozonoyer/sapientinc-sudoku-extreme-timvink-sudoku-solver.

Dataset has "question" (puzzle, "." for blanks) and "answer" (solution).
We convert "." -> "0" to match the tokenizer convention (vocab 0-9) and
produce input_token_ids / prompt_token_ids like the standard sudoku task.
"""

from typing import (
    Any,
    Dict,
    List,
)

from xlm.datamodule import SimpleSpaceTokenizer
from xlm.utils.rank_zero import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)
ranked_logger = RankedLogger(__name__, rank_zero_only=False)


def _replace(tokens: List[int], zero_id: int, mask_id: int) -> List[int]:
    return [mask_id if c == zero_id else c for c in tokens]


def _normalize_dots(s: str) -> str:
    """Convert '.' (blank) to '0' for tokenizer compatibility."""
    return s.replace(".", "0")


def sudoku_extreme_preprocess_fn(
    example: Dict[str, Any],
    tokenizer: SimpleSpaceTokenizer,
) -> Dict[str, Any]:
    """Preprocess sapientinc-sudoku-extreme examples.

    Uses "question" (puzzle) and "answer" (solution). Blanks are "." in the
    dataset; we convert to "0" before tokenizing.
    
    Also processes "trajectory" field which contains a list of strings
    representing step-by-step board configurations from question to solution.
    """
    partial_sequence: str = _normalize_dots(example["question"])
    ground_truth_sequence: str = _normalize_dots(example["answer"])
    prompt_ids: List[int] = [
        tokenizer._convert_token_to_id(str(ch)) for ch in partial_sequence
    ]
    zero_id = tokenizer._convert_token_to_id("0")
    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        raise ValueError("Mask token not found in tokenizer")
    prompt_ids = _replace(prompt_ids, zero_id, mask_id)
    input_ids = [
        tokenizer._convert_token_to_id(str(ch)) for ch in ground_truth_sequence
    ]
    input_ids = _replace(input_ids, zero_id, mask_id)
    example["input_token_ids"] = input_ids
    example["prompt_token_ids"] = prompt_ids
    
    # Process trajectory if present
    if "trajectory" in example and example["trajectory"] is not None:
        trajectory: List[str] = example["trajectory"]
        trajectory_token_ids: List[List[int]] = []
        for step in trajectory:
            normalized_step: str = _normalize_dots(step)
            step_token_ids: List[int] = [
                tokenizer._convert_token_to_id(str(ch)) for ch in normalized_step
            ]
            step_token_ids = _replace(step_token_ids, zero_id, mask_id)
            trajectory_token_ids.append(step_token_ids)
        example["trajectory_token_ids"] = trajectory_token_ids
    
    return example


def sudoku_extreme_kaggle_filter_fn(example: Dict[str, Any]) -> bool:
    return example["source"] == "puzzles0_kaggle"


# ---------------------------------------------------------------------------
# Hard-tier filters for the Sudoku qualitative study.
#
# Used as ``datamodule.dataset_managers.val.infill_prediction.filter_fn=...`` to
# bias the dump pool toward the long tail of difficulty without dumping the
# whole 422k-puzzle test split. The filters live here (rather than in
# ``doublebackprop``) so that they're discoverable from
# ``xlm.utils.module_loading.get_function`` like the existing ``kaggle`` filter.
# ---------------------------------------------------------------------------

# Strategies above the "single" / "candidate-line" baseline. Any of these
# appearing in ``strategies_used`` means the timvink solver had to do real
# heuristic work; if "Brute Force" is also present the puzzle additionally
# required search/lookahead.
_NON_TRIVIAL_STRATEGIES: frozenset[str] = frozenset({
    # Advanced
    "Naked Pair", "Naked Pairs",
    "Naked Triple", "Naked Triples",
    "Naked Quad", "Naked Quads",
    "Hidden Pair", "Hidden Pairs",
    "Hidden Triple", "Hidden Triples",
    "Hidden Quad", "Hidden Quads",
    # Master
    "X-Wing", "X-Wings",
    "Swordfish",
    "Jellyfish", "Jellyfishes",
    "Forcing Chain", "Forcing Chains",
})


def _strategies_set(example: Dict[str, Any]) -> set:
    s = example.get("strategies_used")
    if s is None:
        return set()
    try:
        return {str(x).strip() for x in s}
    except TypeError:
        return set()


def sudoku_extreme_hard_filter_fn(example: Dict[str, Any]) -> bool:
    """Default 'hard' filter: keep puzzles in the top decile by difficulty.

    A puzzle qualifies as hard if any of:

    - ``rating >= 50``: timvink solver's tiered difficulty score (p90 of the
      test split is ~51, p95 is ~64). Catches the heavy brute-force tail.
    - ``num_steps >= 8``: long heuristic chains (p90 of the test split is 7).
    - any strategy in ``_NON_TRIVIAL_STRATEGIES``: Advanced or Master tier in
      the corrected tier mapping (catches the ~6k Advanced + ~225 Master
      puzzles that were silently bucketed as BruteForce in the original sweep).

    Together these cover ~13% of the test split (vs. ~1.5% for tier-only and
    ~10% for rating-only); good for a dump pool that's small but not trivial.
    """
    rating = example.get("rating")
    if rating is not None and rating >= 50:
        return True
    num_steps = example.get("num_steps")
    if num_steps is not None and num_steps >= 8:
        return True
    if _strategies_set(example) & _NON_TRIVIAL_STRATEGIES:
        return True
    return False


def sudoku_extreme_extreme_filter_fn(example: Dict[str, Any]) -> bool:
    """'Extreme' filter: top ~1% of the test split by rating *or* tier.

    A puzzle qualifies as extreme if any of:

    - ``rating >= 100``: ~1% of the split (p99 = 100).
    - ``num_steps >= 12``: ~1% (p99 = 12).
    - any Master-tier strategy (X-Wing / Swordfish / Jellyfish / Forcing Chain).

    Designed for a small but pure hard slice when the goal is to localize
    where loopholing+BPTT actually starts to separate from StopGrad.
    """
    _MASTER = frozenset({
        "X-Wing", "X-Wings",
        "Swordfish",
        "Jellyfish", "Jellyfishes",
        "Forcing Chain", "Forcing Chains",
    })
    rating = example.get("rating")
    if rating is not None and rating >= 100:
        return True
    num_steps = example.get("num_steps")
    if num_steps is not None and num_steps >= 12:
        return True
    if _strategies_set(example) & _MASTER:
        return True
    return False


def sudoku_extreme_deduction_only_filter_fn(example: Dict[str, Any]) -> bool:
    """'Deduction-only hard' filter: Advanced/Master strategies *and* no Brute Force.

    A puzzle qualifies if it satisfies BOTH:

    1. ``"Brute Force"`` is **not** in ``strategies_used`` (excludes the ~358k
       BruteForce-tier puzzles entirely; we don't expect a forward-only diffusion
       model to do recursive backtracking).
    2. ``strategies_used`` contains at least one Advanced or Master tier strategy
       (Naked/Hidden Pairs/Triples/Quads, X-Wing/Swordfish/Jellyfish/Forcing Chain).

    This is the "real difficulty axis" cohort: ~6,190 puzzles in the test split
    (5,965 Advanced + 225 Master), so a 2,000-puzzle uniform-shuffled slice will
    contain ~1,930 Advanced + ~70 Master in expectation -- enough power to put a
    tight CI on the BPTT-vs-StopGrad paired delta separately for each tier.

    Sister filters (`sudoku_extreme_hard_filter_fn`, `_extreme_filter_fn`) use
    OR over rating/num_steps/strategy, which pulls in lots of long-but-easy
    BruteForce-tail puzzles. This one uses AND so the cohort is *only* the
    deduction-needed puzzles.
    """
    s = _strategies_set(example)
    if "Brute Force" in s:
        return False
    return bool(s & _NON_TRIVIAL_STRATEGIES)
