# Portions of this file are derived from the GenMol project (https://github.com/NVIDIA-Digital-Bio/genmol)
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES;
# Modifications Copyright (c) 2026 Dhruvesh Patel
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Molecule generation task utilities and metrics.

This module provides:
- Data preprocessing for SAFE molecular representations
- Conversion utilities between SAFE and SMILES formats
- Comprehensive metrics for evaluating molecular generation (diversity, QED, SA, validity, uniqueness)
"""

import re
from typing import Any, Dict, List, Optional, Tuple
import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase
from torchmetrics import Metric

from xlm.utils.rank_zero import warn_once, RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)

# Import SAFE library for molecular encoding/decoding
try:
    import safe as sf
    from safe.tokenizer import SAFETokenizer
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
except ImportError:
    raise ImportError(
        "Please install safe-mol and rdkit: pip install safe-mol rdkit"
    )

# Import TDC for molecular metrics
try:
    from tdc import Oracle, Evaluator
except ImportError:
    raise ImportError("Please install TDC: pip install pytdc")


################################################################################
# region: SAFE String Conversion Utilities (reimplemented from GenMol)


def safe2bracketsafe(safe_str: str) -> str:
    """Convert standard SAFE notation to bracket SAFE format.

    Bracket SAFE wraps interfragment attachment points in angle brackets.
    Example: "1" -> "<1>", "%10" -> "<%10>"

    Based on genmol/src/genmol/utils/bracket_safe_converter.py:133-137

    Args:
        safe_str: SAFE string in standard notation

    Returns:
        SAFE string in bracket notation, or original string if conversion fails
    """
    try:
        # Convert SAFE to molecule
        mol = Chem.MolFromSmiles(safe_str)
        if mol is None:
            return safe_str

        # Use SAFE encoder with custom bracket notation
        from safe.converter import SAFEConverter

        # Create a simple converter
        converter = SAFEConverter()
        encoded = converter.encoder(
            mol, canonical=False, randomize=True, allow_empty=True
        )

        # Find all attachment points in the encoded string
        # Pattern matches: digits not preceded by % and not followed by >, or %\d+ patterns
        attach_points = set(re.findall(r"(?<!%)(\d+)(?!>)", encoded))
        attach_points.update(re.findall(r"(%\d+)", encoded))

        # Sort for canonical ordering
        attach_points = sorted(attach_points)

        # Replace each attachment point with bracketed version
        for attach in attach_points:
            # Wrap in angle brackets
            bracketed = f"<{attach}>"
            # Use word boundary to avoid replacing parts of other numbers
            encoded = re.sub(
                f"(?<!<){re.escape(attach)}(?!>)", bracketed, encoded
            )

        return encoded

    except Exception:
        # On any error, return original string
        return safe_str


def bracketsafe2safe(safe_str: str) -> str:
    """Convert bracket SAFE notation back to standard SAFE format.

    Removes angle brackets from interfragment attachment points and renumbers
    them to avoid conflicts with intrafragment attachment points.

    Based on genmol/src/genmol/utils/bracket_safe_converter.py:140-153

    Args:
        safe_str: SAFE string in bracket notation

    Returns:
        SAFE string in standard notation
    """
    try:
        # Find all intrafragment attachment points (not in brackets)
        # These are digits not preceded by < and not preceded by %
        intrafrag_points = [
            m.group(0)
            for m in re.finditer(r"(?<!<)(?<!%)(\d+)(?!>)", safe_str)
        ]
        # Also get %\d+ patterns not in brackets
        intrafrag_points.extend(
            [
                m.group(0).lstrip("%")
                for m in re.finditer(r"(?<!<)(%\d+)(?!>)", safe_str)
            ]
        )

        # Find maximum intrafragment number to avoid conflicts
        starting_num = (
            max([int(i) for i in intrafrag_points]) + 1
            if intrafrag_points
            else 0
        )

        # Find all interfragment points (in brackets)
        interfrag_matches = [
            (m.start(0), m.end(0), m.group(1))
            for m in re.finditer(r"<(\d+|%\d+)>", safe_str)
        ]

        # Convert to list for character-by-character replacement
        safe_list = list(safe_str)

        # Replace brackets and renumber
        for start, end, number in reversed(
            interfrag_matches
        ):  # Reverse to maintain indices
            # Remove % if present
            num_val = number.lstrip("%") if number.startswith("%") else number
            # Calculate new number
            new_num = int(num_val) + starting_num
            # Format with % if >= 10
            new_num_str = f"%{new_num}" if new_num >= 10 else str(new_num)

            # Replace brackets and content
            safe_list[start] = ""  # Remove <
            safe_list[end - 1] = ""  # Remove >
            # Replace the number itself
            safe_list[start + 1 : end - 1] = list(new_num_str) + [""] * (
                end - start - len(new_num_str) - 2
            )

        # Join and remove empty strings
        result = "".join(safe_list).replace("", "")
        return re.sub(r"\s+", "", result)  # Remove any whitespace

    except Exception:
        # On error, try simple bracket removal
        return safe_str.replace("<", "").replace(">", "")


def safe_to_smiles(safe_str: str, fix: bool = True) -> Optional[str]:
    """Convert SAFE string to SMILES using safe library.

    Based on genmol/src/genmol/utils/utils_chem.py:26-30

    Args:
        safe_str: SAFE molecular representation
        fix: If True, filter out invalid fragments before decoding

    Returns:
        SMILES string, or None if conversion fails
    """
    try:
        if fix:
            # Filter out fragments that can't be decoded
            valid_fragments = []
            for frag in safe_str.split("."):
                if sf.decode(frag, ignore_errors=True) is not None:
                    valid_fragments.append(frag)
            safe_str = ".".join(valid_fragments)

        # Decode SAFE to SMILES
        smiles = sf.decode(safe_str, canonical=True, ignore_errors=True)
        return smiles

    except Exception:
        return None


def safe_strings_to_smiles(
    safe_strings: List[str], use_bracket_safe: bool = False, fix: bool = True
) -> List[str]:
    """Convert batch of SAFE strings to SMILES strings.

    Based on genmol/src/genmol/sampler.py:81-89

    Args:
        safe_strings: List of SAFE molecular representations
        use_bracket_safe: If True, convert from bracket SAFE first
        fix: If True, filter invalid fragments

    Returns:
        List of SMILES strings (invalid conversions are skipped)
    """
    smiles_list = []

    for safe_str in safe_strings:
        # Convert from bracket SAFE if needed
        if use_bracket_safe:
            safe_str = bracketsafe2safe(safe_str)

        # Convert to SMILES
        smiles = safe_to_smiles(safe_str, fix=fix)

        if smiles:
            # Take largest fragment (removes salts, counter-ions, etc.)
            fragments = smiles.split(".")
            largest_fragment = sorted(fragments, key=len)[-1]
            smiles_list.append(largest_fragment)

    return smiles_list


# endregion: SAFE String Conversion Utilities
################################################################################

################################################################################
# region: Data Preprocessing


class SerializableSAFETokenizer:
    """Wrapper around SAFE tokenizer that handles pickling/deepcopy.

    The underlying tokenizer from the safe library has a custom PreTokenizer
    that cannot be serialized. This wrapper provides dummy serialization by
    storing the model path and re-instantiating the tokenizer on unpickle.
    """

    def __init__(
        self, pretrained_model_name_or_path: str = "datamol-io/safe-gpt"
    ):
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self._tokenizer = None
        self._initialize_tokenizer()

    def _initialize_tokenizer(self):
        """Initialize or re-initialize the tokenizer."""
        self._tokenizer = SAFETokenizer.from_pretrained(
            self.pretrained_model_name_or_path
        ).get_pretrained()
        self._tokenizer.add_tokens(["<", ">"])  # for bracket_safe
        self._tokenizer.full_vocab_size = (
            self._tokenizer.vocab_size + 2
        )  # +2 for < and >

    def __getstate__(self):
        """Return state for pickling - only store the model path."""
        return {
            "pretrained_model_name_or_path": self.pretrained_model_name_or_path
        }

    def __setstate__(self, state):
        """Restore from pickled state - re-instantiate tokenizer."""
        self.pretrained_model_name_or_path = state[
            "pretrained_model_name_or_path"
        ]
        self._tokenizer = None
        self._initialize_tokenizer()

    def __getattr__(self, name):
        """Delegate all attribute access to the underlying tokenizer."""
        if name in [
            "_tokenizer",
            "pretrained_model_name_or_path",
            "_initialize_tokenizer",
        ]:
            return object.__getattribute__(self, name)
        return getattr(self._tokenizer, name)

    def __dir__(self):
        """Include both wrapper and tokenizer attributes in dir()."""
        return list(set(dir(self.__class__) + dir(self._tokenizer)))

    def len(self):
        return self._tokenizer.full_vocab_size


def get_safe_tokenizer(
    pretrained_model_name_or_path: str = "datamol-io/safe-gpt",
):
    tok = SerializableSAFETokenizer(pretrained_model_name_or_path)
    return tok


def _safe_preprocess_fn(
    example: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    use_bracket_safe: bool = False,
) -> Dict[str, Any]:
    """Preprocess SAFE molecular data for training.

    Args:
        example: Dataset example containing 'text' or 'safe' field
        tokenizer: Tokenizer for encoding
        use_bracket_safe: If True, convert to bracket SAFE notation

    Returns:
        Example with 'token_ids' field added
    """
    # Get SAFE string from example (support both 'text' and 'safe' keys)
    text = example.get("safe")

    # Convert to bracket SAFE if requested
    if use_bracket_safe:
        text = safe2bracketsafe(text)

    # Tokenize
    example["token_ids"] = tokenizer.encode(text, add_special_tokens=False)

    return example


def safe_bracket_preprocess_fn(
    example: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
) -> Dict[str, Any]:
    return _safe_preprocess_fn(example, tokenizer, use_bracket_safe=True)


def safe_preprocess_fn(
    example: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
) -> Dict[str, Any]:
    return _safe_preprocess_fn(example, tokenizer, use_bracket_safe=False)


def safe_bracket_on_the_fly_processor_combined(
    example: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    block_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Works directly on the raw strings"""
    example = _safe_preprocess_fn(example, tokenizer, use_bracket_safe=True)
    if block_size is None:
        block_size = 256
        warn_once(
            logger, f"No block_size provided, using default: {block_size}"
        )
    res = {"input_ids": example["token_ids"][:block_size]}
    return res


def genmol_fragment_preprocess_fn(
    example: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    *,
    fragment_column: str = "linker_design",
) -> Dict[str, Any]:
    """Preprocess GenMol fragment CSV data for fragment-constrained generation.

    Converts SMILES fragments and targets to SAFE format, then to bracket SAFE,
    and creates prompt_token_ids (fragment) and input_token_ids (full molecule).

    Based on GenMol's fragment evaluation dataset structure:
    - Input: Fragment SMILES (from `fragment_column`, default 'linker_design')
    - Target: Full molecule SMILES (from 'smiles' column)

    To use a different fragment column, pass `fragment_column` via
    `preprocess_function_kwargs` in the dataset config, or set
    `_fragment_column` in the example dict (overrides kwarg).

    Args:
        example: Dataset example containing:
            - column named by `fragment_column`: SMILES with [n*] attachment points
            - 'smiles': Full target molecule SMILES
            - '_fragment_column' (optional): Overrides `fragment_column` if set
        tokenizer: Tokenizer for encoding
        fragment_column: CSV column to use as fragment input (default: 'linker_design').
            Override via datamodule ... preprocess_function_kwargs.fragment_column.

    Returns:
        Example with 'prompt_token_ids' (fragment) and 'input_token_ids' (full molecule)
    """
    # Per-example override wins over preprocess_function_kwargs
    fragment_column = example.get("_fragment_column", fragment_column)
    
    # Get fragment SMILES (input)
    fragment_smiles = example.get(fragment_column)
    if fragment_smiles is None or not fragment_smiles:
        raise ValueError(f"Example must contain '{fragment_column}' field with fragment SMILES")
    
    # Get target SMILES (full molecule)
    target_smiles = example.get("smiles")
    if target_smiles is None:
        raise ValueError("Example must contain 'smiles' field with target molecule SMILES")
    
    # Convert SMILES to SAFE format
    try:
        # Convert fragment SMILES to SAFE
        # Handle multiple fragments separated by '.' (e.g., linker_design has two fragments)
        fragment_safe_parts = []
        for frag_smiles in fragment_smiles.split("."):
            frag_smiles = frag_smiles.strip()
            if not frag_smiles:
                continue
            # Convert SMILES to molecule, then to SAFE
            mol = Chem.MolFromSmiles(frag_smiles)
            if mol is None:
                logger.warning(f"Failed to parse fragment SMILES: {frag_smiles}")
                continue
            # Use SAFE encoder to convert molecule to SAFE string
            from safe.converter import SAFEConverter
            converter = SAFEConverter()
            frag_safe = converter.encoder(mol, canonical=False, randomize=True, allow_empty=True)
            fragment_safe_parts.append(frag_safe)
        fragment_safe = ".".join(fragment_safe_parts)
        
        # Convert target SMILES to SAFE
        target_mol = Chem.MolFromSmiles(target_smiles)
        if target_mol is None:
            raise ValueError(f"Failed to parse target SMILES: {target_smiles}")
        from safe.converter import SAFEConverter
        converter = SAFEConverter()
        target_safe = converter.encoder(target_mol, canonical=False, randomize=True, allow_empty=True)
        
    except Exception as e:
        raise ValueError(f"Failed to convert SMILES to SAFE: {e}")
    
    # Convert to bracket SAFE
    fragment_bracket_safe = safe2bracketsafe(fragment_safe)
    target_bracket_safe = safe2bracketsafe(target_safe)
    
    # Tokenize
    fragment_token_ids = tokenizer.encode(fragment_bracket_safe, add_special_tokens=False)
    target_token_ids = tokenizer.encode(target_bracket_safe, add_special_tokens=False)
    
    example["prompt_token_ids"] = fragment_token_ids
    example["input_token_ids"] = target_token_ids
    
    return example


# endregion: Data Preprocessing
################################################################################


################################################################################
# region: Post-Hoc Evaluators


class DeNovoEval:
    """Post-hoc evaluator for de novo molecule generation.

    Computes molecular properties on logged predictions at epoch end, matching
    GenMol's evaluation semantics. Computes:
    - Per-sample: QED, SA, SMILES (added to each prediction dict)
    - Global: Diversity, Validity, Uniqueness (aggregated across all samples)

    This approach enables:
    - Global metric computation (diversity/uniqueness on full generated set)
    - Exact match with GenMol's evaluation methodology
    - Reusable components for other tasks (frag, lead, pmo)

    Args:
        use_bracket_safe: If True, decode from bracket SAFE format
        compute_diversity: If True, compute diversity metric
        compute_validity: If True, compute validity metric
        compute_uniqueness: If True, compute uniqueness metric
        compute_qed: If True, compute QED scores
        compute_sa: If True, compute SA scores
    """

    def __init__(
        self,
        use_bracket_safe: bool = False,
        compute_diversity: bool = True,
        compute_validity: bool = True,
        compute_uniqueness: bool = True,
        compute_qed: bool = True,
        compute_sa: bool = True,
    ):
        self.use_bracket_safe = use_bracket_safe
        self.compute_diversity = compute_diversity
        self.compute_validity = compute_validity
        self.compute_uniqueness = compute_uniqueness
        self.compute_qed = compute_qed
        self.compute_sa = compute_sa

        # Lazy-loaded TDC oracles (to avoid import overhead)
        self._oracle_qed = None
        self._oracle_sa = None
        self._evaluator_diversity = None
        self._evaluator_validity = None
        self._evaluator_uniqueness = None

    @property
    def oracle_qed(self):
        """Lazy load QED oracle."""
        if self._oracle_qed is None and self.compute_qed:
            self._oracle_qed = Oracle("qed")
        return self._oracle_qed

    @property
    def oracle_sa(self):
        """Lazy load SA oracle."""
        if self._oracle_sa is None and self.compute_sa:
            self._oracle_sa = Oracle("sa")
        return self._oracle_sa

    @property
    def evaluator_diversity(self):
        """Lazy load diversity evaluator."""
        if self._evaluator_diversity is None and self.compute_diversity:
            self._evaluator_diversity = Evaluator("diversity")
        return self._evaluator_diversity

    @property
    def evaluator_validity(self):
        """Lazy load validity evaluator."""
        if self._evaluator_validity is None and self.compute_validity:
            self._evaluator_validity = Evaluator("validity")
        return self._evaluator_validity

    @property
    def evaluator_uniqueness(self):
        """Lazy load uniqueness evaluator."""
        if self._evaluator_uniqueness is None and self.compute_uniqueness:
            self._evaluator_uniqueness = Evaluator("uniqueness")
        return self._evaluator_uniqueness

    def eval(
        self,
        predictions: List[Dict[str, Any]],
        tokenizer: Any = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Evaluate predictions and return updated predictions + aggregated metrics.

        Args:
            predictions: List of prediction dicts with 'text' field containing SAFE strings
            tokenizer: Optional tokenizer (not used for denovo, but kept for interface consistency)

        Returns:
            Tuple of:
            - predictions: Updated list with per-sample metrics added (smiles, qed, sa)
            - aggregated_metrics: Dict of global metric values
        """
        if not predictions:
            return predictions, {}

        # 1. Convert SAFE to SMILES for all predictions
        all_smiles = []
        for pred in predictions:
            safe_str = pred.get("text", "")
            smiles = self._safe_to_smiles_with_bracket_handling(safe_str)
            pred["smiles"] = smiles  # Add SMILES to prediction dict
            if (
                smiles is not None
            ):  # Explicitly check for None (invalid conversions)
                all_smiles.append(smiles)

        # 2. Compute per-sample metrics
        if self.compute_qed:
            self._compute_qed_per_sample(predictions)

        if self.compute_sa:
            self._compute_sa_per_sample(predictions)

        # 3. Compute global aggregated metrics
        aggregated_metrics = {}

        total_generated = len(predictions)

        if all_smiles:
            if self.compute_diversity:
                aggregated_metrics.update(self._compute_diversity(all_smiles))

            if self.compute_validity:
                aggregated_metrics.update(
                    self._compute_validity(all_smiles, total_generated)
                )

            if self.compute_uniqueness:
                aggregated_metrics.update(self._compute_uniqueness(all_smiles))

            # Aggregate QED/SA statistics
            if self.compute_qed:
                aggregated_metrics.update(
                    self._aggregate_qed_stats(predictions)
                )

            if self.compute_sa:
                aggregated_metrics.update(
                    self._aggregate_sa_stats(predictions)
                )

            # Compute quality: fraction with QED >= 0.6 AND SA <= 4
            if self.compute_qed and self.compute_sa:
                aggregated_metrics.update(
                    self._compute_quality(predictions, total_generated)
                )

        return predictions, aggregated_metrics

    # -------------------------------------------------------------------------
    # Helper Methods (Reusable for other GenMol tasks: frag, lead, pmo)
    # -------------------------------------------------------------------------

    def _safe_to_smiles_with_bracket_handling(
        self, safe_str: str
    ) -> Optional[str]:
        """Convert SAFE to SMILES, handling bracket notation.

        Reusable helper for all GenMol tasks.

        Args:
            safe_str: SAFE string representation

        Returns:
            SMILES string (largest fragment), or None if conversion fails
        """
        if not safe_str:
            return None

        # Handle bracket SAFE notation if needed
        if self.use_bracket_safe:
            safe_str = bracketsafe2safe(safe_str)

        # Convert to SMILES
        smiles = safe_to_smiles(safe_str, fix=True)

        if (
            smiles is not None
        ):  # Explicitly check for None (invalid conversions)
            # Take largest fragment (removes salts, counter-ions, etc.)
            fragments = smiles.split(".")
            return sorted(fragments, key=len)[-1]

        return None

    def _compute_qed_per_sample(
        self, predictions: List[Dict[str, Any]]
    ) -> None:
        """Compute QED for each prediction and add to dict.

        Reusable for frag/lead/pmo tasks.

        Args:
            predictions: List of prediction dicts (modified in-place)
        """
        for pred in predictions:
            smiles = pred.get("smiles")
            if smiles is not None:  # Only compute for valid SMILES
                try:
                    qed_val = self.oracle_qed(smiles)
                    pred["qed"] = (
                        float(qed_val) if qed_val is not None else None
                    )
                except Exception:
                    pred["qed"] = None
            else:
                pred["qed"] = None

    def _compute_sa_per_sample(
        self, predictions: List[Dict[str, Any]]
    ) -> None:
        """Compute SA for each prediction and add to dict.

        Reusable for frag/lead/pmo tasks.

        Args:
            predictions: List of prediction dicts (modified in-place)
        """
        for pred in predictions:
            smiles = pred.get("smiles")
            if smiles is not None:  # Only compute for valid SMILES
                try:
                    sa_val = self.oracle_sa(smiles)
                    pred["sa"] = float(sa_val) if sa_val is not None else None
                except Exception:
                    pred["sa"] = None
            else:
                pred["sa"] = None

    def _compute_diversity(self, smiles_list: List[str]) -> Dict[str, float]:
        """Compute diversity on full SMILES list.

        Diversity is the average pairwise Tanimoto distance between
        Morgan fingerprints. Computed globally on all unique molecules.

        Reusable for frag/lead/pmo tasks.

        Args:
            smiles_list: List of all SMILES strings

        Returns:
            Dict with 'diversity' key
        """
        unique_smiles = list(set(smiles_list))

        # Need at least 2 molecules for pairwise comparison
        if len(unique_smiles) >= 2:
            try:
                diversity_val = self.evaluator_diversity(unique_smiles)
                return {"diversity": float(diversity_val)}
            except Exception:
                pass

        return {"diversity": 0.0}

    def _compute_validity(
        self, smiles_list: List[str], total_generated: int
    ) -> Dict[str, float]:
        """Compute validity fraction.

        Validity = (# valid SMILES) / (# total generated)

        Reusable for frag/lead/pmo tasks.

        Args:
            smiles_list: List of valid SMILES strings
            total_generated: Total number of generations attempted

        Returns:
            Dict with 'validity' key
        """
        validity = (
            len(smiles_list) / total_generated if total_generated > 0 else 0.0
        )
        return {"validity": float(validity)}

    def _compute_uniqueness(self, smiles_list: List[str]) -> Dict[str, float]:
        """Compute uniqueness fraction.

        Uniqueness = (# unique SMILES) / (# total valid SMILES)

        Reusable for frag/lead/pmo tasks.

        Args:
            smiles_list: List of valid SMILES strings

        Returns:
            Dict with 'uniqueness' key
        """
        unique_count = len(set(smiles_list))
        total_count = len(smiles_list)
        uniqueness = unique_count / total_count if total_count > 0 else 0.0
        return {"uniqueness": float(uniqueness)}

    def _aggregate_qed_stats(
        self, predictions: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Aggregate QED statistics (mean, std, min, max).

        Reusable for frag/lead/pmo tasks.

        Args:
            predictions: List of prediction dicts with 'qed' field

        Returns:
            Dict with qed_mean, qed_std, qed_min, qed_max keys
        """
        qed_values = [
            p["qed"] for p in predictions if p.get("qed") is not None
        ]

        if not qed_values:
            return {
                "qed_mean": 0.0,
                "qed_std": 0.0,
                "qed_min": 0.0,
                "qed_max": 0.0,
            }

        import numpy as np

        return {
            "qed_mean": float(np.mean(qed_values)),
            "qed_std": float(np.std(qed_values)),
            "qed_min": float(np.min(qed_values)),
            "qed_max": float(np.max(qed_values)),
        }

    def _aggregate_sa_stats(
        self, predictions: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Aggregate SA statistics (mean, std, min, max).

        Reusable for frag/lead/pmo tasks.

        Args:
            predictions: List of prediction dicts with 'sa' field

        Returns:
            Dict with sa_mean, sa_std, sa_min, sa_max keys
        """
        sa_values = [p["sa"] for p in predictions if p.get("sa") is not None]

        if not sa_values:
            return {
                "sa_mean": 0.0,
                "sa_std": 0.0,
                "sa_min": 0.0,
                "sa_max": 0.0,
            }

        import numpy as np

        return {
            "sa_mean": float(np.mean(sa_values)),
            "sa_std": float(np.std(sa_values)),
            "sa_min": float(np.min(sa_values)),
            "sa_max": float(np.max(sa_values)),
        }

    def _compute_quality(
        self, predictions: List[Dict[str, Any]], total_generated: int
    ) -> Dict[str, float]:
        """Compute quality: fraction with QED >= 0.6 AND SA <= 4.

        Quality measures molecules that are both drug-like (high QED)
        and synthetically accessible (low SA). This is the most valuable
        set of molecules for drug discovery.

        Following GenMol's denovo evaluation (lines 52-54).

        Reusable for frag/lead/pmo tasks.

        Args:
            predictions: List of prediction dicts with 'qed' and 'sa' fields
            total_generated: Total number of generations attempted

        Returns:
            Dict with 'quality' key
        """
        # Count molecules meeting both criteria
        quality_count = sum(
            1
            for p in predictions
            if p.get("qed") is not None
            and p.get("sa") is not None
            and p["qed"] >= 0.6
            and p["sa"] <= 4.0
        )

        quality = (
            quality_count / total_generated if total_generated > 0 else 0.0
        )
        return {"quality": float(quality)}


class FragmentEval(DeNovoEval):
    """Post-hoc evaluator for fragment-constrained molecule generation.

    Extends DeNovoEval with fragment-specific metrics:
    - All de novo metrics (validity, uniqueness, quality, QED, SA, diversity)
    - Distance: Tanimoto distance between generated and target molecules

    Based on GenMol's fragment evaluation methodology. Computes:
    - Per-sample: QED, SA, SMILES, distance (if target available)
    - Global: Diversity, Validity, Uniqueness, Quality, Distance (mean)

    Args:
        use_bracket_safe: If True, decode from bracket SAFE format
        compute_diversity: If True, compute diversity metric
        compute_validity: If True, compute validity metric
        compute_uniqueness: If True, compute uniqueness metric
        compute_qed: If True, compute QED scores
        compute_sa: If True, compute SA scores
        compute_distance: If True, compute Tanimoto distance to target
    """

    def __init__(
        self,
        use_bracket_safe: bool = False,
        compute_diversity: bool = True,
        compute_validity: bool = True,
        compute_uniqueness: bool = True,
        compute_qed: bool = True,
        compute_sa: bool = True,
        compute_distance: bool = True,
    ):
        super().__init__(
            use_bracket_safe=use_bracket_safe,
            compute_diversity=compute_diversity,
            compute_validity=compute_validity,
            compute_uniqueness=compute_uniqueness,
            compute_qed=compute_qed,
            compute_sa=compute_sa,
        )
        self.compute_distance = compute_distance

    def eval(
        self,
        predictions: List[Dict[str, Any]],
        tokenizer: Any = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Evaluate fragment generation predictions.

        Args:
            predictions: List of prediction dicts with:
                - 'text': Generated SAFE string (full molecule)
                - 'truth': Target SAFE string (full molecule, optional)
                - 'raw_input': Fragment prompt SAFE string (optional)
            tokenizer: Optional tokenizer (not used, kept for interface consistency)

        Returns:
            Tuple of:
            - predictions: Updated list with per-sample metrics added
            - aggregated_metrics: Dict of global metric values
        """
        # First run all the de novo metrics
        predictions, aggregated_metrics = super().eval(predictions, tokenizer)

        if not predictions:
            return predictions, aggregated_metrics

        # Convert target SAFE to SMILES for distance computation
        target_smiles_list = []
        for pred in predictions:
            truth_safe = pred.get("truth", "")
            if truth_safe:
                truth_smiles = self._safe_to_smiles_with_bracket_handling(
                    truth_safe
                )
                pred["truth_smiles"] = truth_smiles
                if truth_smiles is not None:
                    target_smiles_list.append(truth_smiles)
            else:
                pred["truth_smiles"] = None

        # Compute per-sample distance if targets are available
        if self.compute_distance and target_smiles_list:
            self._compute_distance_per_sample(predictions)

            # Aggregate distance statistics
            aggregated_metrics.update(self._aggregate_distance_stats(predictions))

        return predictions, aggregated_metrics

    def _compute_distance_per_sample(
        self, predictions: List[Dict[str, Any]]
    ) -> None:
        """Compute Tanimoto distance for each prediction-target pair.

        Distance = 1 - Tanimoto similarity (using Morgan fingerprints).
        Lower distance means more similar to target.

        Args:
            predictions: List of prediction dicts (modified in-place)
        """
        from rdkit import DataStructs
        from rdkit.Chem import AllChem

        for pred in predictions:
            generated_smiles = pred.get("smiles")
            target_smiles = pred.get("truth_smiles")

            if generated_smiles is None or target_smiles is None:
                pred["distance"] = None
                continue

            try:
                # Generate molecules
                gen_mol = Chem.MolFromSmiles(generated_smiles)
                target_mol = Chem.MolFromSmiles(target_smiles)

                if gen_mol is None or target_mol is None:
                    pred["distance"] = None
                    continue

                # Compute Morgan fingerprints (radius=2, 2048 bits)
                gen_fp = AllChem.GetMorganFingerprintAsBitVect(
                    gen_mol, radius=2, nBits=2048
                )
                target_fp = AllChem.GetMorganFingerprintAsBitVect(
                    target_mol, radius=2, nBits=2048
                )

                # Compute Tanimoto similarity (0-1)
                similarity = DataStructs.TanimotoSimilarity(gen_fp, target_fp)

                # Distance = 1 - similarity (0 = identical, 1 = completely different)
                distance = 1.0 - similarity
                pred["distance"] = float(distance)

            except Exception:
                pred["distance"] = None

    def _aggregate_distance_stats(
        self, predictions: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Aggregate distance statistics (mean, std, min, max).

        Args:
            predictions: List of prediction dicts with 'distance' field

        Returns:
            Dict with distance_mean, distance_std, distance_min, distance_max keys
        """
        distance_values = [
            p["distance"]
            for p in predictions
            if p.get("distance") is not None
        ]

        if not distance_values:
            return {
                "distance_mean": 0.0,
                "distance_std": 0.0,
                "distance_min": 0.0,
                "distance_max": 0.0,
            }

        import numpy as np

        return {
            "distance_mean": float(np.mean(distance_values)),
            "distance_std": float(np.std(distance_values)),
            "distance_min": float(np.min(distance_values)),
            "distance_max": float(np.max(distance_values)),
        }


# endregion: Post-Hoc Evaluators
################################################################################


################################################################################
# region: DEPRECATED - TorchMetrics-based Implementation (kept for reference)
# NOTE: This approach doesn't match GenMol's evaluation semantics.
# Use DeNovoEval above for proper post-hoc evaluation.


class _MolGenMetric_DEPRECATED(Metric):
    """Comprehensive metric for molecule generation evaluation.

    Computes multiple molecular properties in a single pass:
    - Diversity: Average pairwise Tanimoto distance of Morgan fingerprints
    - QED: Quantitative Estimate of Drug-likeness (mean, std, min, max)
    - SA: Synthetic Accessibility score (mean, std, min, max)
    - Validity: Fraction of valid SMILES
    - Uniqueness: Fraction of unique molecules

    All metrics are computed using the TDC (Therapeutics Data Commons) library.
    States are properly synchronized across distributed processes.

    Args:
        use_bracket_safe: If True, decode from bracket SAFE format
        **kwargs: Additional arguments for Metric base class
    """

    # Metric properties
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = (
        None  # Mixed (higher for some, lower for SA)
    )
    full_state_update: bool = False
    # declarations for mypy
    diversity_sum: Tensor
    diversity_count: Tensor
    qed_sum: Tensor
    qed_sum_sq: Tensor
    qed_min: Tensor
    qed_max: Tensor
    qed_count: Tensor
    sa_sum: Tensor
    sa_sum_sq: Tensor
    sa_min: Tensor
    sa_max: Tensor
    sa_count: Tensor
    validity_sum: Tensor
    validity_count: Tensor
    uniqueness_sum: Tensor
    uniqueness_count: Tensor
    use_bracket_safe: bool

    def __init__(self, use_bracket_safe: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.use_bracket_safe = use_bracket_safe

        # Initialize TDC oracles (at initialization, not per update)
        self.evaluator_diversity = Evaluator("diversity")
        self.evaluator_validity = Evaluator("validity")
        self.evaluator_uniqueness = Evaluator("uniqueness")
        self.oracle_qed = Oracle("qed")
        self.oracle_sa = Oracle("sa")

        # States for diversity (collection-level metric)
        self.add_state(
            "diversity_sum", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state(
            "diversity_count", default=torch.tensor(0), dist_reduce_fx="sum"
        )

        # States for QED (per-sample metric with statistics)
        self.add_state(
            "qed_sum", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state(
            "qed_sum_sq", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )  # For std
        self.add_state(
            "qed_min", default=torch.tensor(float("inf")), dist_reduce_fx="min"
        )
        self.add_state(
            "qed_max",
            default=torch.tensor(float("-inf")),
            dist_reduce_fx="max",
        )
        self.add_state(
            "qed_count", default=torch.tensor(0), dist_reduce_fx="sum"
        )

        # States for SA (per-sample metric with statistics)
        self.add_state(
            "sa_sum", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state(
            "sa_sum_sq", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )  # For std
        self.add_state(
            "sa_min", default=torch.tensor(float("inf")), dist_reduce_fx="min"
        )
        self.add_state(
            "sa_max", default=torch.tensor(float("-inf")), dist_reduce_fx="max"
        )
        self.add_state(
            "sa_count", default=torch.tensor(0), dist_reduce_fx="sum"
        )

        # States for validity (collection-level metric)
        self.add_state(
            "validity_sum", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state(
            "validity_count", default=torch.tensor(0), dist_reduce_fx="sum"
        )

        # States for uniqueness (collection-level metric)
        self.add_state(
            "uniqueness_sum", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state(
            "uniqueness_count", default=torch.tensor(0), dist_reduce_fx="sum"
        )

    def update(self, smiles_list: List[str]) -> None:
        """Update metric states with a batch of SMILES strings.

        Args:
            smiles_list: List of SMILES strings from decoded generations
        """
        if not smiles_list:
            return

        # Convert to tensor types for proper device handling
        device = self.device

        # 1. Compute diversity (on unique SMILES only)
        unique_smiles = list(set(smiles_list))
        if len(unique_smiles) >= 2:  # Need at least 2 for pairwise diversity
            try:
                diversity_val = self.evaluator_diversity(unique_smiles)
                self.diversity_sum += torch.tensor(
                    diversity_val, dtype=self.dtype, device=device
                )
                self.diversity_count += 1
            except Exception:
                pass  # Skip if diversity computation fails

        # 2. Compute QED for each molecule
        qed_values = []
        for smiles in smiles_list:
            try:
                qed_val = self.oracle_qed(smiles)
                if qed_val is not None:
                    qed_values.append(qed_val)
            except Exception:
                pass  # Skip invalid molecules

        if qed_values:
            qed_tensor = torch.tensor(
                qed_values, dtype=self.dtype, device=device
            )
            self.qed_sum += qed_tensor.sum()
            self.qed_sum_sq += (qed_tensor**2).sum()
            self.qed_min = torch.min(self.qed_min, qed_tensor.min())
            self.qed_max = torch.max(self.qed_max, qed_tensor.max())
            self.qed_count += len(qed_values)

        # 3. Compute SA for each molecule
        sa_values = []
        for smiles in smiles_list:
            try:
                sa_val = self.oracle_sa(smiles)
                if sa_val is not None:
                    sa_values.append(sa_val)
            except Exception:
                pass  # Skip invalid molecules

        if sa_values:
            sa_tensor = torch.tensor(
                sa_values, dtype=self.dtype, device=device
            )
            self.sa_sum += sa_tensor.sum()
            self.sa_sum_sq += (sa_tensor**2).sum()
            self.sa_min = torch.min(self.sa_min, sa_tensor.min())
            self.sa_max = torch.max(self.sa_max, sa_tensor.max())
            self.sa_count += len(sa_values)

        # 4. Compute validity
        try:
            validity_val = self.evaluator_validity(smiles_list)
            self.validity_sum += torch.tensor(
                validity_val, dtype=self.dtype, device=device
            )
            self.validity_count += 1
        except Exception:
            pass

        # 5. Compute uniqueness
        try:
            uniqueness_val = self.evaluator_uniqueness(smiles_list)
            self.uniqueness_sum += torch.tensor(
                uniqueness_val, dtype=self.dtype, device=device
            )
            self.uniqueness_count += 1
        except Exception:
            pass

    def compute(self) -> Dict[str, Tensor]:
        """Compute final metric values.

        Returns:
            Dictionary with all metric values:
            - diversity: Mean diversity score
            - qed_mean, qed_std, qed_min, qed_max: QED statistics
            - sa_mean, sa_std, sa_min, sa_max: SA statistics
            - validity: Mean validity fraction
            - uniqueness: Mean uniqueness fraction
        """
        result = {}

        # Diversity
        if self.diversity_count > 0:
            result["diversity"] = self.diversity_sum / self.diversity_count
        else:
            result["diversity"] = torch.tensor(
                0.0, dtype=self.dtype, device=self.device
            )

        # QED statistics
        if self.qed_count > 0:
            mean = self.qed_sum / self.qed_count
            variance = (self.qed_sum_sq / self.qed_count) - (mean**2)
            std = torch.sqrt(
                torch.clamp(variance, min=0.0)
            )  # Clamp for numerical stability

            result["qed_mean"] = mean
            result["qed_std"] = std
            result["qed_min"] = self.qed_min
            result["qed_max"] = self.qed_max
        else:
            result["qed_mean"] = torch.tensor(
                0.0, dtype=self.dtype, device=self.device
            )
            result["qed_std"] = torch.tensor(
                0.0, dtype=self.dtype, device=self.device
            )
            result["qed_min"] = torch.tensor(
                0.0, dtype=self.dtype, device=self.device
            )
            result["qed_max"] = torch.tensor(
                0.0, dtype=self.dtype, device=self.device
            )

        # SA statistics
        if self.sa_count > 0:
            mean = self.sa_sum / self.sa_count
            variance = (self.sa_sum_sq / self.sa_count) - (mean**2)
            std = torch.sqrt(torch.clamp(variance, min=0.0))

            result["sa_mean"] = mean
            result["sa_std"] = std
            result["sa_min"] = self.sa_min
            result["sa_max"] = self.sa_max
        else:
            result["sa_mean"] = torch.tensor(
                0.0, dtype=self.dtype, device=self.device
            )
            result["sa_std"] = torch.tensor(
                0.0, dtype=self.dtype, device=self.device
            )
            result["sa_min"] = torch.tensor(
                0.0, dtype=self.dtype, device=self.device
            )
            result["sa_max"] = torch.tensor(
                0.0, dtype=self.dtype, device=self.device
            )

        # Validity
        if self.validity_count > 0:
            result["validity"] = self.validity_sum / self.validity_count
        else:
            result["validity"] = torch.tensor(
                0.0, dtype=self.dtype, device=self.device
            )

        # Uniqueness
        if self.uniqueness_count > 0:
            result["uniqueness"] = self.uniqueness_sum / self.uniqueness_count
        else:
            result["uniqueness"] = torch.tensor(
                0.0, dtype=self.dtype, device=self.device
            )

        return result


def _molgen_update_fn_DEPRECATED(
    batch: Dict[str, Any],
    loss_dict: Dict[str, Any],
    tokenizer: Any = None,
    use_bracket_safe: bool = False,
) -> Dict[str, Any]:
    """DEPRECATED: Update function for MolGenMetric.

    Use DeNovoEval.eval() instead for proper post-hoc evaluation.

    Decodes generated token IDs to SAFE strings, converts to SMILES,
    and returns them for metric computation.

    Args:
        batch: Input batch (not used, but required by MetricWrapper interface)
        loss_dict: Dictionary containing 'ids' key with generated token IDs
        tokenizer: Tokenizer for decoding
        use_bracket_safe: If True, use bracket SAFE notation

    Returns:
        Dictionary with 'smiles_list' key for metric update
    """
    if tokenizer is None:
        raise ValueError("Tokenizer is required for molgen metrics")

    # Decode generated IDs to SAFE strings
    safe_strings = tokenizer.batch_decode(
        loss_dict["ids"], skip_special_tokens=True
    )

    # Convert SAFE to SMILES
    smiles_list = safe_strings_to_smiles(
        safe_strings, use_bracket_safe=use_bracket_safe, fix=True
    )

    return {"smiles_list": smiles_list}


# endregion: DEPRECATED
################################################################################
