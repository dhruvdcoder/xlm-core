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

"""SAFE / molecular generation task utilities and post-hoc evaluation.

This module provides:
- Data preprocessing for SAFE molecular representations
- Conversion utilities between SAFE and SMILES formats
- Post-hoc evaluators (:class:`DeNovoEval`, :class:`FragmentEval`) for diversity, QED, SA, validity, uniqueness, etc.
- Optional FCD vs training set (``DeNovoEval`` + TDC) requires ``pip install xlm[fcd]`` or ``fcd_torch`` / TensorFlow ``FCD``
"""

import itertools
from pathlib import Path
import random
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import os

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase

from xlm.utils.rank_zero import warn_once, RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)

# Import SAFE library for molecular encoding/decoding
try:
    import datamol as dm
    import safe as sf
    from safe.converter import SAFEConverter
    from safe.tokenizer import SAFETokenizer
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
except ImportError:
    raise ImportError(
        "Please install safe-mol, rdkit, and datamol: pip install safe-mol rdkit datamol"
    )

# Import TDC for molecular metrics
try:
    from tdc import Oracle, Evaluator
except ImportError:
    raise ImportError("Please install TDC: pip install pytdc")


# use this file as a reference
ZINC_LENGTH_REF_FILE = Path(__file__).parent / "zinc_len.pkl"

################################################################################
# region: SAFE / Bracket SAFE conversion (from GenMol bracket_safe_converter.py)
#
# SMILES refresher:
#   - Linear notation: atoms (C, N, ...), bonds (=, #), branches in parentheses,
#     ring closures by matching digit pairs (e.g. two "1" atoms close a ring),
#     or %dd for two-digit ring indices (10, 11, ...).
#
# SAFE (Serial Arrangement of Fragments Encoding):
#   - The molecule is fragmented along BRICS-like bonds. Each piece is a SMILES
#     substring; pieces are joined with "." (disconnected SMILES). Cut ends that
#     belonged to the same bond share a ring-closure-style label (1–9 or %10),
#     so the decoder can stitch the graph back together—same digit syntax as
#     ordinary SMILES ring closures, but meaning "reconnect across fragments".
#     see: https://safe-docs.datamol.io/stable/tutorials/getting-started.html
#
# Bracket SAFE:
#   - Fragments still use normal SMILES ring closures internally. Inter-fragment
#     attachment labels use the same digit tokens, so they can be confused with
#     intra-fragment closures. Bracket SAFE wraps only the *attachment* digits in
#     angle brackets: <1>, <2>, … or <%10> when the index is two digits.
#     see: https://github.com/NVIDIA-Digital-Bio/genmol?tab=readme-ov-file#introduction
#
# Illustrative pattern (actual strings depend on BRICS cuts and canonicalization):
#   SMILES:  one connected string for the whole molecule
#   SAFE:    fragA...1....fragB...1...   (matching "1" = stitch point)
#   Bracket: fragA...<1>....fragB...<1>...


class BracketSAFEConverter(SAFEConverter):
    """SAFE encoder that marks inter-fragment attachment digits with < >.

    Inherits the standard SAFE pipeline (BRICS fragmentation, dummy atoms at
    cuts, MolToSmiles per fragment) and only changes the final string: attachment
    markers like [1*] become <1> / <%N> so they are distinct from in-fragment
    ring-closure digits.
    """

    def encoder(
        self,
        inp: Union[str, dm.Mol],
        canonical: bool = True,
        randomize: Optional[bool] = False,
        seed: Optional[int] = None,
        constraints: Optional[List[dm.Mol]] = None,
        allow_empty: bool = False,
        rdkit_safe: bool = True,
    ):
        """The only difference from SAFEConverter.encoder appears at the end when inter-fragment attachment digits are replaced with <n>."""
        # Optional randomization: shuffle the within molecule index of the atoms.
        # This will result in different walk order for the same molecule, resulting in different SMILES string for the same molecule, and therefore fragment order for data aug
        # TODO: handle seed better
        rng = None
        if randomize:
            rng = np.random.default_rng(seed)
            if not canonical:
                inp = dm.to_mol(inp, remove_hs=False)
                inp = self.randomize(inp, rng)

        # Normalize input to a SMILES string (SAFEConverter expects SMILES here)
        if isinstance(inp, dm.Mol):
            inp = dm.to_smiles(
                inp, canonical=canonical, randomize=False, ordered=False
            )

        # Base-class hook: reserved for branch/ring bookkeeping (same as reference)
        branch_numbers = self._find_branch_number(inp)

        # Parse SMILES to RDKit mol; optionally drop stereochemistry
        mol = dm.to_mol(inp, remove_hs=False)
        if self.ignore_stereo:
            mol = dm.remove_stereochemistry(mol)

        # Next, turn any pre-existing attachment points into dummy isotope atoms so that they can be distinguished from the new dummy atoms created by the fragmentation.
        # see: https://safe-docs.datamol.io/stable/tutorials/how-it-works.html for more details.
        # Pre-tag existing dummy atoms (*): isotope = cut index so each site is unique
        # Label dummy atoms (attachment points after fragmentation) with isotopes
        # so RDKit can emit distinct SMILES for each cut site.
        bond_map_id = 1
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0:  # this is a dummy attachment atom
                atom.SetAtomMapNum(
                    0
                )  # clear the map number on the dummy which would show up as [*:1], etc.
                atom.SetIsotope(
                    bond_map_id
                )  # set a unique isotope number for the dummy. This will show up as [1*], etc.
                bond_map_id += 1

        # Optional explicit Hs;
        if self.require_hs:
            mol = dm.add_hs(mol)
        # Bonds chosen for cleavage (BRICS / SAFE rules from base class).
        matching_bonds = self._fragment(mol, allow_empty=allow_empty)

        # If user passed constraint mols, remember atom sets we must not split inside
        substructed_ignored = []
        if constraints is not None:
            substructed_ignored = list(
                itertools.chain(
                    *[
                        mol.GetSubstructMatches(constraint, uniquify=True)
                        for constraint in constraints
                    ]
                )
            )

        # Map chosen (atom_i, atom_j) pairs to RDKit bond indices for FragmentOnBonds
        bonds = []
        for i_a, i_b in matching_bonds:
            # Skip cuts that would break inside a user-preserved substructure.
            if any(
                (i_a in ignore_x and i_b in ignore_x)
                for ignore_x in substructed_ignored
            ):
                continue
            obond = mol.GetBondBetweenAtoms(i_a, i_b)
            bonds.append(obond.GetIdx())

        # Physically cut those bonds: each cut becomes a pair of dummy attachment atoms
        if len(bonds) > 0:
            mol = Chem.FragmentOnBonds(
                mol,
                bonds,
                dummyLabels=[
                    (i + bond_map_id, i + bond_map_id)
                    for i in range(len(bonds))
                ],
            )

        # Split disconnected components; order fragments (random perm or largest first)
        frags = list(Chem.GetMolFrags(mol, asMols=True))
        if randomize:
            frags = rng.permutation(frags).tolist()
        elif canonical:
            frags = sorted(
                frags,
                key=lambda x: x.GetNumAtoms(),
                reverse=True,
            )

        # One canonical isomeric SMILES per fragment, rooted at a real (non-dummy) atom
        frags_str = []
        for frag in frags:
            non_map_atom_idxs = [
                atom.GetIdx()
                for atom in frag.GetAtoms()
                if atom.GetAtomicNum() != 0
            ]
            frags_str.append(
                Chem.MolToSmiles(
                    frag,
                    isomericSmiles=True,
                    canonical=True,  # needs to always be true
                    rootedAtAtom=non_map_atom_idxs[0],
                )
            )

        # SAFE string: fragments joined by "." (same as disconnected SMILES)
        scaffold_str = ".".join(frags_str)

        # Bracket SAFE: find dummy attachment tokens in the string, renumber 1..K, wrap as <n>
        # Dummy attachment sites appear as [n*] or isotope-mapped forms; replace
        # each with a fresh inter-fragment index wrapped in < > (SMILES digit rules).
        attach_pos = set(
            re.findall(r"(\[\d+\*\]|!\[[^:]*:\d+\])", scaffold_str)
        )
        if canonical:
            attach_pos = sorted(attach_pos)
        # ONLY DIFFERENCE BETWEEN SAFE and BRACKET SAFE. setting starting_num=1 instead of 0. and using < > below.
        starting_num = 1
        for attach in attach_pos:
            val = (
                str(starting_num) if starting_num < 10 else f"%{starting_num}"
            )
            # ONLY DIFFERENCE
            val = "<" + val + ">"
            attach_regexp = re.compile(r"(" + re.escape(attach) + r")")
            scaffold_str = attach_regexp.sub(val, scaffold_str)
            starting_num += 1

        # Post-process SMILES: drop spurious parens around <...> and normalize ring digits
        # Strip redundant parens around bracket-only attachment tokens.
        wrong_attach = re.compile(r"\((<[\%\d+]*>)\)")
        scaffold_str = wrong_attach.sub(r"\g<1>", scaffold_str)
        # RDKit-style: unwrap (bond_symbol)(digit) when digit is a ring closure.
        if rdkit_safe:
            pattern = r"\(([=-@#\/\\]{0,2})(%?\d{1,2})\)"
            replacement = r"\g<1>\g<2>"
            scaffold_str = re.sub(pattern, replacement, scaffold_str)
        # Final Bracket SAFE string
        return scaffold_str


def safe2bracketsafe(safe_str: str, seed: Optional[int] = None) -> str:
    """Standard SAFE -> Bracket SAFE via full re-encode (GenMol reference).

    Parses the SAFE string as a molecule, then runs BracketSAFEConverter.encoder
    so attachment points are identified from dummy-atom patterns, not by guessing
    digits in the string.
    """
    try:
        return BracketSAFEConverter().encoder(
            Chem.MolFromSmiles(safe_str),
            allow_empty=True,
            canonical=False,
            randomize=True,
            seed=1,
            # seed=seed
            # or np.random.randint(
            #    0, 1000000
            # ),  # for reproducibility draw from global random state
        )
    except:  # noqa: E722
        return safe_str


def bracketsafe2safe(
    safe_str: str, fix_non_single_bond_fragments: bool = True
) -> str:
    """Bracket SAFE -> standard SAFE (GenMol reference).

    Intra-fragment ring closures stay as bare digits / %dd. Tokens inside <n>
    are inter-fragment attachment labels: strip brackets and renumber so those
    labels do not collide with any digit already used inside a fragment.

    Args:
        safe_str: Bracket SAFE string.
        fix_non_single_bond_fragments: If True (default), unwrap parenthesised
            bond+ring-closure patterns like ``(=4)`` to ``=4`` after bracket
            removal. RDKit cannot parse the former when they come from cut
            non-single bonds; set False only if the encoder is fixed upstream.
    """
    # Collect numeric labels that are *not* inter-fragment (not inside <...>).
    # (?<!%)\\d(?!>) matches a single ring-closure digit not part of %dd and not
    # followed by '>' (i.e. not the closing part of a bracket token).
    intrafrag_points = [
        m.group(0) for m in re.finditer(r"(?<!%)\d(?!>)", safe_str)
    ] + [m.group(0).lstrip("%") for m in re.finditer(r"%\d+", safe_str)]
    # New attachment indices must be strictly larger than any intra-fragment index.
    starting_num = (
        max([int(i) for i in intrafrag_points]) + 1 if intrafrag_points else 0
    )
    # Inter-fragment markers use only <digits> in the reference (see encoder).
    interfrag_points = [
        (m.start(0), m.end(0)) for m in re.finditer(r"<\d+>", safe_str)
    ]

    safe_str = list(safe_str)
    for start, end in interfrag_points:
        # Temporarily blank the angle brackets; the slice between them becomes the
        # integer to offset, then we pad with spaces so string length stays stable.
        safe_str[start] = safe_str[end - 1] = " "
        num_to_replace = (
            int("".join(safe_str[start + 1 : end - 1])) + starting_num
        )
        num_to_replace = (
            "%" + str(num_to_replace)
            if num_to_replace >= 10
            else str(num_to_replace)
        )
        safe_str[start + 1 : end - 1] = [num_to_replace] + [" "] * (
            end - start - 3
        )
    safe_str = re.sub(" ", "", "".join(safe_str))
    # TEMPORARY PATCH: BracketSAFEConverter can cut non-single bonds, serialising
    # them as branches with a bond specifier, e.g. C(=<1>)C. After bracket removal
    # this becomes C(=4)C, which RDKit cannot parse (ring-closure inside a branch
    # with a bond type). Strip the parens to produce the valid inline form C=4.
    # Disable with fix_non_single_bond_fragments=False if the encoder is fixed.
    if fix_non_single_bond_fragments:
        safe_str = re.sub(
            r"\(([=#@/\\]{0,2})(%?\d{1,2})\)", r"\g<1>\g<2>", safe_str
        )
    return safe_str


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


def filter_by_substructure(sequences: List[str], substruct: str) -> List[str]:
    """Keep only SMILES that contain ``substruct`` as a substructure.

    Replicates GenMol's ``genmol.utils.utils_chem.filter_by_substructure``;
    used for fragment-completion evaluation (motif extension, scaffold
    decoration, superstructure generation).

    Args:
        sequences: Generated SMILES strings (one molecule each, as in GenMol).
        substruct: Fragment SMILES with attachment markers (``*``), matching
            the constraint used at generation time.

    Returns:
        Subset of ``sequences`` that satisfy the substructure constraint.
    """
    substruct = sf.utils.standardize_attach(substruct)
    substruct = Chem.DeleteSubstructs(
        Chem.MolFromSmarts(substruct), Chem.MolFromSmiles("*")
    )
    substruct = Chem.MolFromSmarts(Chem.MolToSmiles(substruct))
    return sf.utils.filter_by_substructure_constraints(sequences, substruct)


def get_distance(smiles: str, generated_smiles: List[str]) -> Optional[float]:
    """Mean Tanimoto distance from a reference molecule to generated molecules.

    Replicates GenMol's ``scripts/exps/frag/run.py:get_distance``.

    Args:
        smiles: Reference (original) molecule SMILES.
        generated_smiles: List of generated SMILES (typically unique valid).

    Returns:
        Mean Tanimoto distance, or ``None`` if computation fails.
    """
    from rdkit import DataStructs
    from rdkit.Chem import AllChem

    try:
        ref_mol = Chem.MolFromSmiles(smiles)
        if ref_mol is None:
            return None
        ref_fp = AllChem.GetMorganFingerprintAsBitVect(ref_mol, 2, 1024)

        gen_fps = []
        for smi in generated_smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                gen_fps.append(
                    AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)
                )

        if not gen_fps:
            return None

        return float(
            np.mean(
                DataStructs.BulkTanimotoSimilarity(
                    ref_fp, gen_fps, returnDistance=True
                )
            )
        )
    except Exception:
        return None


def get_fcd_distance(
    generated_smiles: List[str], reference_smiles: List[str]
) -> Optional[float]:
    """pyTDC FCD distance wrapper has multiple bugs. So we use our own that call fcd_torch directly"""
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    try:
        from fcd_torch import FCD
        from fcd import canonical_smiles
    except ImportError:
        raise ImportError("Please install fcd_torch: pip install fcd_torch")

    fcd = FCD(device="cpu", n_jobs=8)
    # filter out empty SMILES
    generated_smiles = canonical_smiles(
        [smiles for smiles in generated_smiles if smiles]
    )
    reference_smiles = canonical_smiles(
        [smiles for smiles in reference_smiles if smiles]
    )
    fcd_distance = fcd(reference_smiles, generated_smiles)
    fcd_distance = np.exp(-0.2 * fcd_distance)
    return fcd_distance


def safe_strings_to_smiles(
    safe_strings: List[str],
    use_bracket_safe: bool = False,
    fix: bool = True,
    fix_non_single_bond_fragments: bool = True,
) -> List[str]:
    """Convert batch of SAFE strings to SMILES strings.

    Based on genmol/src/genmol/sampler.py:81-89

    Args:
        safe_strings: List of SAFE molecular representations
        use_bracket_safe: If True, convert from bracket SAFE first
        fix: If True, filter invalid fragments
        fix_non_single_bond_fragments: Passed to :func:`bracketsafe2safe` when
            ``use_bracket_safe`` is True (see that docstring).

    Returns:
        List of SMILES strings (invalid conversions are skipped)
    """
    smiles_list = []

    for b_safe_str in safe_strings:
        # Convert from bracket SAFE if needed
        if use_bracket_safe:
            safe_str = bracketsafe2safe(
                b_safe_str,
                fix_non_single_bond_fragments=fix_non_single_bond_fragments,
            )
        else:
            safe_str = b_safe_str

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

    def __len__(self):
        return self.len()


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


# GenMol fragment_linking_onestep / run.py: linker_design and scaffold_morphing
# share the same CSV column and encoding (scaffold_morphing is an alias).
LINKER_TASKS = frozenset({"linker_design", "scaffold_morphing"})
TASK_TO_COLUMN = {"scaffold_morphing": "linker_design"}


def genmol_fragment_preprocess_fn(
    example: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    *,
    fragment_column: str = "motif_extension",
) -> Dict[str, Any]:
    """Preprocess GenMol fragment CSV data for fragment-constrained generation.

    Converts SMILES fragments and targets to SAFE format, then to bracket SAFE,
    and creates prompt_token_ids (fragment) and input_token_ids (full molecule).

    Based on GenMol's fragment evaluation dataset structure:
    - Input: Fragment SMILES
    - Target: Full molecule SMILES (from 'smiles' column)

    **Fragment completion** (``motif_extension``, ``scaffold_decoration``,
    ``superstructure_generation``): matches GenMol ``fragment_completion``
    (superstructure handling, ``BracketSAFEConverter(ignore_stereo=True)``).

    **Fragment linking** (``linker_design``, ``scaffold_morphing``): matches
    GenMol ``fragment_linking_onestep`` (``BracketSAFEConverter(slicer=None)``
    on the fragment; ``scaffold_morphing`` reads the ``linker_design`` column).

    To use a different fragment column, pass `fragment_column` via
    `preprocess_function_kwargs` in the dataset config, or set
    `_fragment_column` in the example dict (overrides kwarg).

    Eg: data point
    name    smiles    linker_design    motif_extension    scaffold_decoration    superstructure_generation
    BARICITINIB    CCS(=O)(=O)N1CC(CC#N)(n2cc(-c3ncnc4[nH]ccc34)cn2)C1    [11*]C1(CC#N)CN(S(=O)(=O)CC)C1.[13*]c1ncnc2[nH]ccc12    [11*]C1(CC#N)CN(S(=O)(=O)CC)C1    [1*]N1CC([2*])(n2cc(-c3ncnc4[nH]ccc34)cn2)C1    c1nc(-c2cnn(C3CNC3)c2)c2cc[nH]c2n1

    Args:
        example: Dataset example containing:
            - column named by `fragment_column`: SMILES with [n*] attachment points
            - 'smiles': Full target molecule SMILES
            - '_fragment_column' (optional): Overrides `fragment_column` if set
        tokenizer: Tokenizer for encoding
        fragment_column: CSV column to use as fragment input (default:
            ``motif_extension``). Override via datamodule
            ``preprocess_function_kwargs.fragment_column``.

    Returns:
        Example with 'prompt_token_ids' (fragment) and 'input_token_ids' (full molecule)
    """
    # Per-example override wins over preprocess_function_kwargs
    fragment_column = example.get("_fragment_column", fragment_column)
    data_column = TASK_TO_COLUMN.get(fragment_column, fragment_column)

    # Get fragment SMILES (input); alias tasks map to CSV column (e.g. scaffold_morphing -> linker_design)
    fragment_smiles = example.get(data_column)
    if fragment_smiles is None or not fragment_smiles:
        raise ValueError(
            f"Example must contain '{data_column}' field with fragment SMILES"
        )

    # Get target SMILES (full molecule)
    target_smiles = example.get("smiles")
    if target_smiles is None:
        raise ValueError(
            "Example must contain 'smiles' field with target molecule SMILES"
        )

    # GenMol fragment_completion vs fragment_linking_onestep (sampler.py).
    try:
        is_linker_task = fragment_column in LINKER_TASKS

        if not is_linker_task:
            # Superstructure generation: input has no '*' — pick a random attach core
            if "*" not in fragment_smiles:
                frag_mol = Chem.MolFromSmiles(fragment_smiles)
                if frag_mol is None:
                    raise ValueError(
                        f"Failed to parse fragment SMILES: {fragment_smiles}"
                    )
                cores = sf.utils.list_individual_attach_points(
                    frag_mol, depth=3
                )
                if not cores:
                    raise ValueError(
                        "list_individual_attach_points returned no cores for "
                        f"superstructure fragment: {fragment_smiles!r}"
                    )
                fragment_smiles = random.choice(cores)

        # Encode directly to Bracket SAFE.  We cannot use SAFEConverter then
        # safe2bracketsafe because safe2bracketsafe round-trips through
        # Chem.MolFromSmiles, which fails on fragment SAFE strings that have
        # dangling (unmatched) ring-closure digits at attachment points.
        if is_linker_task:
            fragment_converter = BracketSAFEConverter(slicer=None)
            target_converter = BracketSAFEConverter(ignore_stereo=True)
        else:
            fragment_converter = BracketSAFEConverter(ignore_stereo=True)
            target_converter = fragment_converter

        fragment_bracket_safe = (
            fragment_converter.encoder(
                fragment_smiles,
                allow_empty=True,
                seed=np.random.randint(0, 1000000),
            )
            + "."
        )

        target_mol = Chem.MolFromSmiles(target_smiles)
        if target_mol is None:
            raise ValueError(f"Failed to parse target SMILES: {target_smiles}")
        target_bracket_safe = target_converter.encoder(
            target_mol, allow_empty=True, seed=np.random.randint(0, 1000000)
        )

    except Exception as e:
        raise ValueError(f"Failed to convert SMILES to SAFE: {e}")

    # Fragment after superstructure handling (for substructure filter at eval)
    example["fragment_smiles"] = fragment_smiles
    # Keep raw SMILES for distance computation
    example["original_smiles"] = target_smiles

    # Tokenize
    fragment_token_ids = tokenizer.encode(
        fragment_bracket_safe, add_special_tokens=False
    )
    target_token_ids = tokenizer.encode(
        target_bracket_safe, add_special_tokens=False
    )

    example["prompt_token_ids"] = fragment_token_ids
    example["input_token_ids"] = target_token_ids

    return example


# endregion: Data Preprocessing
################################################################################


def _fcd_row_smiles(row: Any, column: str) -> Optional[str]:
    """Return stripped SMILES string from a dataset row, or None."""
    if isinstance(row, dict):
        v = row.get(column)
    else:
        try:
            v = row[column]
        except (KeyError, TypeError, IndexError):
            v = None
    if isinstance(v, str) and v.strip():
        return v.strip()
    return None


def _fcd_hf_split_first_n(split: str, n: int) -> str:
    """Build a Hugging Face ``split`` argument for the first ``n`` rows.

    If ``split`` already includes a range (e.g. ``train[:5000]``), it is passed
    through unchanged.
    """
    if n <= 0:
        return split
    if "[" in split:
        return split
    return f"{split}[:{n}]"


def _load_fcd_reference_smiles(
    path: str,
    *,
    name: Optional[str] = None,
    split: str = "train",
    column: str = "smiles",
    n: int = 10000,
) -> List[str]:
    """Load the first ``n`` rows of a map-style Hugging Face dataset split.

    Used as the training/reference distribution for TDC ``fcd_distance``.
    Expects ``column`` to contain SMILES (not SAFE), matching decoded
    ``pred["smiles"]`` from the evaluator.

    Rows are taken in on-disk / hub order via split slicing (e.g.
    ``train[:10000]``). **The split should already be shuffled at dataset
    creation time** if you want an i.i.d.-style reference sample.

    Args:
        path: First argument to ``datasets.load_dataset`` (hub id or builder).
        name: Optional dataset configuration name (second positional arg).
        split: Base split name (e.g. ``train``) or a full slice (e.g. ``train[:5000]``).
        column: Column name with SMILES strings.
        n: When ``split`` has no ``[...]`` range, load at most this many rows.

    Returns:
        List of non-empty SMILES from those rows (invalid/empty cells skipped).
    """
    from datasets import load_dataset

    if n <= 0:
        return []

    split_arg = _fcd_hf_split_first_n(split, n)
    if name is not None:
        ds = load_dataset(path, name, split=split_arg)
    else:
        ds = load_dataset(path, split=split_arg)
    out: List[str] = []
    for i in range(len(ds)):
        s = _fcd_row_smiles(ds[i], column)
        if s is not None:
            out.append(s)
    return out


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
        compute_fcd: If True, compute TDC Fréchet ChemNet score vs a reference set
        fcd_reference_dataset_path: Hugging Face ``load_dataset`` path (hub id or
            builder). Required when ``compute_fcd`` is True.
        fcd_reference_dataset_name: Optional dataset config name for
            ``load_dataset(path, name, split=...)``.
        fcd_reference_split: Split to load for reference SMILES.
        fcd_reference_column: Column containing SMILES strings.
        fcd_num_reference: Number of reference rows to load via split slicing
            (e.g. ``train[:fcd_num_reference]``) when ``fcd_reference_split`` has
            no explicit range.
        fcd_unique_generated: If True, pass deduplicated valid generated SMILES to FCD;
            if False, pass all valid generations (including duplicates).
    """

    def __init__(
        self,
        use_bracket_safe: bool = False,
        compute_diversity: bool = True,
        compute_validity: bool = True,
        compute_uniqueness: bool = True,
        compute_qed: bool = True,
        compute_sa: bool = True,
        compute_fcd: bool = False,
        fcd_reference_dataset_path: Optional[str] = None,
        fcd_reference_dataset_name: Optional[str] = None,
        fcd_reference_split: str = "train",
        fcd_reference_column: str = "smiles",
        fcd_num_reference: int = 1000,
        fcd_unique_generated: bool = False,
    ):
        self.use_bracket_safe = use_bracket_safe
        self.compute_diversity = compute_diversity
        self.compute_validity = compute_validity
        self.compute_uniqueness = compute_uniqueness
        self.compute_qed = compute_qed
        self.compute_sa = compute_sa
        self.compute_fcd = compute_fcd
        self.fcd_reference_dataset_path = fcd_reference_dataset_path
        self.fcd_reference_dataset_name = fcd_reference_dataset_name
        self.fcd_reference_split = fcd_reference_split
        self.fcd_reference_column = fcd_reference_column
        self.fcd_num_reference = fcd_num_reference
        self.fcd_unique_generated = fcd_unique_generated

        # Lazy-loaded TDC oracles (to avoid import overhead)
        self._oracle_qed = None
        self._oracle_sa = None
        self._evaluator_diversity = None
        self._evaluator_validity = None
        self._evaluator_uniqueness = None
        self._evaluator_fcd = None
        self._fcd_reference_smiles: Optional[List[str]] = None

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

    @property
    def evaluator_fcd(self):
        """Lazy load TDC FCD evaluator (requires fcd_torch or TensorFlow FCD)."""
        if self._evaluator_fcd is None and self.compute_fcd:
            # self._evaluator_fcd = Evaluator("fcd_distance")
            self._evaluator_fcd = get_fcd_distance
        return self._evaluator_fcd

    def _ensure_fcd_reference_loaded(self) -> bool:
        """Load and cache reference SMILES once; return True if non-empty."""
        if self._fcd_reference_smiles is not None:
            return len(self._fcd_reference_smiles) > 0
        path = self.fcd_reference_dataset_path
        if not path:
            warn_once(
                logger,
                "compute_fcd is True but fcd_reference_dataset_path is unset; skipping FCD.",
            )
            self._fcd_reference_smiles = []
            return False
        warn_once(
            logger,
            "FCD reference SMILES are the first fcd_num_reference rows of the HF "
            "split (via split slicing, e.g. train[:N]). Assume the dataset split is "
            "already shuffled at export time; otherwise reference statistics may be biased.",
        )
        try:
            self._fcd_reference_smiles = _load_fcd_reference_smiles(
                path,
                name=self.fcd_reference_dataset_name,
                split=self.fcd_reference_split,
                column=self.fcd_reference_column,
                n=self.fcd_num_reference,
            )
        except Exception as exc:
            logger.warning(
                "Failed to load FCD reference dataset from %r: %s",
                path,
                exc,
            )
            self._fcd_reference_smiles = []
        if not self._fcd_reference_smiles:
            warn_once(
                logger,
                "FCD reference SMILES list is empty after load; skipping FCD.",
            )
        return len(self._fcd_reference_smiles) > 0

    def eval(
        self,
        predictions: List[Dict[str, Any]],
        tokenizer: Any = None,
        **kwargs: Any,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Evaluate predictions and return updated predictions + aggregated metrics.

        Args:
            predictions: List of prediction dicts with 'text' field containing SAFE strings
            tokenizer: Optional tokenizer (not used for denovo, but kept for interface consistency)
            **kwargs: Additional keyword arguments (ignored, accepted for
                forward-compatibility with ``CompositePostHocEvaluator``
                and ``Harness.compute_post_hoc_metrics``, which pass
                ``dataloader_name``).

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

            if self.compute_fcd:
                if self._ensure_fcd_reference_loaded():
                    gen_fcd = (
                        list(dict.fromkeys(all_smiles))
                        if self.fcd_unique_generated
                        else all_smiles
                    )
                    if gen_fcd:
                        try:
                            fcd_val = self.evaluator_fcd(
                                gen_fcd, self._fcd_reference_smiles
                            )
                            aggregated_metrics["fcd_tdc"] = float(fcd_val)
                        except Exception as exc:
                            logger.warning(
                                "FCD computation failed: %s",
                                exc,
                            )
                            import traceback

                            logger.error(f"{traceback.format_exc()}")

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
            # This is heuristic used in genmol but it can artifically raise the validity score.
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

    Matches GenMol's ``scripts/exps/frag/run.py`` evaluation semantics:
    predictions are split into **groups** of ``num_samples`` (one group per
    input fragment), metrics are computed **per group**, then **averaged**
    across groups.

    Per-group metrics (GenMol run.py:74-92):
    - validity  = len(valid_smiles) / num_samples
    - uniqueness = len(unique_smiles) / len(valid_smiles)
    - diversity  = pairwise Tanimoto on unique molecules (0 if <=1)
    - quality    = len(qed>=0.6 & sa<=4) / num_samples
    - distance   = mean Tanimoto distance from original molecule to all
                   unique generated molecules (GenMol ``get_distance``)

    Args:
        task_name: Fragment task name (e.g. ``motif_extension``,
            ``scaffold_decoration``, ``superstructure_generation``,
            ``linker_design``, ``scaffold_morphing``). The latter two use the
            same linking pipeline as GenMol ``fragment_linking_onestep``;
            ``scaffold_morphing`` reads the ``linker_design`` CSV column.
        num_samples: Number of generations per fragment (must match the
            ``replicate_examples`` ``num_samples`` used in the datamodule).
        use_bracket_safe: If True, decode from bracket SAFE format.
        compute_substructure_filter: If True, apply ``filter_by_substructure``
            on each sample before computing metrics (GenMol default).
        compute_diversity: If True, compute diversity metric.
        compute_validity: If True, compute validity metric.
        compute_uniqueness: If True, compute uniqueness metric.
        compute_qed: If True, compute QED scores.
        compute_sa: If True, compute SA scores.
        compute_distance: If True, compute Tanimoto distance to target.
    """

    def __init__(
        self,
        task_name: str,
        num_samples: int = 1,
        use_bracket_safe: bool = False,
        compute_substructure_filter: bool = True,
        compute_diversity: bool = True,
        compute_validity: bool = True,
        compute_uniqueness: bool = True,
        compute_qed: bool = True,
        compute_sa: bool = True,
        compute_distance: bool = True,
        **kwargs: Any,
    ):
        super().__init__(
            use_bracket_safe=use_bracket_safe,
            compute_diversity=compute_diversity,
            compute_validity=compute_validity,
            compute_uniqueness=compute_uniqueness,
            compute_qed=compute_qed,
            compute_sa=compute_sa,
            **kwargs,
        )
        self.task_name = task_name
        self.num_samples = num_samples
        self.compute_substructure_filter = compute_substructure_filter
        self.compute_distance = compute_distance

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def eval(
        self,
        predictions: List[Dict[str, Any]],
        tokenizer: Any = None,
        **kwargs: Any,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Evaluate fragment generation predictions.

        Predictions are assumed to arrive in contiguous groups of
        ``self.num_samples`` (produced by ``replicate_examples``).

        Args:
            predictions: List of prediction dicts with:
                - 'text': Generated SAFE string (full molecule)
                - 'truth': Target SAFE string (full molecule, optional)
                - 'fragment_smiles': Fragment SMILES used for conditioning (for
                  linker tasks, two components joined by ``.`` as in the CSV)
            tokenizer: Optional tokenizer (not used).
            **kwargs: Ignored; accepts e.g. ``dataloader_name`` from the harness
                or ``CompositePostHocEvaluator``.

        Returns:
            Tuple of (predictions, aggregated_metrics).
        """
        if not predictions:
            return predictions, {}

        # 1. SAFE -> SMILES for every prediction
        for pred in predictions:
            safe_str = pred.get("text", "")
            smiles = self._safe_to_smiles_with_bracket_handling(safe_str)
            pred["smiles"] = smiles

        # 2. Substructure filter (per sample, before any aggregation)
        if self.compute_substructure_filter:
            for pred in predictions:
                frag = pred.get("fragment_smiles")
                smi = pred.get("smiles")
                if frag and smi is not None:
                    kept = filter_by_substructure([smi], frag)
                    if not kept:
                        pred["smiles"] = None

        # 3. Per-sample property computation (QED, SA, truth SMILES, distance)
        if self.compute_qed:
            self._compute_qed_per_sample(predictions)
        if self.compute_sa:
            self._compute_sa_per_sample(predictions)

        for pred in predictions:
            truth_safe = pred.get("truth", "")
            if truth_safe:
                pred["truth_smiles"] = (
                    self._safe_to_smiles_with_bracket_handling(truth_safe)
                )
            else:
                pred["truth_smiles"] = None

        # 4. Group predictions and compute per-group metrics
        groups = self._split_into_groups(predictions)
        aggregated_metrics = self._aggregate_group_metrics(groups)

        return predictions, aggregated_metrics

    # ------------------------------------------------------------------
    # Grouping
    # ------------------------------------------------------------------

    @staticmethod
    def _split_into_groups(
        predictions: List[Dict[str, Any]],
    ) -> List[List[Dict[str, Any]]]:
        """Group predictions by ``(fragment_smiles, truth)``.

        Uses an order-preserving dict so that the first occurrence of each
        key determines group order, but non-contiguous duplicates are still
        collected into the same group.
        """
        group_map: Dict[Any, List[Dict[str, Any]]] = {}
        for pred in predictions:
            key = (pred.get("fragment_smiles"), pred.get("truth"))
            group_map.setdefault(key, []).append(pred)

        return list(group_map.values())

    # ------------------------------------------------------------------
    # Per-group -> averaged metrics  (mirrors GenMol run.py:74-92)
    # ------------------------------------------------------------------

    def _aggregate_group_metrics(
        self, groups: List[List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Compute metrics per group and average across groups."""
        validity_vals: List[float] = []
        uniqueness_vals: List[float] = []
        diversity_vals: List[float] = []
        quality_vals: List[float] = []
        distance_vals: List[float] = []

        for group in groups:
            num_generated = len(group)
            valid_smiles = [
                p["smiles"] for p in group if p.get("smiles") is not None
            ]

            # -- validity --
            if self.compute_validity:
                validity_vals.append(
                    len(valid_smiles) / num_generated
                    if num_generated > 0
                    else 0.0
                )

            if not valid_smiles:
                if self.compute_uniqueness:
                    uniqueness_vals.append(0.0)
                if self.compute_qed and self.compute_sa:
                    quality_vals.append(0.0)
                continue

            # -- uniqueness --
            if self.compute_uniqueness:
                unique_smiles = list(dict.fromkeys(valid_smiles))
                uniqueness_vals.append(len(unique_smiles) / len(valid_smiles))

            # -- diversity (on unique set) --
            if self.compute_diversity:
                unique_set = list(set(valid_smiles))
                if len(unique_set) <= 1:
                    diversity_vals.append(0.0)
                else:
                    try:
                        diversity_vals.append(
                            float(self.evaluator_diversity(unique_set))
                        )
                    except Exception:
                        diversity_vals.append(0.0)

            # -- quality --
            if self.compute_qed and self.compute_sa:
                group_valid = [p for p in group if p.get("smiles") is not None]
                q_count = sum(
                    1
                    for p in group_valid
                    if p.get("qed") is not None
                    and p.get("sa") is not None
                    and p["qed"] >= 0.6
                    and p["sa"] <= 4.0
                )
                quality_vals.append(
                    q_count / num_generated if num_generated > 0 else 0.0
                )

            # -- distance --
            if self.compute_distance:
                original_smi = group[0].get("truth_smiles")
                unique_valid = list(dict.fromkeys(valid_smiles))
                if original_smi and unique_valid:
                    dist = get_distance(original_smi, unique_valid)
                    if dist is not None:
                        distance_vals.append(dist)

        # Average across groups
        result: Dict[str, Any] = {}

        if validity_vals:
            result[f"{self.task_name}/validity"] = float(
                np.mean(validity_vals)
            )
        if uniqueness_vals:
            result[f"{self.task_name}/uniqueness"] = float(
                np.mean(uniqueness_vals)
            )
        if diversity_vals:
            result[f"{self.task_name}/diversity"] = float(
                np.mean(diversity_vals)
            )
        if quality_vals:
            result[f"{self.task_name}/quality"] = float(np.mean(quality_vals))
        if distance_vals:
            result[f"{self.task_name}/distance"] = float(
                np.mean(distance_vals)
            )

        # Also add QED / SA summary stats (flat, not per-group -- useful for
        # logging but not part of GenMol's per-group protocol)
        if self.compute_qed:
            result.update(
                {
                    f"{self.task_name}/{k}": v
                    for k, v in self._aggregate_qed_stats(
                        [p for g in groups for p in g]
                    ).items()
                }
            )
        if self.compute_sa:
            result.update(
                {
                    f"{self.task_name}/{k}": v
                    for k, v in self._aggregate_sa_stats(
                        [p for g in groups for p in g]
                    ).items()
                }
            )

        return result


# endregion: Post-Hoc Evaluators
################################################################################
