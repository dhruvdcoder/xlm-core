import os
from pathlib import Path
from pprint import pformat
from typing import (
    Callable,
    Dict,
    List,
    Literal,
    Protocol,
    Sequence,
    Union,
    TypedDict,
    Mapping,
    Any,
    Optional,
    cast,
)
import shutil
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import (
    RandomSampler,
    StatefulDistributedSampler,
)
from jaxtyping import Integer
from torch import Tensor as TT
from torch.utils.data import DataLoader, IterableDataset, SequentialSampler
from datasets.distributed import split_dataset_by_node
import torch
from transformers import (
    BatchEncoding,
    PreTrainedTokenizer,
    BertTokenizer as _BertTokenizer,
    BertTokenizerFast as _BertTokenizerFast,
    GPT2Tokenizer as _GPT2Tokenizer,
    GPT2TokenizerFast as _GPT2TokenizerFast,
    PreTrainedTokenizerBase,
)
from lightning.pytorch.strategies import DDPStrategy as LDDPStrategy
from lightning.pytorch.strategies import (
    SingleDeviceStrategy as LSingleDeviceStrategy,
)
from lightning.fabric.strategies import DDPStrategy as FDDPStrategy
from lightning.fabric.strategies import (
    SingleDeviceStrategy as FSingleDeviceStrategy,
)
import lightning as L
from xlm import flags
from xlm.utils.imports import get_function
from xlm.utils.rank_zero import rank_zero_only
from xlm.noise import NoiseSchedule
from xlm.utils.rank_zero import RankedLogger
import datasets
from transformers import (
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
    AddedToken,
)

logger = RankedLogger(__name__, rank_zero_only=True)
ranked_logger = RankedLogger(__name__, rank_zero_only=False)

safe_imported = False
try:
    import safe as sf
    from safe.tokenizer import SAFETokenizer as _SAFETokenizer

    safe_imported = True
except ImportError:
    safe_imported = False
    _SAFETokenizer = _BertTokenizer


################################################################################
# region: Types

SingleDeviceStrategy = (LSingleDeviceStrategy, FSingleDeviceStrategy)
DDPStrategy = (LDDPStrategy, FDDPStrategy)


class Tokenizer(Protocol):
    mask_token_id: int
    pad_token_id: int
    cls_token_id: int
    eos_token_id: int
    bos_token_id: int
    mask_token: str
    pad_token: str
    cls_token: str
    eos_token: str
    bos_token: str

    def __call__(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        return_tensors: Optional[str] = None,
    ) -> BatchEncoding: ...

    @property
    def vocab_size(self) -> int: ...

    def __len__(self) -> int: ...

    def decode(
        self,
        token_ids: Union[List[int], Integer[TT, " seq_len"]],
        skip_special_tokens: bool = True,
    ) -> str: ...

    def batch_decode(
        self,
        token_ids: Union[List[List[int]], Integer[TT, " batch seq_len"]],
        skip_special_tokens: bool = True,
    ) -> List[str]: ...

    def convert_tokens_to_ids(
        self, tokens: Union[str, List[str]]
    ) -> Union[int, List[int]]: ...

    def convert_ids_to_tokens(
        self, ids: Union[int, List[int]]
    ) -> Union[str, List[str]]: ...


class BaseBatch(TypedDict):
    """Dict with the keys that are present in input batches for all models.

    Attributes:
        input_ids (Integer[TT, " batch seq_len"]): The input ids to the model.
        attention_mask (Integer[TT, " batch seq_len"]): 1 for tokens that are not padding.
        token_type_ids (Integer[TT, " batch seq_len"]): Can depend on the model type.
            For ILM and IDLM: 0 for CLS, 1 for BOS and prefix, 2 for other tokens.
    """

    input_ids: Integer[TT, " batch seq_len"]
    attention_mask: Integer[TT, " batch seq_len"]
    token_type_ids: Integer[
        TT, " batch seq_len"
    ]  # CLS=0, BOS+REST>=1, prefix=1, non_prefix=2
    # CLEANUP: There is no use for token_type_ids for ARLM and MLM. Instead we should only have constraint tensor.


class DataLoaderKwargs(TypedDict):
    batch_size: int
    num_workers: int
    shuffle: bool
    pin_memory: bool


class Collator(Protocol):
    tokenizer: Tokenizer
    block_size: int
    noise_schedule: NoiseSchedule

    def __call__(
        self,
        batch: List[Mapping[str, Any]],
    ) -> Mapping[str, Any]: ...


# endregion: Types
################################################################################


################################################################################
# region: Tokenizers


class TokenizerMixin:
    """
    1. Adds a `full_vocab_size` property.
    2. provides a `post_creation` method that should be called after creating the tokenizer.
      to ensure all the special tokens are present.
    """

    @property
    def full_vocab_size(self) -> int:
        return self.__len__()

    def post_creation(self):
        """Check the presence of the special tokens and update the post processor."""
        for special_token in [
            "eos_token",
            "bos_token",
            "cls_token",
            "pad_token",
            "mask_token",
            "sep_token",
            "unk_token",
        ]:
            if (token := getattr(self, special_token)) is None:
                raise ValueError(f"{special_token} is not set")


class BertTokenizer(TokenizerMixin, _BertTokenizer):  # type: ignore
    def __init__(self, *args, **kwargs):  # type: ignore
        super().__init__(*args, **kwargs)  # type: ignore
        self.add_special_tokens(
            {"cls_token": "[CLS]", "bos_token": "[BOS]", "eos_token": "[EOS]"}
        )

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, *args, **kwargs
    ):
        tokenizer = super().from_pretrained(
            pretrained_model_name_or_path, *args, **kwargs
        )
        tokenizer.post_creation()
        return tokenizer


class BertTokenizerFast(TokenizerMixin, _BertTokenizerFast):  # type: ignore
    def __init__(self, *args, **kwargs):  # type: ignore
        super().__init__(*args, **kwargs)  # type: ignore
        self.add_special_tokens(
            {"cls_token": "[CLS]", "bos_token": "[BOS]", "eos_token": "[EOS]"}
        )

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, *args, **kwargs
    ):
        tokenizer = super().from_pretrained(
            pretrained_model_name_or_path, *args, **kwargs
        )
        tokenizer.post_creation()
        return tokenizer


class GPT2Tokenizer(TokenizerMixin, _GPT2Tokenizer):  # type: ignore
    def __init__(self, *args, **kwargs):  # type: ignore
        super().__init__(*args, **kwargs)  # type: ignore
        self.add_special_tokens(
            {
                "cls_token": "<|cls|>",
                "bos_token": "<|bos|>",
                "pad_token": "<|pad|>",
                "unk_token": "<|unk|>",
                "mask_token": "<|mask|>",
                "sep_token": "<|sep|>",
                "eos_token": "<|endoftext|>",  # original
            }
        )

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, *args, **kwargs
    ):
        tokenizer = super().from_pretrained(
            pretrained_model_name_or_path, *args, **kwargs
        )
        tokenizer.post_creation()
        return tokenizer


class GPT2TokenizerFast(TokenizerMixin, _GPT2TokenizerFast):  # type: ignore
    def __init__(self, *args, **kwargs):  # type: ignore
        super().__init__(*args, **kwargs)  # type: ignore
        self.add_special_tokens(
            {
                "cls_token": "<|cls|>",
                "bos_token": "<|bos|>",
                "pad_token": "<|pad|>",
                "unk_token": "<|unk|>",
                "mask_token": "<|mask|>",
                "sep_token": "<|sep|>",
                "eos_token": "<|endoftext|>",  # original
            }
        )

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, *args, **kwargs
    ):
        tokenizer = super().from_pretrained(
            pretrained_model_name_or_path, *args, **kwargs
        )
        tokenizer.post_creation()
        return tokenizer


class SAFETokenizer(TokenizerMixin, _SAFETokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _tok = self.get_pretrained()
        _tok.add_tokens(["<", ">"])  # for bracket_safe

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, *args, **kwargs
    ):
        tokenizer = super().from_pretrained(
            pretrained_model_name_or_path, *args, **kwargs
        )
        tokenizer.mask_token = tokenizer.tokenizer.mask_token
        tokenizer.pad_token = tokenizer.tokenizer.pad_token
        tokenizer.cls_token = tokenizer.tokenizer.cls_token
        tokenizer.eos_token = tokenizer.tokenizer.eos_token
        tokenizer.bos_token = tokenizer.tokenizer.bos_token
        tokenizer.sep_token = tokenizer.tokenizer.sep_token
        tokenizer.unk_token = tokenizer.tokenizer.unk_token
        tokenizer.post_creation()

        return tokenizer


class SimpleSpaceTokenizer(PreTrainedTokenizer):
    """Splits on spaces"""

    def __init__(self, vocab: Sequence[str], **kwargs):
        """
        Args:
            vocab (Sequence[str]): List of desired tokens. Following are list of all of the special tokens with
                their corresponding ids:
                    "[PAD]": 0,
                    "[UNK]": 1,
                    "[MASK]": 2,
                    "[EOS]": 3,
                    "[BOS]": 4,
                an id (starting at 5) will be assigned to each character.

            model_max_length (int): Model maximum sequence length.
        """
        self.vocab = vocab
        eos_token = AddedToken("[EOS]", lstrip=False, rstrip=False)
        cls_token = AddedToken("[CLS]", lstrip=False, rstrip=False)
        pad_token = AddedToken("[PAD]", lstrip=False, rstrip=False)
        bos_token = AddedToken("[BOS]", lstrip=False, rstrip=False)
        mask_token = AddedToken("[MASK]", lstrip=True, rstrip=False)
        sep_token = AddedToken("[SEP]", lstrip=False, rstrip=False)
        unk_token = AddedToken("[UNK]", lstrip=False, rstrip=False)

        self._vocab_str_to_int = {
            "[PAD]": 0,
            "[CLS]": 1,
            "[MASK]": 2,
            "[EOS]": 3,
            "[BOS]": 4,
            "[SEP]": 5,
            "[UNK]": 6,
            **{ch: i + 7 for i, ch in enumerate(vocab)},
        }
        self._vocab_int_to_str = {
            v: k for k, v in self._vocab_str_to_int.items()
        }

        super().__init__(
            eos_token=eos_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            bos_token=bos_token,
            sep_token=sep_token,
            unk_token=unk_token,
            add_prefix_space=False,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    @property
    def full_vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    def tokenize(self, text: str, **kwargs) -> List[str]:
        # suppose text is a like "split1 split2 split3", convert to character if split* not in vocab
        tokens = []
        for token in text.split(" "):
            if not token:  # skip empty tokens
                continue
            if token not in self._vocab_str_to_int:
                raise ValueError(f"Token {token} not in vocab")
            tokens.append(token)
        return tokens

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int[token]  # let it raise keyerror

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str[index]

    def _decode(
        self,
        token_ids: Union[int, List[int]],
        skip_special_tokens: bool = False,
        **kwargs,
    ) -> str:
        return " ".join(
            [self._vocab_int_to_str[token_id] for token_id in token_ids]
        )

    def convert_tokens_to_string(self, tokens):
        return " ".join(tokens)

    def get_vocab(self) -> Dict[str, int]:
        return self._vocab_str_to_int

    @classmethod
    def for_numbers(cls, vocab_size: int, **kwargs) -> "SimpleSpaceTokenizer":
        return cls(vocab=list(map(str, range(vocab_size))), **kwargs)

    def save_pretrained(
        self, save_directory: Union[str, os.PathLike], **kwargs
    ):
        raise NotImplementedError

    @classmethod
    def from_pretrained(
        cls, save_directory: Union[str, os.PathLike], **kwargs
    ):
        raise NotImplementedError

    def from_txt(cls, txt_file: Union[str, os.PathLike], **kwargs):
        vocab = []
        with open(txt_file, "r") as f:
            for line in f:
                vocab.append(line.strip())
        return cls(vocab=vocab, **kwargs)


# endregion: Tokenizers
################################################################################

################################################################################
# region: Preprocess functions


def tokenizing_preprocess_fn(
    example: Dict[str, Any], tokenizer: PreTrainedTokenizerBase
) -> Dict[str, Any]:
    example["token_ids"] = tokenizer.encode(  # type: ignore
        example["text"], add_special_tokens=False
    )
    return example


# endregion: Preprocess functions
################################################################################


################################################################################
# region: Base DataModule


class DatasetManager:
    def __init__(
        self,
        collator: Collator,  # can depend on the model
        # full_name: str,  # e.g. "repo/ds_name/<split>/<type>", eg. `/lm1b/train/lm`
        full_name: str,  # e.g. "repo/ds_name/split", eg. `one-billion-word-benchmark/lm1b/train`
        full_name_debug: str,  # Used with DEBUG_OVERFIT is set to True. So if this manager is for val or test split, replace the split with train here..
        # split_to_download: str,  # e.g. "train", "val", "test", "predict"
        dataloader_kwargs: DataLoaderKwargs,  # e.g. {"batch_size": 1, "num_workers": 1, "shuffle": True, "pin_memory": True}
        preprocess_function: Optional[str] = None,  # e.g. "preprocess_fn"
        preprocess_function_kwargs: Optional[Dict[str, Any]] = None,
        on_the_fly_processor: Optional[str] = None,  # e.g. "ids_to_example_fn"
        on_the_fly_processor_kwargs: Optional[
            Dict[str, Any]
        ] = None,  # e.g. {"block_size": 128}
        on_the_fly_group_processor: Optional[
            str
        ] = None,  # e.g. "group_texts_ilm"
        model_name: Optional[
            str
        ] = None,  # used to create model specific cache dir if supplied
        columns_to_remove: Optional[List[str]] = None,  # e.g. ["text"]
        stages: List[
            Literal["fit", "validate", "test", "predict"]
        ] = None,  # e.g. ["fit", "validate"]
        iterable_dataset_shards: Optional[int] = None,  # e.g. 1000
        shuffle_buffer_size: Optional[int] = None,  # e.g. 10000
        shuffle_seed: Optional[int] = 42,  # e.g. 42
        split_by_node: bool = True,  # e.g. True
        dataset: Optional[datasets.Dataset] = None,  # only set later
        rewrite_manual_cache: bool = False,
        use_manual_cache: bool = True,
    ):
        self.collator = collator
        self._full_name = full_name
        self._full_name_debug = full_name_debug
        if flags.DEBUG_OVERFIT:
            logger.info(
                f"Using {self._full_name_debug} instead of {full_name} because DEBUG_OVERFIT is set to True"
            )
            self.full_name = self._full_name_debug
        else:
            self.full_name = full_name
        # self.split_to_download = split_to_download
        self.dataloader_kwargs = dataloader_kwargs
        self.preprocess_function = preprocess_function
        self.preprocess_function_kwargs = preprocess_function_kwargs or {}
        self.on_the_fly_processor = on_the_fly_processor
        self.on_the_fly_processor_kwargs = on_the_fly_processor_kwargs or {}
        self.on_the_fly_group_processor = on_the_fly_group_processor
        self.model_name = model_name
        self.columns_to_remove = columns_to_remove
        self.stages = stages if stages is not None else ["fit", "validate"]
        self.iterable_dataset_shards = iterable_dataset_shards
        self.shuffle_buffer_size = shuffle_buffer_size
        self.shuffle_seed = shuffle_seed
        self.split_by_node = split_by_node
        self.dataset = dataset
        self.dataset = None
        self.is_iterable_dataset = iterable_dataset_shards is not None
        self.rewrite_manual_cache = rewrite_manual_cache
        self.use_manual_cache = use_manual_cache

    def __repr__(self) -> str:
        return f"DatasetManager(full_name={self.full_name})"

    @property
    def _split_to_download(self) -> str:
        return self.full_name.split("/")[-1]

    # @property
    # def type(self) -> str:
    #    return self.full_name.split("/")[-1]

    # @property
    # def split(self) -> Literal["train", "val", "test", "predict"]:
    #    return self.full_name.split("/")[-2]

    @property
    def name(self) -> str:
        return "/".join(self.full_name.split("/")[:2])

    def set_epoch(self, epoch: int) -> None:
        # only datasets.IterableDataset has set_epoch to shuffle the buffer
        # TODO (fabric): For map-style datasets that use DistributedSampler, one needs to
        # call set_epoch on the DistributedSampler of the dataloader.
        if hasattr(self.dataset, "set_epoch") and not flags.DEBUG_OVERFIT:
            self.dataset.set_epoch(epoch)  # type: ignore

    def _download(self, num_proc: Optional[int] = None) -> datasets.Dataset:
        name = self.name
        if name[0] == "/":
            name = name[1:]
        logger.info(f"Downloading {self.full_name}")
        ds = datasets.load_dataset(
            name,
            split=self._split_to_download,
            num_proc=num_proc,
            token=os.environ.get("HF_HUB_KEY"),
        )
        return ds

    def _preprocess(
        self,
        ds: datasets.Dataset,
        tokenizer: Tokenizer,
        num_proc: Optional[int] = None,
    ) -> datasets.Dataset:
        if self.preprocess_function is not None:
            preprocess_fn: Callable[..., Any] = get_function(
                self.preprocess_function
            )
            fn_kwargs = {
                "tokenizer": tokenizer,
                **self.preprocess_function_kwargs,
            }
            ds = ds.map(
                preprocess_fn,
                batched=False,
                num_proc=num_proc,
                fn_kwargs=fn_kwargs,
                remove_columns=self.columns_to_remove,
            )
        return ds

    def _get_cache_dir(self, manual_cache_dir: str) -> Path:
        return Path(manual_cache_dir) / self.full_name

    def _clean_manual_cache(self, manual_cache_dir: str) -> None:
        cache_dir = self._get_cache_dir(manual_cache_dir)
        logger.info(
            f"Cleaning manual cache for {self.full_name} at {cache_dir}"
        )
        # Fallback to pathlib if fsspec fails
        shutil.rmtree(cache_dir, ignore_errors=True)

    def _manually_cache(
        self, ds: datasets.Dataset, manual_cache_dir: str
    ) -> None:
        cache_dir = self._get_cache_dir(manual_cache_dir)
        logger.info(f"Caching {self.full_name} to {cache_dir}")
        ds.save_to_disk(cache_dir, num_shards=self.iterable_dataset_shards)

    def _load_from_cache(self, manual_cache_dir: str) -> datasets.Dataset:
        cache_dir = self._get_cache_dir(manual_cache_dir)
        return datasets.load_from_disk(cache_dir)

    def _check_cache(self, manual_cache_dir: str) -> bool:
        cache_dir = self._get_cache_dir(manual_cache_dir)
        return cache_dir.exists()

    def _apply_on_the_fly_processors(
        self, dataset: datasets.Dataset, tokenizer: Tokenizer
    ):
        if self.on_the_fly_processor is not None:
            processor: Callable = get_function(self.on_the_fly_processor)
            kwargs = {
                "tokenizer": tokenizer,
                **self.on_the_fly_processor_kwargs,
            }
            dataset = dataset.map(processor, batched=False, fn_kwargs=kwargs)
            return dataset
        return dataset

    def _apply_on_the_fly_group_processors(
        self,
        dataset: datasets.Dataset,
        tokenizer: Tokenizer,
        stage: Literal["fit", "validate", "test", "predict"],
        block_size: int,
        pad_id: int,
        type_id_extension: int = 2,
        attn_extension: int = 0,
    ):
        if self.on_the_fly_group_processor is not None:
            group_processor: Callable = get_function(
                self.on_the_fly_group_processor
            )
            dataset = dataset.map(
                group_processor,
                batched=True,
                fn_kwargs={
                    "tokenizer": tokenizer,
                    "block_size": self.block_size,
                    "block_size": block_size,
                    "pad_id": pad_id,
                    "type_id_extension": type_id_extension,
                    "attn_extension": attn_extension,
                },
            )
            return dataset
        return dataset

    def prepare_data(
        self,
        manual_cache_dir: str,
        tokenizer: Tokenizer,
        num_proc: Optional[int] = None,
        load: bool = False,
    ) -> Optional[datasets.Dataset]:
        # check cache, download, preprocess, cache if not already cached.
        if self.use_manual_cache:
            if not self._check_cache(manual_cache_dir):
                logger.info(
                    f"No manual cache for {self.full_name} found at {manual_cache_dir}"
                )
                ds = self._download(num_proc=num_proc)
                ds = self._preprocess(ds, tokenizer, num_proc=num_proc)
                self._manually_cache(ds, manual_cache_dir)
            elif self.rewrite_manual_cache:
                logger.info(
                    f"Rewriting manual cache for {self.full_name} at {manual_cache_dir}"
                )
                ds = self._download(num_proc=num_proc)
                logger.info(
                    f"Cleaned {ds.cleanup_cache_files()} automatic cached files"
                )
                self._clean_manual_cache(manual_cache_dir)
                ds = self._preprocess(ds, tokenizer, num_proc=num_proc)
                self._manually_cache(ds, manual_cache_dir)
            else:
                logger.info(
                    f"Found manual cache for {self.full_name} at {self._get_cache_dir(manual_cache_dir)}"
                )
                if load:
                    ds = self._load_from_cache(manual_cache_dir)
                else:
                    ds = None
        else:
            logger.info(
                f"Downloading {self.full_name} without using manual cache"
            )
            ds = self._download(num_proc=num_proc)
            ds = self._preprocess(ds, tokenizer, num_proc=num_proc)
        return ds

    def setup(
        self,
        stage: Literal["fit", "validate", "test", "predict"],
        manual_cache_dir: str,
        tokenizer: Tokenizer,
        block_size: int,
        is_ddp: bool,
        rank: int,
        world_size: int,
        num_dataset_workers: Optional[int] = None,
    ) -> None:
        if stage in self.stages and self.dataset is None:
            if self.use_manual_cache:
                dataset = self._load_from_cache(manual_cache_dir)
            else:
                dataset = self.prepare_data(
                    manual_cache_dir,
                    tokenizer,
                    num_proc=num_dataset_workers,
                    load=True,
                )
            if self.is_iterable_dataset:
                dataset = dataset.to_iterable_dataset(
                    num_shards=self.iterable_dataset_shards
                )
            if (
                self.shuffle_buffer_size is not None
                and not flags.DEBUG_OVERFIT
            ):
                dataset = dataset.shuffle(
                    buffer_size=self.shuffle_buffer_size,
                    seed=self.shuffle_seed,
                )
            dataset = self._apply_on_the_fly_processors(dataset, tokenizer)
            dataset = self._apply_on_the_fly_group_processors(
                dataset,
                tokenizer,
                stage,
                block_size=block_size,
                pad_id=tokenizer.pad_token_id,
                type_id_extension=2,
                attn_extension=0,
            )
            self.dataset = dataset
            if (
                is_ddp
                and world_size > 1
                and self.split_by_node
                and self.is_iterable_dataset
            ):
                logger.info(
                    f"Splitting dataset {self.full_name} by node for {world_size=}"
                )
                self.dataset = split_dataset_by_node(
                    self.dataset, rank=rank, world_size=world_size
                )

    def get_dataloader(
        self,
        type: Literal["train", "val", "test", "predict"],
        is_ddp: bool,
        rank: int,
        world_size: int,
    ) -> Union[DataLoader, StatefulDataLoader]:
        # case 1: DDP and IterDataset=> Dataset is stateful, handles shuffling and already split_by_node. No sampler needed.
        # case 2: DDP and MapDataset=> Need StatefulDistributedSampler with deterministic seed.
        # case 3: Single GPU and MapDataset=> Use StatefulRandomSampler with deterministic seed. Simply setting shuffle=True on StatefulDataloader is equivalent.
        # case 4: Single GPU and IterDataset=> Dataset is stateful and handles shuffling. Does not need splitting.
        if (
            type == "train"
            and is_ddp
            and world_size > 1
            and self.is_iterable_dataset
        ):  # case 1
            DL = StatefulDataLoader
            logger.info(
                f"Using StatefulDataLoader for {type} of {self.full_name} dataloader"
            )
            sampler = None
            # remove shuffle from dataloader kwargs if provided
            if "shuffle" in self.dataloader_kwargs:
                logger.warning(
                    f"Shuffle is set to True for {self.full_name} {type} dataloader for DDP with IterableDataset. This will be ignored as shuffling is handled by the dataset in this case."
                )
                self.dataloader_kwargs.pop("shuffle")
            # Checks:
            num_workers = self.dataloader_kwargs.get("num_workers", 1)
            assert (
                self.iterable_dataset_shards is not None
            ), "iterable_dataset_shards must be set"
            num_shards_per_worker = self.iterable_dataset_shards / (
                world_size * num_workers
            )
            per_worker_batch_size = self.dataloader_kwargs.get("batch_size", 1)
            if num_shards_per_worker > per_worker_batch_size:
                raise ValueError(
                    f"{num_shards_per_worker=} > {per_worker_batch_size=}. This may cause the training to hang even with drop_last=True. "
                    "Either increase the num_workers or world_size to bring down the num_shards_per_worker or reduce the num_shards or increase the batch_size."
                )
        elif (
            type == "train"
            and is_ddp
            and world_size > 1
            and not self.is_iterable_dataset
        ):  # case 2
            # We setup the generator here because we want its seed to be
            # be dependent on the global seed. Letting the sampler create
            # its own generator will use a hardcoded seed of 0 as of today (https://github.com/pytorch/data/issues/1440)
            # while issue 1440 was resolved for RandomSampler, I believe it still exists for StatefulDistributedSampler.
            # Therefore, we continue to send our own seed.
            DL = StatefulDataLoader
            logger.info(
                f"Using StatefulDataLoader for {type} of {self.full_name} dataloader"
            )
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            sampler = StatefulDistributedSampler(
                self.dataset,
                seed=seed,
                shuffle=True if not flags.DEBUG_OVERFIT else False,
                # num_replicas=world_size, # Let it retrieve these values from the distributed context
                # rank=rank,
            )
        elif (
            type == "train" and not is_ddp and not self.is_iterable_dataset
        ):  # case 3
            DL = StatefulDataLoader
            sampler = (
                RandomSampler(self.dataset)
                if not flags.DEBUG_OVERFIT
                else SequentialSampler(self.dataset)
            )
        elif (
            type == "train" and not is_ddp and self.is_iterable_dataset
        ):  # case 4
            DL = StatefulDataLoader
            sampler = None
        elif type in ["val", "test", "predict"]:
            sampler = None
            DL = DataLoader
            if self.dataloader_kwargs.get("shuffle", False):
                logger.warning(
                    f"Shuffle is set to True for {type} dataloader. This will be ignored as {type} dataloaders are not shuffled."
                )
            self.dataloader_kwargs["shuffle"] = False
        else:
            raise ValueError(f"Invalid dataloader type: {type}")

        dataloader = DL(
            self.dataset,
            collate_fn=self.collator,
            sampler=sampler,
            **self.dataloader_kwargs,
        )
        return dataloader


class LocalDatasetManager(DatasetManager):
    def __init__(
        self,
        *args,
        ds_type: Optional[str] = None,
        load_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        if ds_type is None:
            raise ValueError("ds_type is required")
        super().__init__(*args, **kwargs)
        self.load_kwargs = load_kwargs or {}
        self.ds_type = ds_type

    def _download(self, num_proc: Optional[int] = None) -> datasets.Dataset:
        if self.ds_type == "csv":
            # If data_files is specified in load_kwargs, use it; otherwise construct from path
            load_kwargs_copy = self.load_kwargs.copy()
            if "data_files" in load_kwargs_copy:
                data_files = load_kwargs_copy.pop("data_files")
                ds = datasets.load_dataset(
                    "csv",
                    data_files=data_files,
                    **load_kwargs_copy,
                    num_proc=num_proc,
                )["train"]
            else:
                file_name = f"{self._split_to_download}.csv"
                _path = Path(self.full_name).parent
                data_files = str(_path / file_name)
                ds = datasets.load_dataset(
                    "csv",
                    data_files=data_files,
                    **self.load_kwargs,
                    num_proc=num_proc,
                )["train"]
            return ds
        else:
            raise ValueError(f"Unsupported dataset type: {self.ds_type}")


class UnconditionalGenerationDatasetManager:
    """
    This is used for unconditional generation, where we don't have any input text.
    """

    def __init__(
        self,
        dataset_constructor_str: str,  # like "xlm.lm.ilm.datamodule_ilm.ILMEmptyDataset"
        collator: Collator,
        num_examples: int,
        dataloader_kwargs: DataLoaderKwargs,
        split_by_node: bool = True,
        dataset_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.num_examples = num_examples
        self.split_by_node = split_by_node
        self.dataset_constructor: Callable[
            [Tokenizer, int], IterableDataset
        ] = get_function(dataset_constructor_str)
        self.collator = collator
        self.dataloader_kwargs = dataloader_kwargs
        self.dataset_kwargs = dataset_kwargs or {}

    def __repr__(self) -> str:
        return f"ILMUnconditionalGenerationDatasetManager()"

    @property
    def name(self) -> str:
        return "empty"

    def set_epoch(self, epoch: int) -> None:
        return

    def prepare_data(
        self,
        manual_cache_dir: str,
        tokenizer: Tokenizer,
        num_proc: Optional[int] = None,
    ) -> datasets.Dataset:
        return

    def setup(
        self,
        stage: Literal["fit", "validate", "test", "predict"],
        manual_cache_dir: str,
        tokenizer: Tokenizer,
        block_size: int,
        is_ddp: bool,
        rank: int,
        world_size: int,
        num_dataset_workers: Optional[int] = None,
    ) -> None:
        # if is_ddp and world_size > 1 and self.split_by_node:
        #    examples_per_node = self.num_examples // world_size
        # else:
        #    examples_per_node = self.num_examples
        examples_per_node = self.num_examples
        dataset = self.dataset_constructor(
            tokenizer,
            examples_per_node,
            **self.dataset_kwargs,
        )
        self.dataset = dataset

    def get_dataloader(
        self,
        type: Literal["train", "val", "test", "predict"],
        is_ddp: bool,
        rank: int,
        world_size: int,
    ) -> Union[DataLoader, StatefulDataLoader]:
        if type == "train":
            raise ValueError(
                "Train dataloader is not supported for unconditional generation."
            )
        return DataLoader(
            self.dataset, collate_fn=self.collator, **self.dataloader_kwargs
        )


class BaseDataModule(L.LightningDataModule):
    """
    Base class for all datamodules.
    """

    no_trainer_mode: bool = False
    dataloader_names: Dict[
        Literal["train", "val", "test", "predict"], Dict[int, str]
    ]
    dataloader_ids: Dict[
        Literal["train", "val", "test", "predict"], Dict[str, int]
    ]
    tokenizer: Tokenizer
    """The tokenizer.[Required]"""
    global_batch_size: int
    train_dataloader_batch_size: int

    def __init__(self, *args, **kwargs):  # type: ignore
        super().__init__()  # LightningDataModule.__init__() does not take any arguments
        self.prepare_data_per_node = False
        self.no_trainer_mode = False

    def print_batch(
        self,
        batch: BaseBatch,
        split: Literal["train", "val", "test", "predict"],
        dataloader_idx: Optional[int] = None,
    ):
        """Required to print train and validation batches at the beginning of the epoch."""
        raise NotImplementedError("Not implemented")

    def _check_grad_accum(self):
        if self.trainer is None:
            logger.warning("Trainer is not setup. Cannot check grad accum.")
            return
        if self._is_ddp():
            num_nodes = self.trainer.num_nodes
            num_gpus_per_node = self.trainer.num_devices
            accum_steps = self.trainer.accumulate_grad_batches
            logger.info(
                f"global_batch_size: {self.global_batch_size} | num_nodes: {num_nodes} | num_gpus_per_node: {num_gpus_per_node} | accum_steps: {accum_steps}"
            )
            if (
                self.global_batch_size
                % (
                    self.train_dataloader_kwargs["batch_size"]
                    * num_nodes
                    * num_gpus_per_node
                    * accum_steps
                )
                != 0
            ):
                raise ValueError(
                    f"Global batch size ({self.global_batch_size}) is not equal to "
                    f"per_device_batch_size ({self.train_dataloader_kwargs['batch_size']}) * num_nodes ({num_nodes}) * num_gpus_per_node ({num_gpus_per_node}) * accum_steps ({accum_steps})."
                )

    def _is_ddp(self) -> bool:
        if self.trainer is not None:
            strategy = self.trainer.strategy
            if isinstance(strategy, DDPStrategy):
                return True
            elif isinstance(strategy, SingleDeviceStrategy):
                return False
            else:
                raise ValueError(
                    f"Dataloader does not support {type(strategy)} strategy"
                )
        else:
            logger.warning(
                "Tried to detect DDP strategy before trainer was set."
                " Are you calling `LightningDataModule.*_dataloader()` methods manually?"
                " Make sure you know what you are doing!"
            )
            return False

    @property
    def rank(self) -> int:
        if self.trainer is not None:
            return self.trainer.global_rank
        elif self.no_trainer_mode:
            logger.warning(
                "No trainer mode. Returning rank 0. If using multiple GPUs, this will result in incorrect results."
            )
            return 0
        else:
            raise ValueError("Trainer is not set")

    @property
    def world_size(self) -> int:
        if self.trainer is not None:
            return self.trainer.world_size
        elif self.no_trainer_mode:
            logger.warning(
                "No trainer mode. Returning world size 1. If using multiple GPUs, this will result in incorrect results."
            )
            return 1
        else:
            raise ValueError("Trainer is not set")


class TextDataModule(BaseDataModule):
    def __init__(
        self,
        manual_cache_dir: str,
        tokenizer: Tokenizer,
        # dataset_managers: List[DatasetManager],
        dataset_managers: Dict[
            Literal["train", "val", "test", "predict"],
            Dict[str, DatasetManager],  # k=dl_name
        ],
        noise_schedule: Optional[NoiseSchedule] = None,
        rewrite_manual_cache: bool = False,
        block_size: int = 128,
        global_batch_size: int = 512,
        verbosity: Literal["warning", "info", "debug"] = "info",
        num_unconditional_samples: Optional[int] = None,
        num_dataset_workers: Optional[int] = None,
        print_batch_fn: Optional[str] = None,
    ):
        super().__init__()
        self.manual_cache_dir = manual_cache_dir
        self.num_dataset_workers = num_dataset_workers
        self.dataset_managers = dataset_managers
        self.noise_schedule = noise_schedule
        self.block_size = block_size
        self.global_batch_size = global_batch_size
        self.num_unconditional_samples = num_unconditional_samples
        self.tokenizer = tokenizer
        self.rewrite_manual_cache = rewrite_manual_cache
        datasets.utils.logging.set_verbosity(
            datasets.utils.logging.log_levels[verbosity]
        )
        # dataloader names
        self.dataloader_names = {
            "train": {},
            "val": {},
            "test": {},
            "predict": {},
        }
        self.dataloader_ids = {
            "train": {},
            "val": {},
            "test": {},
            "predict": {},
        }
        # check the dataloader_ids for each dataset_manager
        for split, ds_managers in self.dataset_managers.items():
            if split not in ["train", "val", "test", "predict"]:
                raise ValueError(f"Invalid split {split} in dataset_managers")
            if split == "train" and len(ds_managers) != 1:
                raise ValueError(f"There should be exactly one train dataset")
            for dataloader_idx, (
                dataloader_name,
                ds_manager,
            ) in enumerate(ds_managers.items()):
                self.dataloader_names[split][dataloader_idx] = dataloader_name
                self.dataloader_ids[split][dataloader_name] = dataloader_idx

        # determine train dataloader kwargs
        for split, ds_managers in self.dataset_managers.items():
            if split == "train":
                for dataloader_name, ds_manager in ds_managers.items():
                    self.train_dataloader_kwargs = ds_manager.dataloader_kwargs
                    self.train_dataloader_batch_size = (
                        ds_manager.dataloader_kwargs["batch_size"]
                    )

        dl_names_str = pformat(self.dataloader_names)
        logger.info(f"Dataloader names:\n{dl_names_str}")
        if print_batch_fn is not None:
            self.print_batch_fn = get_function(print_batch_fn)
        else:
            self.print_batch_fn = print_batch_base

    def print_batch(
        self,
        batch,
        split: Literal["train", "val", "test", "predict"],
        dataloader_idx: Optional[int] = None,
    ):
        dl_name = (
            self.dataloader_names[split][dataloader_idx]
            if dataloader_idx is not None
            else ""
        )
        self.print_batch_fn(batch, split, self.tokenizer, dl_name)

    def prepare_data(self) -> None:
        ds_wrapper: DatasetManager
        for split, ds_managers in self.dataset_managers.items():
            for dl_name, ds_wrapper in ds_managers.items():
                ds_wrapper.prepare_data(
                    manual_cache_dir=self.manual_cache_dir,
                    tokenizer=self.tokenizer,
                    num_proc=self.num_dataset_workers,
                )

    def setup(
        self, stage: Literal["fit", "validate", "test", "predict"]
    ) -> None:
        ds_wrapper: DatasetManager
        for split, ds_managers in self.dataset_managers.items():
            for dl_name, ds_wrapper in ds_managers.items():
                ds_wrapper.setup(
                    stage,
                    manual_cache_dir=self.manual_cache_dir,
                    tokenizer=self.tokenizer,
                    block_size=self.block_size,
                    is_ddp=self._is_ddp(),
                    rank=self.rank,
                    world_size=self.world_size,
                    num_dataset_workers=self.num_dataset_workers,
                )

    def _get_dataloaders(
        self, split: Literal["train", "val", "test", "predict"]
    ) -> Any:
        dataloaders = []

        for split_, ds_managers in self.dataset_managers.items():
            for dl_name, ds_wrapper in ds_managers.items():
                if split_ == split:
                    dl = ds_wrapper.get_dataloader(
                        split,
                        is_ddp=self._is_ddp(),
                        rank=self.rank,
                        world_size=self.world_size,
                    )
                    dataloaders.append(dl)

        if split == "train" and len(dataloaders) != 1:
            raise ValueError(
                "there should be exactly one train dataset and dataloader"
            )
        return dataloaders if split != "train" else dataloaders[0]

    def train_dataloader(self) -> Any:
        return self._get_dataloaders("train")

    def val_dataloader(self) -> Any:
        return self._get_dataloaders("val")

    def test_dataloader(self) -> Any:
        return self._get_dataloaders("test")

    def predict_dataloader(self) -> Any:
        return self._get_dataloaders("predict")

    def set_epoch(self, epoch: int) -> None:
        ds_wrapper: DatasetManager
        for split, ds_managers in self.dataset_managers.items():
            for dl_name, ds_wrapper in ds_managers.items():
                ds_wrapper.set_epoch(epoch)


# endregion: Base DataModule
################################################################################


################################################################################
# region: Processors


def token_ids_to_input_ids(
    example: Dict[Literal["token_ids"], List[int]],
    tokenizer: PreTrainedTokenizerBase,
    block_size: Optional[int] = None,
):
    return {
        "input_ids": example["token_ids"][:block_size],
    }


def token_ids_to_input_ids_and_prompt_ids(
    example: Dict[Literal["token_ids"], List[int]],
    tokenizer: PreTrainedTokenizerBase,
    block_size: Optional[int] = None,
):
    return {
        "input_ids": example["input_token_ids"][:block_size],
        "prompt_ids": example["prompt_token_ids"][:block_size],
    }


def ids_to_example_fn(
    example: Dict[Literal["token_ids"], List[int]],
    tokenizer: PreTrainedTokenizerBase,
    block_size: Optional[int] = None,
) -> Dict[Literal["input_ids", "attention_mask", "token_type_ids"], List[int]]:
    """Convert raw token_ids to input_ids, attention_mask, and token_type_ids.

    Does:
        1. Calls `tokenizer.build_inputs_with_special_tokens` and `tokenizer.create_token_type_ids_from_sequences`
            to produce `input_ids` and `token_type_ids`.
        2. Creates an `attention_mask` of all ones.

    Does not do:
        1. Padding/truncation.

    Args:
        example: A dictionary with a "token_ids" key, and value which is a list of token ids.
        tokenizer: A tokenizer that implements `PretrainedTokenizerBase` interface.
            Specifically, it should have `build_inputs_with_special_tokens` and
            `create_token_type_ids_from_sequences` methods overridden if the default
            implementations are not correct.
        block_size: The block size to pad/truncate the input_ids to.
    Returns:
        A dictionary with "input_ids", "attention_mask", and "token_type_ids" keys.
    """
    input_ids = tokenizer.build_inputs_with_special_tokens(
        example["token_ids"]
    )
    attention_mask = [1] * len(input_ids)
    token_type_ids = tokenizer.create_token_type_ids_from_sequences(
        example["token_ids"]
    )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }


class DefaultEmptyDataset(IterableDataset):
    def __init__(
        self,
        tokenizer: Tokenizer,
        num_examples: int,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        empty_text: str = "",
    ):
        """
        Args:
            tokenizer_kwargs: Keyword arguments for the tokenizer.

            empty_text: For MLM, you will want to set the `empty_text` to a sequence of all mask tokens.
        """
        self.tokenizer = tokenizer
        self.num_examples = num_examples
        self.tokenizer_kwargs = tokenizer_kwargs or {}
        self.empty_text = empty_text

    def __iter__(self):
        for _ in range(self.num_examples):
            ex = self.tokenizer(
                self.empty_text,
                add_special_tokens=False,
                **self.tokenizer_kwargs,
            )
            yield ex


# endregion: Processors
################################################################################


################################################################################
# region: Collators


class BaseCollatorInput(TypedDict):
    """Dict with values that are lists of raw input_ids of variable length.

    This is the input to the collator for pre-training.

    The elements of the lists can be of different lengths.

    Attributes:
        input_ids (List[int]): The input ids.
    """

    input_ids: List[int]


class Seq2SeqCollatorInput(TypedDict):
    """Dict with values that are lists of raw input_ids, attention_mask, and token_type_ids.

    This is the input to the collator for pre-training.

    The elements of the lists can be of different lengths.

    Attributes:
        input_ids (List[int]): The input ids.
        prompt_ids (List[int]): The target ids.
    """

    input_ids: List[int]
    prompt_ids: List[int]


class InfillingCollatorInput(TypedDict):
    input_ids: List[int]  # inputs with blanks (masks)
    target_ids: List[int]  # targets with all tokens


class DefaultCollator(Collator):
    """Simply stacks the input_ids, attention_mask, and token_type_ids and returns a batch."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        block_size: int,
        noise_schedule: NoiseSchedule,
    ):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.noise_schedule = noise_schedule

    @property
    def vocab_size(self) -> int:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not set")
        return len(self.tokenizer)

    def __call__(
        self,
        examples: List[BaseCollatorInput],
    ) -> BaseBatch:
        return {
            "input_ids": torch.stack(
                [torch.tensor(e["input_ids"]) for e in examples]
            ),
            "attention_mask": torch.stack(
                [torch.tensor(e["attention_mask"]) for e in examples]
            ),
            "token_type_ids": torch.stack(
                [torch.tensor(e["token_type_ids"]) for e in examples]
            ),
        }


def pad_truncate(
    examples,
    max_len,
    pad_token_id,
    attn_extension: int = 0,
    type_extension: int = 2,
) -> BaseBatch:
    return {
        "input_ids": torch.tensor(
            [
                example["input_ids"][:max_len]
                + [pad_token_id] * max(0, max_len - len(example["input_ids"]))
                for example in examples
            ],
            dtype=torch.long,
        ),
        "attention_mask": torch.tensor(
            [
                example["attention_mask"][:max_len]
                + [attn_extension]
                * max(0, max_len - len(example["attention_mask"]))
                for example in examples
            ],
            dtype=torch.long,
        ),
        "token_type_ids": torch.tensor(
            [
                example["token_type_ids"][:max_len]
                + [type_extension]
                * max(0, max_len - len(example["token_type_ids"]))
                for example in examples
            ],
            dtype=torch.long,
        ),
    }


def pad_prefix_suffix(tokenizer, examples, max_seq_len) -> BaseBatch:
    """
    [<varibale prefix>] [<bos> <variable suffix> <eos>]
    """
    prefixes = []
    suffixes = []
    max_prefix_len = 0
    max_suffix_len = 0
    for example in examples:
        bos_index = example["input_ids"].index(tokenizer.bos_token_id)
        prefix = example["input_ids"][: bos_index + 1]
        max_prefix_len = max(max_prefix_len, len(prefix))
        suffix = example["input_ids"][bos_index + 1 :]
        max_suffix_len = max(max_suffix_len, len(suffix))
        prefixes.append(prefix)
        suffixes.append(suffix)
    return {
        "input_ids": torch.tensor(
            [
                [tokenizer.pad_token_id]
                * max(0, max_prefix_len - len(prefixes[i]))
                + prefixes[i][:max_prefix_len]
                + suffixes[i][:max_suffix_len]
                + [tokenizer.pad_token_id]
                * max(0, max_suffix_len - len(suffixes[i]))
                for i in range(len(examples))
            ][:max_seq_len],
            dtype=torch.long,
        ),
        "attention_mask": torch.tensor(
            [
                [0] * max(0, max_prefix_len - len(prefixes[i]))
                + [1] * len(prefixes[i])
                + [0] * len(suffixes[i])
                + [0] * max(0, max_suffix_len - len(suffixes[i]))
                for i in range(len(examples))
            ][:max_seq_len],
            dtype=torch.long,
        ).bool(),
        "token_type_ids": torch.tensor(
            [
                [2] * max(0, max_prefix_len - len(prefixes[i]))
                + [1] * len(prefixes[i])
                + [2] * len(suffixes[i])
                + [2] * max(0, max_suffix_len - len(suffixes[i]))
                for i in range(len(examples))
            ][:max_seq_len],
            dtype=torch.long,
        ),
    }


def pad_prefix_suffix2(
    examples,
    max_seq_len,
    pad_token_id,
    bos_token_id,
    eos_token_id,
    prefix_type_extension=0,
    suffix_type_extension=2,
    return_tensors=True,
) -> Union[BaseBatch, Dict[str, List[List[int]]]]:
    """
    [<varibale prefix>] [<bos> <variable suffix> <eos>]
    """
    prefixes = []
    suffixes = []
    max_prefix_len = 0
    max_suffix_len = 0
    for example in examples:
        bos_index = example["input_ids"].index(bos_token_id)
        prefix = example["input_ids"][: bos_index + 1]
        max_prefix_len = max(max_prefix_len, len(prefix))
        suffix = example["input_ids"][bos_index + 1 :]
        max_suffix_len = max(max_suffix_len, len(suffix))
        prefixes.append(prefix)
        suffixes.append(suffix)
    input_ids = [
        [pad_token_id] * max(0, max_prefix_len - len(prefixes[i]))
        + prefixes[i][:max_prefix_len]
        + suffixes[i][:max_suffix_len]
        + [pad_token_id] * max(0, max_suffix_len - len(suffixes[i]))
        for i in range(len(examples))
    ]
    attention_mask = [
        [0] * max(0, max_prefix_len - len(prefixes[i]))
        + [1] * len(prefixes[i])
        + [0] * len(suffixes[i])
        + [0] * max(0, max_suffix_len - len(suffixes[i]))
        for i in range(len(examples))
    ]
    token_type_ids = [
        [prefix_type_extension] * max(0, max_prefix_len - len(prefixes[i]))
        + [1] * len(prefixes[i])
        + [suffix_type_extension] * len(suffixes[i])
        + [suffix_type_extension] * max(0, max_suffix_len - len(suffixes[i]))
        for i in range(len(examples))
    ]
    if return_tensors:
        input_ids = torch.tensor(input_ids, dtype=torch.long)  # type: ignore
        attention_mask = torch.tensor(attention_mask, dtype=torch.bool)  # type: ignore
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)  # type: ignore

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }


def pad_left_truncate(
    examples,
    max_len,
    pad_token_id,
    attn_extension: int = 0,
    type_extension: int = 2,
    return_tensors: bool = True,
) -> Union[BaseBatch, Dict[str, List[List[int]]]]:
    input_ids: List[List[int]] = [
        [pad_token_id] * max(0, max_len - len(example["input_ids"]))
        + example["input_ids"][:max_len]
        for example in examples
    ]
    attention_mask: List[List[int]] = [
        [attn_extension] * max(0, max_len - len(example["attention_mask"]))
        + example["attention_mask"][:max_len]
        for example in examples
    ]
    token_type_ids: List[List[int]] = [
        [type_extension] * max(0, max_len - len(example["token_type_ids"]))
        + example["token_type_ids"][:max_len]
        for example in examples
    ]
    if return_tensors:
        input_ids = torch.tensor(input_ids, dtype=torch.long)  # type: ignore
        attention_mask = torch.tensor(attention_mask, dtype=torch.bool)  # type: ignore
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)  # type: ignore

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }


def pad_dynamic(
    examples,
    pad_token_id,
    attn_extension: int = 0,
    type_extension: int = 2,
) -> BaseBatch:
    max_len = max(len(e["input_ids"]) for e in examples)
    return pad_truncate(
        examples,
        max_len,
        pad_token_id,
        attn_extension=attn_extension,
        type_extension=type_extension,
    )


class DefaultCollatorWithPadding(DefaultCollator):
    """Like DefaultCollator, but pads (truncates if needed) the input_ids, attention_mask, and token_type_ids to self.max_length."""

    def get_max_len(self, examples: List[BaseCollatorInput]) -> int:
        return self.block_size

    def __call__(self, examples: List[BaseCollatorInput]) -> BaseBatch:
        max_len = self.get_max_len(examples)
        return pad_truncate(
            examples,
            max_len,
            self.tokenizer.pad_token_id,
            attn_extension=0,
            type_extension=2,
        )


class DefaultCollatorWithDynamicPadding(DefaultCollatorWithPadding):
    """Like DefaultCollator, but pads to the max length in the batch."""

    def get_max_len(self, examples: List[BaseCollatorInput]) -> int:
        max_in_batch = max(len(e["input_ids"]) for e in examples)
        return min(max_in_batch, self.block_size)


# endregion: Collators
################################################################################


################################################################################
# region: Utilities


def print_batch_base(
    batch: Dict[str, Any],
    split: Literal["train", "val", "test", "predict"],
    tokenizer: Tokenizer,
    dataloader_name: str = "",
):

    logger.info(
        f"Printing first entries of the tensors in batch for {split}/{dataloader_name}..."
    )
    print("input tokens:")
    print(tokenizer.decode(batch["input_ids"][0]))
    print("input_ids:")
    print(batch["input_ids"][0])
    print("attention_mask:")
    print(batch["attention_mask"][0])
    print("token_type_ids:")
    print(batch["token_type_ids"][0])


# endregion: Utilities
################################################################################
