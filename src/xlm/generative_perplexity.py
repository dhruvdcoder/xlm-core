import json
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import (
    ContextManager,
    Counter,
    Iterator,
    List,
    Optional,
    Protocol,
    Tuple,
    TypedDict,
    Union,
    cast,
)

import lightning as L
import torch
from jaxtyping import Float, Integer
from torch import Tensor as TT
from torchmetrics import Perplexity
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from xlm.datamodule import Tokenizer
from xlm.utils.rank_zero import rank_zero_info, rank_zero_warn, warn_once

logger = logging.getLogger(__name__)


############################################################
# region: Evaluators


class GenerativePerplexityEvaluatorResult(TypedDict):
    logits: Float[TT, " *batch seq_len"]
    target: Integer[
        TT, " *batch seq_len"
    ]  # contains tokens in the input tokenized according to the evaluator with ignore_index set for pad


class GenerativePerplexityEvaluator(Protocol):
    def __call__(
        self, samples: List[str]
    ) -> Optional[GenerativePerplexityEvaluatorResult]: ...

    @property
    def batch_size(self) -> int: ...

    @property
    def ignore_index(self) -> int: ...

    @property
    def name(self) -> str: ...

    def __repr__(self) -> str: ...

    def load(
        self,
        generator_tokenizer: Union[
            PreTrainedTokenizer, PreTrainedTokenizerFast
        ],
        device: Union[str, torch.device],
    ) -> None: ...

    def unload(self) -> None: ...

    def loaded(
        self,
        generator_tokenizer: Union[
            PreTrainedTokenizer, PreTrainedTokenizerFast
        ],
        device: Union[str, torch.device],
    ) -> ContextManager["GenerativePerplexityEvaluator"]: ...

    # the context manager helps us unload the model when done.


class AutoModelForCausalLMGenerativePerplexityEvaluator(
    GenerativePerplexityEvaluator
):
    supported_models = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]

    def __init__(
        self, name: str, batch_size: int = 64, device: Optional[str] = None
    ):
        """
        Args:
            name: Name of the pretrained model. Currently supported models: gpt2-*
            batch_size: Batch size for the evaluator. The default should work on A100. (default: 64)
        """
        if name not in self.supported_models:
            # raise ValueError(
            #    f"Unsupported model: {name}. Supported models: {self.supported_models}."
            # )
            logger.warning(
                f"Unsupported model: {name}. Supported models: {self.supported_models}."
            )
        self._name = name
        self._batch_size = batch_size
        self._ignore_index = -100
        self.pretrained_model = None
        self.tokenizer = None
        self.device = device

    def __repr__(self) -> str:
        return f"AutoModelForCausalLMGenerativePerplexityEvaluator(name={self.name}, batch_size={self.batch_size})"

    @property
    def name(self) -> str:
        # convert any / to __
        # return self._name.replace("/", "__")
        return self._name

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def ignore_index(self) -> int:
        return self._ignore_index

    def load(
        self,
        generator_tokenizer: Optional[
            Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
        ],
        device: Union[str, torch.device],
    ) -> None:
        """
        Load the pretrained model and tokenizer.

        1. EOS
            We need to make sure that if the generator generates EOS token. Then the evaluator
            can evaluate it correctly. That is, the eval tokenizer recognizes it as single EOS token and the eval
            model is trained to predict it.
        2. PAD
            We need PAD token in eval tokenizer because retokenization will produce sequences of varying lengths.
            Some models like GPT2 only have EOS token. In those cases, we will not be able to evaluate EOS generation
            because PAD will be set to EOS and the EOS token will be lost.
        """
        
        self.pretrained_model = (
            cast(
                torch.nn.Module,
                AutoModelForCausalLM.from_pretrained(self.name),
            )
            .to(device)
            .eval()
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.name
        )
        self.device = device
        original_vocab_size = self.tokenizer.vocab_size
        # eos token
        if generator_tokenizer is not None:
            self.tokenizer.eos_token = generator_tokenizer.eos_token
        # pad token:
        # GPT2 does not have any other special tokens, but we need padding. We set pad to be eos.
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        if original_vocab_size != self.tokenizer.vocab_size:
            rank_zero_warn(
                f"Vocab size changed from {original_vocab_size} to {self.tokenizer.vocab_size} for {self} tokenizer."
                "This may cause embedding lookup error in the eval model."
            )
        rank_zero_info(
            f"Special tokens of {self.name} tokenizer:\n{self.tokenizer.special_tokens_map}"
        )

    def __call__(
        self, samples: List[str]
    ) -> Optional[GenerativePerplexityEvaluatorResult]:
        if len(samples) != 1:
            raise ValueError("batching is not supported yet.")

        encoding = self.tokenizer(  # pyright: ignore[reportOptionalCall]
            samples,
            # padding="longest",
            padding=False,
            truncation=False,
            return_attention_mask=True,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(
            self.device  # pyright: ignore[reportArgumentType]
        )
        inputs: Integer[TT, " *batch seq_len-1"] = encoding.input_ids[:, :-1]
        attention_mask: Integer[TT, " *batch seq_len-1"] = (
            encoding.attention_mask[:, :-1]
        )
        lengths = attention_mask.sum(dim=-1)  # (batch)
        target: Integer[TT, " *batch seq_len-1"] = encoding.input_ids[
            :, 1:
        ].clone()
        # set pad targets to ignore_index. Use attention mask to find pads.
        target = torch.where(
            encoding.attention_mask[:, 1:] == 0,
            self.ignore_index,
            target,
        )
        if inputs.numel() == 0:
            warn_once(
                logger,
                "Empty input received for generative perplexity evaluation. Skipped.",
            )
            return None
        with torch.autocast(
            device_type=self.device.type, dtype=torch.bfloat16
        ):
            outputs = (
                self.pretrained_model(  # pyright: ignore[reportOptionalCall]
                    inputs, attention_mask=attention_mask
                )
            )
        logits = outputs.logits  # (batch, seq_len, vocab_size)
        nlls = torch.nn.functional.cross_entropy(
            logits.transpose(-1, -2),
            target,
            reduction="none",
            ignore_index=self.ignore_index,
        )
        counts = torch.tensor(
            list(Counter(target[0].tolist()).values())
        )  # count the tokens
        total = counts.sum()
        if total != lengths.item():
            raise RuntimeError
        p_for_entropy = counts / total
        entropy = -torch.sum(p_for_entropy * torch.log(p_for_entropy))
        return {
            "logits": logits,
            "target": target,
            "lengths": lengths,
            "nlls": nlls,
            "entropy": entropy,
        }

    def unload(self) -> None:
        self.pretrained_model = None
        self.tokenizer = None
        # clear the cache
        torch.cuda.empty_cache()

    @contextmanager
    def loaded(
        self,
        generator_tokenizer: Union[
            PreTrainedTokenizer, PreTrainedTokenizerFast
        ],
        device: Union[str, torch.device],
    ):
        try:
            self.load(generator_tokenizer, device)
            yield self
        finally:
            self.unload()


class GPT2GenerativePerplexityEvaluator(
    AutoModelForCausalLMGenerativePerplexityEvaluator
):
    def load(
        self,
        generator_tokenizer: Union[
            PreTrainedTokenizer, PreTrainedTokenizerFast
        ],
        device: Union[str, torch.device],
    ) -> None:
        super().load(generator_tokenizer, device)
        # make sure we have pad token
        # If not, make another existing token the pad token.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = "<|pad|>"


# endregion: Evaluators
############################################################


def _compute_generative_perplexity(
    evaluator: GenerativePerplexityEvaluator,
    device: torch.device,
    tokenizer: Tokenizer,
    file_path: Path,
) -> Optional[TT]:
    """
    Depricated version because it works with file path.
    """
    if not file_path.exists():
        logger.error(
            f"No samples found at {file_path}. "
            "If you are using a callback like UnconditionalSampleGenerator, to generate samples, "
            "make sure that it is placed before GenerativePerplexity in the callbacks list."
        )
        return None

    # Load samples from file and evaluate in batches
    samples = []
    generative_perplexity_metric = Perplexity(
        ignore_index=evaluator.ignore_index
    ).to(device)
    generative_perplexity_metric.reset()
    with evaluator.loaded(
        tokenizer,  # type: ignore
        device,
    ):  # load it on the same device for now
        with open(file_path) as f:
            for line in f:
                sample = json.loads(line)
                text = sample["text"]
                samples.append(text)

                if len(samples) == evaluator.batch_size:
                    result: Optional[GenerativePerplexityEvaluatorResult] = (
                        evaluator(samples)
                    )
                    # reset the samples list for the next batch
                    samples = []
                    if result is None:
                        continue
                    logits = result["logits"]
                    target = result["target"]
                    generative_perplexity_metric.update(logits, target)

            # Handle remaining samples in last batch if any
            if samples:
                result = evaluator(samples)
                if result is not None:
                    logits = result["logits"]
                    target = result["target"]
                    generative_perplexity_metric.update(logits, target)

    perplexity = generative_perplexity_metric.compute()
    return perplexity


def _compute_nll_for_sample(
    string: str,
    evaluator: GenerativePerplexityEvaluator,
    device: torch.device,
    tokenizer: Tokenizer,
) -> float:
    # support batching maybe later
    inputs = tokenizer([string], return_tensors="pt")
    length = inputs.input_ids.shape[1] - 1
    inputs.to(device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        evaluator = cast(
            AutoModelForCausalLMGenerativePerplexityEvaluator, evaluator
        )
        outputs = evaluator.pretrained_model(
            **inputs
        )  # only works with AutoModelForCausalLM
    all_log_probs = torch.log_softmax(outputs.logits, dim=-1)[0, :-1, :]
    log_p = all_log_probs[
        torch.arange(all_log_probs.shape[0], device=all_log_probs.device),
        inputs.input_ids[0, 1:],
    ]
    nll = -log_p.mean().item()
    return nll, length


def compute_nll_for_sample(
    string: str,
    evaluator: GenerativePerplexityEvaluator,
) -> Optional[Tuple[float, int, float]]:
    # support batching maybe later
    eval_result = evaluator([string])
    if eval_result is None:
        return None
    total_nll_per_sample: float = eval_result["nlls"].sum(-1).item()
    total_length: int = eval_result["lengths"].sum().item()
    entropy: float = eval_result["entropy"].item()
    return total_nll_per_sample, total_length, entropy


def compute_entropy_and_length_for_sample(
    string: str,
    tokenizer: Tokenizer,
) -> Tuple[float, int]:
    inputs = tokenizer([string], return_tensors="pt")
    length = inputs.input_ids.shape[1] - 1
    counts = torch.tensor(
        list(Counter(inputs.input_ids[0, 1:].tolist()).values())
    )
    total = counts.sum()
    p_for_entropy = counts / total
    entropy = -torch.sum(p_for_entropy * torch.log(p_for_entropy)).item()
    return entropy, length
