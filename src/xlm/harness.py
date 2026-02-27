from collections.abc import Generator
from itertools import chain
import json
from pathlib import Path
import re
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypeVar,
    TypedDict,
    Union,
)

import hydra

from xlm.metrics import MetricWrapper
from lightning.pytorch.utilities import grad_norm

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from transformers import get_scheduler
from xlm.datamodule import BaseDataModule, Tokenizer
from xlm.noise import NoiseSchedule
from xlm.utils.rank_zero import RankedLogger
import lightning as L
from huggingface_hub import PyTorchModelHubMixin

# Import LogPredictions classes from the new module
from xlm.log_predictions import (
    LogPredictions,
)
from xlm.generative_perplexity import (
    GenerativePerplexityEvaluator,
    compute_entropy_and_length_for_sample,
    compute_nll_for_sample,
)

logger = RankedLogger(__name__, rank_zero_only=True)
ranked_logger = RankedLogger(__name__, rank_zero_only=False)


class LRSchedulerWithConfig(TypedDict):
    """We follow the same structure as the one in LightningModule.
    lr_scheduler_config = {
                # REQUIRED: The lr_scheduler instance
                "scheduler": lr_scheduler,
                # The unit of the lr_scheduler's step size, could also be 'step'.
                # 'epoch' updates the lr_scheduler on epoch end whereas 'step'
                # updates it after a optimizer update.
                "interval": "epoch",
                # How many epochs/steps should pass between calls to
                # `lr_scheduler.step()`. 1 corresponds to updating the learning
                # rate after every epoch/step.
                "frequency": 1,
                # Metric to to monitor for schedulers like `ReduceLROnPlateau`
                "monitor": "val_loss",
                # If set to `True`, will enforce that the value specified 'monitor'
                # is available when the lr_scheduler is updated, thus stopping
                # training if not found. If set to `False`, it will only produce a warning
                "strict": True,
            }
    """

    scheduler: LRScheduler
    interval: str  # = "step"
    frequency: int  # = 1
    monitor: Optional[str]  # = None
    strict: bool  # = True


ModelOutput = Dict[Union[Literal["loss"], str], Any]

T_in = TypeVar("T_in", contravariant=True)
T_out = TypeVar("T_out", covariant=True)


class LossFunction(Generic[T_in, T_out], Protocol):
    model: Any
    tokenizer: Tokenizer

    def configure(self, pl_module: "Harness"): ...

    """Called from the device context."""

    def loss_fn(
        self,
        batch: T_in,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> T_out: ...

    """Break the logic in __call__ into smaller functions that are compilable and return them here.
    """

    def __call__(
        self,
        batch: T_in,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> T_out: ...


T_out_pred = TypeVar("T_out_pred")


class Predictor(Generic[T_in, T_out_pred], Protocol):
    tokenizer: Tokenizer
    model: Any

    def predict(
        self,
        batch: T_in,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> T_out_pred: ...

    def to_dict(
        self,
        batch: T_in,
        preds: T_out_pred,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Create json lines from the predictions batch."""
        ...

    def generate(self, prompts: List[str]) -> List[str]: ...


class PredictorHistoryMixin:
    """Mixin class for adding history tracking to predictors.

    This mixin provides generic history tracking capabilities that can be
    used by any predictor that implements iterative generation. History is
    stored as a list of tuples: (decoded_text, confidence_score, step_number).

    Usage:
        class MyPredictor(torch.nn.Module, PredictorHistoryMixin, Predictor[...]):
            def __init__(self, ..., return_history: bool = False):
                super().__init__()
                self.init_history(return_history=return_history)
                ...

            def predict(self, batch):
                history = self.create_history(batch_size)
                for step in range(steps):
                    # ... generation logic ...
                    history = self.update_history_from_state(history, state, step)
                return {"text": ..., "history": history}
    """

    return_history: bool

    def init_history(
        self,
        return_history: bool = False,
        decode_fn: Optional[Callable] = None,
    ):
        """Initialize history tracking.

        Args:
            return_history: Whether to track history during generation.
            decode_fn: Optional custom decode function. If None, will use self.decode.
        """
        self.return_history = return_history
        self._history_decode_fn = decode_fn

    def _get_decode_function(self):
        """Get the appropriate decode function.

        Returns:
            Decode function to use for converting state to text.

        Raises:
            NotImplementedError: If no decode function is available.
        """
        if (
            hasattr(self, "_history_decode_fn")
            and self._history_decode_fn is not None
        ):
            return self._history_decode_fn
        elif hasattr(self, "decode"):
            return self.decode
        else:
            raise NotImplementedError(
                "Must provide decode_fn to init_history() or implement decode() method"
            )

    def create_history(
        self, batch_size: int
    ) -> List[List[Tuple[str, float, int]]]:
        """Create empty history for a batch.

        Args:
            batch_size: Number of sequences in the batch.

        Returns:
            Empty history list for each batch element.
        """
        return [[] for _ in range(batch_size)]

    def update_history_from_state(
        self,
        history: List[List[Tuple[str, float, int]]],
        state: Dict[str, Any],
        step: int,
        confidence_key: str = "confidence",
        active_mask_key: Optional[str] = None,
    ) -> List[List[Tuple[str, float, int]]]:
        """Update history from a state dictionary.

        Args:
            history: Current history list.
            state: Dictionary containing the current state (must be decodable).
            step: Current step number.
            confidence_key: Key in state dict for confidence values (default: "confidence").
            active_mask_key: Optional key for mask indicating which samples are still active.

        Returns:
            Updated history.
        """
        if not self.return_history:
            return history

        # Get decode function
        decode_fn = self._get_decode_function()

        # Decode current state
        decoded = decode_fn(state)

        # Handle different decode return formats
        if isinstance(decoded, tuple):
            decoded_texts = decoded[0]  # Assume first element is the text list
        else:
            decoded_texts = decoded

        # Get confidence values
        if confidence_key in state:
            confidences = state[confidence_key]
            if hasattr(confidences, "tolist"):
                confidences = confidences.tolist()
        else:
            confidences = [1.0] * len(decoded_texts)

        # Determine which samples are active
        if active_mask_key is not None and active_mask_key in state:
            active_mask = state[active_mask_key]
            if hasattr(active_mask, "tolist"):
                active_mask = active_mask.tolist()
        else:
            active_mask = [True] * len(decoded_texts)

        # Update history only for active samples
        for batch_idx, (text, conf, active) in enumerate(
            zip(decoded_texts, confidences, active_mask)
        ):
            if active:
                if isinstance(conf, (list, tuple)):
                    conf = float(conf[0]) if len(conf) > 0 else 1.0
                history[batch_idx].append((text, float(conf), step))

        return history

    def update_history_explicit(
        self,
        history: List[List[Tuple[str, float, int]]],
        texts: List[str],
        confidences: Union[List[float], torch.Tensor],
        step: int,
        active_mask: Optional[Union[List[bool], torch.Tensor]] = None,
    ) -> List[List[Tuple[str, float, int]]]:
        """Update history with explicit values.

        Args:
            history: Current history list.
            texts: Decoded text for each batch element.
            confidences: Confidence/score for each batch element.
            step: Current step number.
            active_mask: Optional mask indicating which samples are still active.

        Returns:
            Updated history.
        """
        if not self.return_history:
            return history

        # Convert tensors to lists if needed
        if hasattr(confidences, "tolist"):
            confidences = confidences.tolist()

        if active_mask is None:
            active_mask = [True] * len(texts)
        elif hasattr(active_mask, "tolist"):
            active_mask = active_mask.tolist()

        for batch_idx, (text, conf, active) in enumerate(
            zip(texts, confidences, active_mask)
        ):
            if active:
                history[batch_idx].append((text, float(conf), step))

        return history

    def format_history_for_output(
        self,
        history: List[List[Tuple[str, float, int]]],
        round_precision: int = 4,
    ) -> List[List[List[Any]]]:
        """Format history for output in to_dict methods.

        Args:
            history: Raw history list.
            round_precision: Number of decimal places to round confidence values.

        Returns:
            Formatted history with rounded confidence values.
        """
        return [
            [
                [text, round(conf, round_precision), step]
                for text, conf, step in seq_history
            ]
            for seq_history in history
        ]


class CustomModuleDict(torch.nn.ModuleDict):
    def get(self, key: str, default: Any = None) -> Any:
        if key in self:
            return self[key]
        else:
            return default


def to_nested_module_dict(
    metrics: Dict[
        Literal["train", "val", "test", "predict"],
        Dict[str, Dict[str, MetricWrapper]],  # k1, k2 = dl_name, metric_name
    ],
) -> CustomModuleDict:
    return CustomModuleDict(
        {
            f"metrics_{stage}": CustomModuleDict(
                {
                    dl_name: torch.nn.ModuleList(list(metrics.values()))
                    for dl_name, metrics in dl_name_to_metrics.items()
                }
            )
            for stage, dl_name_to_metrics in metrics.items()
        }
    )


class Harness(L.LightningModule, PyTorchModelHubMixin):
    """Main module that provides the scaffolding for the codebase."""

    model: nn.Module
    config: DictConfig
    predictor: Predictor
    loss_function: LossFunction
    noise_schedule: NoiseSchedule
    predictions_dir: Path
    tokenizer: Tokenizer
    """Task
    Metrics usually consist of two types of metrics:
        1. diagnostic metrics: These are typically different for different models as well as different tasks.
        2. reported metrics: These are the same for all the models but different for different tasks.
    What we want too do is avoid a full blown (task x model) setup whenever we can but provide it as a last resort.
    The best case scenario is complete decopling. This happens when all the models adhere to the same output signature.
    But this never works for diagnostic metrics.
    In some cases, different tasks can share base metrics of both types. In these cases, we can use inheritance to avoid some code duplication. We would still have (task x model) number of classes though.
    """
    dataloader_names: Dict[
        Literal["train", "val", "test", "predict"], Dict[int, str]
    ]
    diagnostic_metrics: torch.nn.ModuleDict
    # ModuleDict[
    #    Literal["metrics_train", "metrics_val", "metrics_test", "metrics_predict"],
    #   ModuleDict[str, ModuleList[MetricWrapper]],
    # ]
    reported_metrics: torch.nn.ModuleDict
    # ModuleDict[
    #    Literal["metrics_train", "metrics_val", "metrics_test", "metrics_predict"],
    #    torch.nn.ModuleDict[str, torch.nn.ModuleList[MetricWrapper]],
    # ]
    key_to_remove_after_logging: List[str]
    log_predictions: Optional[LogPredictions]
    generative_perplexity_evaluators: Dict[str, GenerativePerplexityEvaluator]

    def __init__(
        self,
        cfg: DictConfig,
        tokenizer: Optional[Tokenizer] = None,
        datamodule: Optional[BaseDataModule] = None,
        write_per_sample_metrics: bool = False,
        log: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ):
        """Initialize the Harness module.

        Args:
            cfg: Configuration dictionary.
            tokenizer: Optional tokenizer instance.
            datamodule: Optional datamodule instance.
            write_per_sample_metrics: Whether to write per-sample metrics.
            **kwargs: Additional keyword arguments.
        """
        # Ensure config is a DictConfig. When loading from checkpoint, we get a dict instance.
        if not isinstance(cfg, DictConfig):
            cfg = OmegaConf.create(cfg)
        # If an override config is provided, merge it.
        if "cfg" in kwargs:
            override_cfg = kwargs.pop("cfg")
            if not isinstance(override_cfg, DictConfig):
                override_cfg = OmegaConf.create(override_cfg)
            _cfg = OmegaConf.merge(cfg, override_cfg)
            assert isinstance(_cfg, DictConfig)
            cfg = _cfg
        super().__init__()
        self.config = cfg
        self.datamodule = datamodule  # hold a reference to the datamodule
        self.update_omegaconf_resolvers()
        self.setup_vocab(tokenizer)
        self.instantiate_noise_schedule()
        self.instantiate_model()
        self.instantiate_predictor()
        self.instantiate_loss_function()
        self.setup_dataloader_names()
        self.setup_metrics(cfg)
        self.setup_generative_perplexity(cfg)
        self.setup_post_hoc_evaluator(cfg)
        self.predictions_dir = Path(cfg.paths.run_dir) / "predictions"
        # validate and save config at the end
        self.validate_config(self.config)
        # Save hyperparameters as a plain dict.
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        # Dictionary to store whether we have printed the batch
        self.printed_batches: Set[str] = set()
        self.last_global_step_logged_at_which_logged_predictions: int = -1
        if cfg.lightning_module.get("write_per_sample_metrics", False):
            self.write_per_sample_metrics = True
        else:
            self.write_per_sample_metrics = write_per_sample_metrics
        self.key_to_remove_after_logging = []
        # We need to manually restore the EMA weights when loading from checkpoint for evaluation
        # because we will not have ema callback then.
        if kwargs.get("manual_ema_restore", False):
            self.manual_ema_restore = True
        else:
            self.manual_ema_restore = False
        self.outer_autocast = False
        if log is not None:
            if "train/loss" not in log.keys():
                log["train/loss"] = "loss"

            self.log_strs = log
        else:
            self.log_strs = {"train/loss": "loss"}

    ############################################################
    # region: Setup methods

    def setup_vocab(self, tokenizer: Optional[Tokenizer]):
        if tokenizer is None:
            self.tokenizer = hydra.utils.instantiate(
                self.config.datamodule.tokenizer
            )
        else:
            self.tokenizer = tokenizer

    def update_omegaconf_resolvers(self):
        OmegaConf.clear_resolver("tokenizer")
        OmegaConf.register_new_resolver(
            "tokenizer", lambda x: getattr(self.tokenizer, x)
        )
        OmegaConf.clear_resolver("datamodule")
        OmegaConf.register_new_resolver(
            "datamodule", lambda x: getattr(self.datamodule, x)
        )
        OmegaConf.clear_resolver("lightning_module")
        OmegaConf.register_new_resolver(
            "lightning_module", lambda x: getattr(self, x)
        )

    def instantiate_noise_schedule(self):
        # first try to get it from the datamodule
        if self.datamodule is not None:
            if hasattr(self.datamodule, "noise_schedule"):
                self.noise_schedule = self.datamodule.noise_schedule  # type: ignore
                return
        # otherwise, instantiate it from the config
        self.noise_schedule = hydra.utils.instantiate(
            self.config.noise_schedule
        )

    def instantiate_model(self):
        self.model = hydra.utils.instantiate(self.config.model)

    def instantiate_predictor(self):
        self.predictor = hydra.utils.instantiate(self.config.predictor)
        if self.predictor.tokenizer is None:
            self.predictor.tokenizer = self.tokenizer
        if self.predictor.noise_schedule is None:
            self.predictor.noise_schedule = self.noise_schedule
        if self.predictor.model is None:
            self.predictor.model = self.model

    def instantiate_loss_function(self):
        self.loss_function = hydra.utils.instantiate(self.config.loss)
        if self.loss_function.tokenizer is None:
            self.loss_function.tokenizer = self.tokenizer
        if self.loss_function.model is None:
            self.loss_function.model = self.model
        # check for consistency with the predictor
        if hasattr(self, "check_loss_predictor_consistency"):
            self.check_loss_predictor_consistency()

    def setup_dataloader_names(self):
        if self.datamodule is None:
            raise ValueError("Datamodule is required")
        self.dataloader_names = self.datamodule.dataloader_names

    def setup_metrics(
        self,
        cfg: DictConfig,
    ):
        """Attache metrics as modules"""
        diagnostic_metrics: Dict[
            Literal["train", "val", "test", "predict"],
            Dict[
                str, Dict[str, MetricWrapper]
            ],  # k1, k2 = dl_name, metric_name
        ] = (
            hydra.utils.instantiate(cfg.diagnostic_metrics) or {}
        )
        reported_metrics: Dict[
            Literal["train", "val", "test", "predict"],
            Dict[
                str, Dict[str, MetricWrapper]
            ],  # k1, k2 = dl_name, metric_name
        ] = (
            hydra.utils.instantiate(cfg.reported_metrics) or {}
        )
        self.diagnostic_metrics = to_nested_module_dict(diagnostic_metrics)
        self.reported_metrics = to_nested_module_dict(reported_metrics)
        if "log_predictions" in cfg:
            self.log_predictions = hydra.utils.instantiate(cfg.log_predictions)
        else:
            self.log_predictions = None

    def setup_generative_perplexity(self, cfg: DictConfig):
        self.generative_perplexity_evaluators: Dict[
            str, GenerativePerplexityEvaluator
        ] = {}
        if (
            cfg.get("generative_perplexity", {}).get("evaluators", None)
            is not None
        ):
            for (
                evaluator_name,
                evaluator_conf,
            ) in cfg.generative_perplexity.evaluators.items():
                evaluator = hydra.utils.instantiate(evaluator_conf)
                self.generative_perplexity_evaluators[evaluator_name] = (
                    evaluator
                )
                logger.info(
                    f"Instantiated generative perplexity evaluator: {evaluator_name}"
                )

    def setup_post_hoc_evaluator(self, cfg: DictConfig):
        """Setup post-hoc evaluator. Can be use for tasks like molecule generation.

        The post-hoc evaluator computes metrics on logged predictions at epoch end,
        enabling global metric computation (e.g., diversity on full generated set).

        Args:
            cfg: Configuration dictionary
        """
        self.post_hoc_evaluator: Optional[Any] = None

        if cfg.get("post_hoc_evaluator", None) is not None:
            self.post_hoc_evaluator = hydra.utils.instantiate(
                cfg.post_hoc_evaluator
            )
            logger.info(
                f"Instantiated post-hoc evaluator: {type(self.post_hoc_evaluator).__name__}"
            )

    def validate_config(self, cfg: DictConfig):
        return

    def setup(self, stage: Literal["fit", "validate", "test", "predict"]):
        pass

    def configure_model(self):
        if self.loss_function is not None:
            if hasattr(self.loss_function, "configure"):
                self.loss_function.configure(self)
        if self.config.get("compile", False):
            # Issue: 1: We need to execute the prediction loop in eager mode
            # We need to execute the prediction loop in eager mode, but torch.compile.set_stance
            # is only available in torch >= 2.6. In torch 2.5, there is no way to tell torch to
            # execute a compiled model in eager mode. So we will wait for lightning to support torch 2.6.
            # logger.info("Wrapping model in torch.compile")
            # self.model.compile(
            #    fullgraph=True,
            #    options={
            #        "trace.graph_diagram": False,
            #        "trace.enabled": False,
            #    },
            # )  # type: ignore
            # Solution: 1: In the meantime, we will compile the loss function callable

            # Issue 2: Compile should ideally be called after wrapping the model in DDP
            # But lightning trainer does not support this yet: https://github.com/Lightning-AI/pytorch-lightning/pull/20269
            # Solution 2: For now we will disable DDP optimizer. This will reduce the efficiency but can't do anything.
            torch._dynamo.config.optimize_ddp = False
            torch._dynamo.config.cache_size_limit = 100  # type: ignore
            logger.info("Compiling loss function")
            self.loss_function.loss_fn = torch.compile(
                self.loss_function.loss_fn,
                backend="inductor",
                fullgraph=False,  # Turn to True if there is no nested autocast. This is not bullet proof solutions to NaN due to fusing of bfloat16 and nested float32 islands, but just simple solution that may work.
                options={
                    "trace.graph_diagram": False,
                    "trace.enabled": False,
                },
                # dynamic=True,
            )  # type: ignore

    @staticmethod
    def create_lr_scheduler(
        optimizer: Optimizer,
        name: str,
        num_warmup_steps: Optional[int] = None,
        fraction_warmup_steps: Optional[float] = None,
        num_training_steps: Optional[int] = None,
        interval: Literal["step", "epoch"] = "step",
        frequency: int = 1,
        monitor: Optional[str] = "train_loss",
        strict: bool = True,
        **kwargs: Any,
    ) -> LRSchedulerWithConfig:
        """Creates a learning rate noise_schedule with the given configuration.

        Args:
            name: Huggingface name of the learning rate noise_schedule. https://huggingface.co/docs/transformers/en/main_classes/optimizer_schedules#transformers.get_scheduler
            optimizer: The optimizer to use with the noise_schedule
            num_training_steps: The total number of training steps.
            num_warmup_steps: The number of warmup steps.
            fraction_warmup_steps: The fraction of training steps to use for warmup.
            interval: The interval at which to update the learning rate.
            frequency: The frequency of the learning rate updates.
            monitor: The metric to monitor for the learning rate noise_schedule.
            strict: Whether to strictly follow the learning rate schedule.
            **kwargs: Additional keyword arguments to pass to the learning rate noise_schedule.

        Returns:
            LRSchedulerWithConfig: The configured learning rate scheduler.
        """
        if num_warmup_steps is None:
            if fraction_warmup_steps is None:
                raise ValueError(
                    "Either num_warmup_steps or fraction_warmup_steps must be provided"
                )
            if num_training_steps is None:
                raise ValueError(
                    "num_training_steps must be provided when using fraction_warmup_steps"
                )
            num_warmup_steps = int(num_training_steps * fraction_warmup_steps)
        logger.info(f"num_warmup_steps: {num_warmup_steps}")

        lr_scheduler = get_scheduler(
            name,
            optimizer,
            num_warmup_steps,
            num_training_steps,
            scheduler_specific_kwargs=(kwargs or None),
        )
        return LRSchedulerWithConfig(
            scheduler=lr_scheduler,
            interval=interval,
            frequency=frequency,
            monitor=monitor,
            strict=strict,
        )

    def configure_optimizers(
        self,
    ) -> Dict[
        Literal["optimizer", "lr_scheduler"],
        Union[Optimizer, LRSchedulerWithConfig],
    ]:
        partial_optimizer = hydra.utils.instantiate(
            self.config.optimizer, _partial_=True
        )
        if hasattr(self.model, "get_param_groups"):
            groups = self.model.get_param_groups()
        else:
            main_params_with_weight_decay = list(
                p for _, p in self.model.get_named_params_for_weight_decay()
            )
            main_params_without_weight_decay = list(
                p for _, p in self.model.get_named_params_for_no_weight_decay()
            )
            logger.info(
                f"Num params with weight decay in the `model`: {len(main_params_with_weight_decay)}"
            )
            logger.info(
                f"Num params without weight decay in the `model`: {len(main_params_without_weight_decay)}"
            )
            groups = [
                {"params": main_params_with_weight_decay},
                {
                    "params": main_params_without_weight_decay,
                    "weight_decay": 0.0,
                },
            ]
        optimizer = partial_optimizer(groups)
        lr_scheduler: LRSchedulerWithConfig = self.create_lr_scheduler(
            optimizer, **self.config.lr_scheduler
        )
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    # endregion: Setup methods
    ############################################################

    ############################################################
    # region: Task-specific methods

    def prepare_batch_for_prediction(
        self, batch: Dict[str, Any]
    ) -> Dict[str, Any]:
        """We need this for some tasks even if we have task sepecific collator,
        mainly because we want to clone some elements of the batch useful for computing metrics.
        TODO: Get rid of this method by cloning in the collator itself.
        """
        return batch

    def compute_loss(
        self,
        batch: Dict[str, Any],
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Computes loss based on the dataloader name.

        For 'lm', the loss function is applied.
        For 'prediction', the predictor's predict_step is used.
        """
        dataloader_name = dataloader_name or "lm"
        if "lm" in dataloader_name:
            return self.loss_function(
                batch, batch_idx, dataloader_idx, dataloader_name
            )
        elif "prediction" in dataloader_name:
            cloned_batch = self.prepare_batch_for_prediction(batch)
            preds = self.predictor.predict(
                cloned_batch, batch_idx, dataloader_idx, dataloader_name
            )
            return preds
        else:
            return {}

    def _step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
        dataloader_idx: int,
        stage: Literal["train", "val", "test", "predict"],
    ) -> Dict[str, Any]:
        dataloader_name = self.dataloader_names[stage][dataloader_idx]
        loss_dict = self.compute_loss(
            batch, batch_idx, dataloader_idx, dataloader_name
        )
        if loss_dict.get("loss", False):
            if bool(loss_dict["loss"].isnan()):
                global_step = self.trainer.global_step
                logger.error(
                    f"NaN loss encountered in training step {global_step} in epoch {self.trainer.current_epoch}. Following is the loss dictionary:\n\n {loss_dict}"
                )
                for k, v in loss_dict.items():
                    if isinstance(v, torch.Tensor) and bool(v.isnan().any()):
                        location = v.isnan().nonzero()
                        logger.error(
                            f"Tensor {k} has nan values at location {location}"
                        )
                    if isinstance(v, torch.Tensor) and bool(v.isinf().any()):
                        location = v.isinf().nonzero()
                        logger.error(
                            f"Tensor {k} has inf values at location {location}"
                        )
                if not self.trainer._detect_anomaly:
                    raise RuntimeError(
                        f"NaN loss encountered in training step {global_step} in epoch {self.trainer.current_epoch}"
                    )
            if bool(loss_dict["loss"].isinf()):
                global_step = self.trainer.global_step
                raise RuntimeError(
                    f"Inf loss ({loss_dict['loss']}) encountered in training step {global_step} in epoch {self.trainer.current_epoch}"
                )
        if stage == "train":
            for k, v in self.log_strs.items():
                self.log(
                    k,
                    loss_dict[v].detach(),
                    on_step=True,
                    on_epoch=False,
                    prog_bar=True,
                    sync_dist=False,
                    rank_zero_only=True,
                    logger=True,
                    add_dataloader_idx=False,
                )
        for metric in chain(
            self.diagnostic_metrics.get(f"metrics_{stage}", {}).get(
                dataloader_name, []
            ),
            self.reported_metrics.get(f"metrics_{stage}", {}).get(
                dataloader_name, []
            ),
        ):
            metric.update(batch, loss_dict, self.tokenizer)
            # add computed value to loss_dict if per sample value is available
            if getattr(metric, "_computed_value", None) is not None:
                if isinstance(metric._computed_value, torch.Tensor):
                    loss_dict[f"metric_{metric.name}"] = (
                        metric._computed_value.detach().tolist()
                    )
                    metric._computed_value = None
                elif isinstance(metric._computed_value, list):
                    loss_dict[f"metric_{metric.name}"] = metric._computed_value
                    metric._computed_value = None
            metric.log(self, batch, loss_dict)

        for key in self.key_to_remove_after_logging:
            loss_dict.pop(key, None)
        return loss_dict

    def training_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
        dataloader_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        return self._step(batch, batch_idx, dataloader_idx or 0, "train")

    @torch._dynamo.disable()
    def validation_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Dict[str, Any]:
        res = self._step(batch, batch_idx, dataloader_idx, "val")
        return res

    @torch._dynamo.disable()
    def test_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Dict[str, Any]:
        return self._step(batch, batch_idx, dataloader_idx, "test")

    @torch._dynamo.disable()
    def predict_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
        dataloader_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        assert dataloader_idx is not None
        return self._step(batch, batch_idx, dataloader_idx, "predict")

    def compute_generative_perplexity(
        self,
        split: Literal["train", "val", "test", "predict"],
        dataloader_name: str,
        epoch: int,
        step: int,
        update_logged_predictions: bool = True,
    ) -> Optional[Dict[str, Any]]:
        if not self.generative_perplexity_evaluators:
            return None
        # read predictions
        predictions: List[Dict[str, Any]] = self.log_predictions.read(
            step, epoch, split, dataloader_name, self
        )
        if not predictions:
            logger.warning(
                f"No predictions found for {split} {dataloader_name} {epoch} {step}"
            )
            return None

        # Check if we have infilling data (raw_input and truth fields)
        has_infilling_data = (
            predictions
            and "raw_input" in predictions[0]
            and "truth" in predictions[0]
        )

        def compute_percentage_change(new_val: float, old_val: float) -> float:
            """Compute percentage change: ((new - old) / old) * 100"""
            if old_val == 0:
                return 0.0 if new_val == 0 else float("inf")
            return ((new_val - old_val) / old_val) * 100.0

        # get lengths and entropies for text (and raw_input/truth if present)
        for pred in predictions:
            # Compute for main text
            string = pred["text"]
            entropy, length = compute_entropy_and_length_for_sample(
                string, self.tokenizer
            )
            pred["entropy"] = entropy
            pred["length"] = length

            # Compute for raw_input and truth if present
            if has_infilling_data:
                raw_entropy, raw_length = (
                    compute_entropy_and_length_for_sample(
                        pred["raw_input"], self.tokenizer
                    )
                )
                pred["raw_entropy"] = raw_entropy
                pred["raw_length"] = raw_length

                truth_entropy, truth_length = (
                    compute_entropy_and_length_for_sample(
                        pred["truth"], self.tokenizer
                    )
                )
                pred["truth_entropy"] = truth_entropy
                pred["truth_length"] = truth_length

                # Compute basic percentage changes (not tied to specific evaluators)
                pred["pct_change_raw_length"] = compute_percentage_change(
                    length, raw_length
                )
                pred["pct_change_raw_entropy"] = compute_percentage_change(
                    entropy, raw_entropy
                )
                pred["pct_change_truth_length"] = compute_percentage_change(
                    length, truth_length
                )
                pred["pct_change_truth_entropy"] = compute_percentage_change(
                    entropy, truth_entropy
                )

        # generate perplexities and prepare for aggregation
        data = {
            "length": [],
            "entropy": [],  # models, own tokenizer
        }

        # Add raw_input and truth data keys if infilling data is present
        if has_infilling_data:
            data.update(
                {
                    "raw_length": [],
                    "raw_entropy": [],
                    "truth_length": [],
                    "truth_entropy": [],
                    "pct_change_raw_length": [],
                    "pct_change_raw_entropy": [],
                    "pct_change_truth_length": [],
                    "pct_change_truth_entropy": [],
                }
            )

        def add_keys(evaluator_name: str):
            data[f"nll_{evaluator_name}"] = []  # per sample mean nll
            data[f"total_nll_{evaluator_name}"] = []  # per sample sum nll
            data[f"length_{evaluator_name}"] = []  # per sample length
            data[f"entropy_{evaluator_name}"] = []  # per sample entropy
            data[f"perplexity_{evaluator_name}"] = []  # per sample perplexity

            # Add keys for raw_input and truth if infilling data is present
            if has_infilling_data:
                data[f"raw_nll_{evaluator_name}"] = []
                data[f"raw_total_nll_{evaluator_name}"] = []
                data[f"raw_length_{evaluator_name}"] = []
                data[f"raw_entropy_{evaluator_name}"] = []
                data[f"raw_perplexity_{evaluator_name}"] = []

                data[f"truth_nll_{evaluator_name}"] = []
                data[f"truth_total_nll_{evaluator_name}"] = []
                data[f"truth_length_{evaluator_name}"] = []
                data[f"truth_entropy_{evaluator_name}"] = []
                data[f"truth_perplexity_{evaluator_name}"] = []

                # Percentage change metrics
                data[f"pct_change_raw_nll_{evaluator_name}"] = []
                data[f"pct_change_raw_perplexity_{evaluator_name}"] = []
                data[f"pct_change_raw_length_{evaluator_name}"] = []
                data[f"pct_change_raw_entropy_{evaluator_name}"] = []
                data[f"pct_change_truth_nll_{evaluator_name}"] = []
                data[f"pct_change_truth_perplexity_{evaluator_name}"] = []
                data[f"pct_change_truth_length_{evaluator_name}"] = []
                data[f"pct_change_truth_entropy_{evaluator_name}"] = []

        for evaluator_name in self.generative_perplexity_evaluators:
            add_keys(evaluator_name)

        with torch.no_grad():
            for (
                evaluator_name,
                evaluator,
            ) in self.generative_perplexity_evaluators.items():
                with evaluator.loaded(
                    self.tokenizer,
                    self.device,
                ):
                    for pred in predictions:
                        # Process main text
                        string = pred["text"]
                        temp = compute_nll_for_sample(string, evaluator)
                        if temp is None:
                            continue
                        nll, _len, entropy = temp
                        pred[f"total_nll_{evaluator_name}"] = nll
                        pred[f"total_length_{evaluator_name}"] = _len
                        pred[f"entropy_{evaluator_name}"] = entropy
                        # per sample mean nll and perplexity
                        mean_nll = nll / _len
                        pred[f"nll_{evaluator_name}"] = mean_nll
                        perplexity = torch.exp(torch.tensor(mean_nll)).item()
                        pred[f"perplexity_{evaluator_name}"] = perplexity
                        data[f"nll_{evaluator_name}"].append(mean_nll)
                        data[f"total_nll_{evaluator_name}"].append(nll)
                        data[f"length_{evaluator_name}"].append(_len)
                        data[f"entropy_{evaluator_name}"].append(entropy)
                        data[f"perplexity_{evaluator_name}"].append(perplexity)
                        data["entropy"].append(pred["entropy"])
                        data["length"].append(pred["length"])

                        # Process raw_input and truth if present
                        if has_infilling_data:
                            # Process raw_input
                            raw_temp = compute_nll_for_sample(
                                pred["raw_input"], evaluator
                            )
                            if raw_temp is not None:
                                raw_nll, raw_len, raw_entropy = raw_temp
                                pred[f"raw_total_nll_{evaluator_name}"] = (
                                    raw_nll
                                )
                                pred[f"raw_total_length_{evaluator_name}"] = (
                                    raw_len
                                )
                                pred[f"raw_entropy_{evaluator_name}"] = (
                                    raw_entropy
                                )
                                raw_mean_nll = raw_nll / raw_len
                                pred[f"raw_nll_{evaluator_name}"] = (
                                    raw_mean_nll
                                )
                                raw_perplexity = torch.exp(
                                    torch.tensor(raw_mean_nll)
                                ).item()
                                pred[f"raw_perplexity_{evaluator_name}"] = (
                                    raw_perplexity
                                )

                                # Compute percentage changes w.r.t raw_input
                                pct_change_nll_raw = compute_percentage_change(
                                    mean_nll, raw_mean_nll
                                )
                                pct_change_perplexity_raw = (
                                    compute_percentage_change(
                                        perplexity, raw_perplexity
                                    )
                                )
                                pct_change_length_raw = (
                                    compute_percentage_change(_len, raw_len)
                                )
                                pct_change_entropy_raw = (
                                    compute_percentage_change(
                                        entropy, raw_entropy
                                    )
                                )
                                pred[
                                    f"pct_change_raw_nll_{evaluator_name}"
                                ] = pct_change_nll_raw
                                pred[
                                    f"pct_change_raw_perplexity_{evaluator_name}"
                                ] = pct_change_perplexity_raw
                                pred[
                                    f"pct_change_raw_length_{evaluator_name}"
                                ] = pct_change_length_raw
                                pred[
                                    f"pct_change_raw_entropy_{evaluator_name}"
                                ] = pct_change_entropy_raw

                                # Add to data for aggregation
                                data[f"raw_nll_{evaluator_name}"].append(
                                    raw_mean_nll
                                )
                                data[f"raw_total_nll_{evaluator_name}"].append(
                                    raw_nll
                                )
                                data[f"raw_length_{evaluator_name}"].append(
                                    raw_len
                                )
                                data[f"raw_entropy_{evaluator_name}"].append(
                                    raw_entropy
                                )
                                data[
                                    f"raw_perplexity_{evaluator_name}"
                                ].append(raw_perplexity)
                                data[
                                    f"pct_change_raw_nll_{evaluator_name}"
                                ].append(pct_change_nll_raw)
                                data[
                                    f"pct_change_raw_perplexity_{evaluator_name}"
                                ].append(pct_change_perplexity_raw)
                                data[
                                    f"pct_change_raw_length_{evaluator_name}"
                                ].append(pct_change_length_raw)
                                data[
                                    f"pct_change_raw_entropy_{evaluator_name}"
                                ].append(pct_change_entropy_raw)

                            # Process truth
                            truth_temp = compute_nll_for_sample(
                                pred["truth"], evaluator
                            )
                            if truth_temp is not None:
                                truth_nll, truth_len, truth_entropy = (
                                    truth_temp
                                )
                                pred[f"truth_total_nll_{evaluator_name}"] = (
                                    truth_nll
                                )
                                pred[
                                    f"truth_total_length_{evaluator_name}"
                                ] = truth_len
                                pred[f"truth_entropy_{evaluator_name}"] = (
                                    truth_entropy
                                )
                                truth_mean_nll = truth_nll / truth_len
                                pred[f"truth_nll_{evaluator_name}"] = (
                                    truth_mean_nll
                                )
                                truth_perplexity = torch.exp(
                                    torch.tensor(truth_mean_nll)
                                ).item()
                                pred[f"truth_perplexity_{evaluator_name}"] = (
                                    truth_perplexity
                                )

                                # Compute percentage changes w.r.t truth
                                pct_change_nll_truth = (
                                    compute_percentage_change(
                                        mean_nll, truth_mean_nll
                                    )
                                )
                                pct_change_perplexity_truth = (
                                    compute_percentage_change(
                                        perplexity, truth_perplexity
                                    )
                                )
                                pct_change_length_truth = (
                                    compute_percentage_change(_len, truth_len)
                                )
                                pct_change_entropy_truth = (
                                    compute_percentage_change(
                                        entropy, truth_entropy
                                    )
                                )
                                pred[
                                    f"pct_change_truth_nll_{evaluator_name}"
                                ] = pct_change_nll_truth
                                pred[
                                    f"pct_change_truth_perplexity_{evaluator_name}"
                                ] = pct_change_perplexity_truth
                                pred[
                                    f"pct_change_truth_length_{evaluator_name}"
                                ] = pct_change_length_truth
                                pred[
                                    f"pct_change_truth_entropy_{evaluator_name}"
                                ] = pct_change_entropy_truth

                                # Add to data for aggregation
                                data[f"truth_nll_{evaluator_name}"].append(
                                    truth_mean_nll
                                )
                                data[
                                    f"truth_total_nll_{evaluator_name}"
                                ].append(truth_nll)
                                data[f"truth_length_{evaluator_name}"].append(
                                    truth_len
                                )
                                data[f"truth_entropy_{evaluator_name}"].append(
                                    truth_entropy
                                )
                                data[
                                    f"truth_perplexity_{evaluator_name}"
                                ].append(truth_perplexity)
                                data[
                                    f"pct_change_truth_nll_{evaluator_name}"
                                ].append(pct_change_nll_truth)
                                data[
                                    f"pct_change_truth_perplexity_{evaluator_name}"
                                ].append(pct_change_perplexity_truth)
                                data[
                                    f"pct_change_truth_length_{evaluator_name}"
                                ].append(pct_change_length_truth)
                                data[
                                    f"pct_change_truth_entropy_{evaluator_name}"
                                ].append(pct_change_entropy_truth)

                            # Add raw_input and truth entropy/length to general data
                            data["raw_entropy"].append(pred["raw_entropy"])
                            data["raw_length"].append(pred["raw_length"])
                            data["truth_entropy"].append(pred["truth_entropy"])
                            data["truth_length"].append(pred["truth_length"])

                            # Add basic percentage changes
                            data["pct_change_raw_length"].append(
                                pred["pct_change_raw_length"]
                            )
                            data["pct_change_raw_entropy"].append(
                                pred["pct_change_raw_entropy"]
                            )
                            data["pct_change_truth_length"].append(
                                pred["pct_change_truth_length"]
                            )
                            data["pct_change_truth_entropy"].append(
                                pred["pct_change_truth_entropy"]
                            )

        # aggregate
        aggregated_data = {}
        for key in data:
            if (
                not key.startswith("total_nll_")
                and not key.startswith("raw_total_nll_")
                and not key.startswith("truth_total_nll_")
            ):
                aggregated_data[key] = torch.tensor(data[key]).double().mean()
                self.log(
                    f"{split}/{key}",
                    aggregated_data[key],
                    prog_bar=False,
                    sync_dist=False,
                    rank_zero_only=True,
                    logger=True,
                    add_dataloader_idx=False,
                )
            elif key.startswith("total_nll_"):
                evaluator_name = re.sub(r"total_nll_", "", key)
                total_length = (
                    torch.tensor(data[f"length_{evaluator_name}"])
                    .double()
                    .sum()
                )
                total_nll = torch.tensor(data[key]).double().sum()
                perplexity = torch.exp(total_nll / total_length).item()
                aggregated_data[f"total_perplexity_{evaluator_name}"] = (
                    perplexity
                )
                self.log(
                    f"{split}/total_perplexity_{evaluator_name}",
                    perplexity,
                    prog_bar=False,
                    sync_dist=False,
                    rank_zero_only=True,
                    logger=True,
                    add_dataloader_idx=False,
                )
            elif key.startswith("raw_total_nll_"):
                evaluator_name = re.sub(r"raw_total_nll_", "", key)
                total_length = (
                    torch.tensor(data[f"raw_length_{evaluator_name}"])
                    .double()
                    .sum()
                )
                total_nll = torch.tensor(data[key]).double().sum()
                perplexity = torch.exp(total_nll / total_length).item()
                aggregated_data[f"raw_total_perplexity_{evaluator_name}"] = (
                    perplexity
                )
                self.log(
                    f"{split}/raw_total_perplexity_{evaluator_name}",
                    perplexity,
                    prog_bar=False,
                    sync_dist=False,
                    rank_zero_only=True,
                    logger=True,
                    add_dataloader_idx=False,
                )
            elif key.startswith("truth_total_nll_"):
                evaluator_name = re.sub(r"truth_total_nll_", "", key)
                total_length = (
                    torch.tensor(data[f"truth_length_{evaluator_name}"])
                    .double()
                    .sum()
                )
                total_nll = torch.tensor(data[key]).double().sum()
                perplexity = torch.exp(total_nll / total_length).item()
                aggregated_data[f"truth_total_perplexity_{evaluator_name}"] = (
                    perplexity
                )
                self.log(
                    f"{split}/truth_total_perplexity_{evaluator_name}",
                    perplexity,
                    prog_bar=False,
                    sync_dist=False,
                    rank_zero_only=True,
                    logger=True,
                    add_dataloader_idx=False,
                )

        if update_logged_predictions:
            self.log_predictions.update_predictions(
                predictions,
                step,
                epoch,
                split,
                dataloader_name,
                self,
            )
        return aggregated_data

    def compute_post_hoc_metrics(
        self,
        split: Literal["train", "val", "test", "predict"],
        dataloader_name: str,
        epoch: int,
        step: int,
        update_logged_predictions: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Compute post-hoc metrics on logged predictions.

        Similar to compute_generative_perplexity, but for arbitrary post-hoc metrics.
        Loads predictions from jsonl, computes per-sample and global metrics,
        and logs aggregated results.

        Args:
            split: train/val/test/predict
            dataloader_name: Name of the dataloader
            epoch: Current epoch
            step: Current step
            update_logged_predictions: If True, update predictions jsonl with per-sample metrics

        Returns:
            Dictionary of aggregated metrics, or None if no evaluator
        """
        if self.post_hoc_evaluator is None:
            return None

        # 1. Read predictions from jsonl
        predictions: List[Dict[str, Any]] = self.log_predictions.read(
            step, epoch, split, dataloader_name, self
        )

        if not predictions:
            logger.warning(
                f"No predictions found for {split} {dataloader_name} {epoch} {step}"
            )
            return None

        # 2. Compute metrics using evaluator
        predictions, aggregated_metrics = self.post_hoc_evaluator.eval(
            predictions, tokenizer=self.tokenizer
        )
        # Add aggregated metrics to all predictions because we don't have access to the logged predictions directory.
        for prediction in predictions:
            prediction.update(**aggregated_metrics)

        # 3. Log aggregated metrics
        for metric_name, metric_value in aggregated_metrics.items():
            self.log(
                f"{split}/{metric_name}",
                metric_value,
                prog_bar=False,
                sync_dist=False,
                rank_zero_only=True,
                logger=True,
                add_dataloader_idx=False,
            )

        # 4. Update predictions jsonl with per-sample metrics
        if update_logged_predictions:
            self.log_predictions.update_predictions(
                predictions,
                step,
                epoch,
                split,
                dataloader_name,
                self,
            )

        return aggregated_metrics

    # endregion: Task-specific methods
    ############################################################

    ############################################################
    # region: Lightning Hooks

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        norm_order = 2.0
        norms = grad_norm(self, norm_type=norm_order)
        if f"grad_{norm_order}_norm_total" in norms:
            self.log(
                "Total gradient (norm)",
                norms[f"grad_{norm_order}_norm_total"],
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                sync_dist=False,
                rank_zero_only=True,
                logger=True,
                add_dataloader_idx=False,
            )

    def on_train_batch_start(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> None:
        if "train/0" in self.printed_batches:
            return
        self.datamodule.print_batch(batch, "train", 0)
        self.printed_batches.add("train/0")

    def on_validation_batch_start(
        self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        if f"val/{dataloader_idx}" in self.printed_batches:
            return
        self.trainer.datamodule.print_batch(batch, "val", dataloader_idx)
        self.printed_batches.add(f"val/{dataloader_idx}")

    def _call_log_predictions(
        self,
        batch: Dict[str, Any],
        outputs: Dict[str, Any],
        split: Literal["train", "val", "test", "predict"],
        dataloader_name: str,
        no_trainer_mode: bool = False,
    ) -> None:
        """Helper method to call log_predictions, supporting both single loggers and lists of loggers."""
        if self.log_predictions is not None:
            self.log_predictions(
                self,
                self.trainer if not no_trainer_mode else None,
                batch,
                outputs,
                split,
                dataloader_name,
            )

    def on_validation_batch_end(
        self,
        outputs: Dict[str, Any],
        batch: Dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        dataloader_name = self.dataloader_names["val"][dataloader_idx]
        if "prediction" in dataloader_name:
            self._call_log_predictions(batch, outputs, "val", dataloader_name)

    def on_test_batch_end(
        self,
        outputs: Dict[str, Any],
        batch: Dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        dataloader_name = self.dataloader_names["test"][dataloader_idx]
        if "prediction" in dataloader_name:
            self._call_log_predictions(batch, outputs, "test", dataloader_name)

    def on_predict_batch_end(
        self,
        outputs: Dict[str, Any],
        batch: Dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        dataloader_name = self.dataloader_names["predict"][dataloader_idx]
        if "prediction" in dataloader_name:
            self._call_log_predictions(
                batch, outputs, "predict", dataloader_name
            )

    def on_test_batch_start(
        self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        if f"test/{dataloader_idx}" in self.printed_batches:
            return
        self.trainer.datamodule.print_batch(batch, "test", dataloader_idx)
        self.printed_batches.add(f"test/{dataloader_idx}")

    def on_train_epoch_start(self) -> None:
        self.model.train()
        if hasattr(self.trainer, "datamodule") and hasattr(
            self.trainer.datamodule, "set_epoch"
        ):
            self.trainer.datamodule.set_epoch(self.trainer.current_epoch)
        else:
            logger.warning(
                "Could not set epoch for datamodule. Either you are not using a datamodule,"
                " or the datamodule does not support setting the epoch using `set_epoch()`."
                "This could lead to no shuffling in the dataloader between epoch if using IterableDataset."
            )

    def on_train_epoch_end(self) -> None:
        # reset train metrics if not passing metric objects to the log()
        pass

    def on_validation_epoch_start(self) -> None:
        self.model.eval()

    def on_validation_epoch_end(self) -> None:
        # reset val metrics if not passing metric objects to the log()
        if self.trainer.is_global_zero:
            for dataloader_name in self.dataloader_names["val"].values():
                if "prediction" in dataloader_name:
                    self.compute_generative_perplexity(
                        "val",
                        dataloader_name,
                        self.trainer.current_epoch,
                        self.trainer.global_step,
                        update_logged_predictions=True,
                    )
                    self.compute_post_hoc_metrics(
                        "val",
                        dataloader_name,
                        self.trainer.current_epoch,
                        self.trainer.global_step,
                        update_logged_predictions=True,
                    )

    def on_test_epoch_start(self) -> None:
        self.model.eval()

    def on_test_epoch_end(self) -> None:
        if self.trainer.is_global_zero:
            for dataloader_name in self.dataloader_names["test"].values():
                if "prediction" in dataloader_name:
                    self.compute_generative_perplexity(
                        "test",
                        dataloader_name,
                        self.trainer.current_epoch,
                        self.trainer.global_step,
                        update_logged_predictions=True,
                    )
                    self.compute_post_hoc_metrics(
                        "test",
                        dataloader_name,
                        self.trainer.current_epoch,
                        self.trainer.global_step,
                        update_logged_predictions=True,
                    )

    def on_predict_end(self) -> None:
        self.compute_generative_perplexity(
            "predict",
            "unconditional_prediction",
            self.trainer.current_epoch,
            self.trainer.global_step,
            update_logged_predictions=True,
        )
        self.compute_post_hoc_metrics(
            "predict",
            "unconditional_prediction",
            self.trainer.current_epoch,
            self.trainer.global_step,
            update_logged_predictions=True,
        )

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        if self.manual_ema_restore:
            if "ema" not in checkpoint:
                raise ValueError(
                    "EMA weights not found in checkpoint but manual_ema_restore is True"
                )
            from torch_ema import ExponentialMovingAverage

            # same code as in EMACallback.on_train_start
            ema = ExponentialMovingAverage(
                [p for p in self.parameters() if p.requires_grad],
                decay=self.decay,
                use_num_updates=self.use_num_updates,
            )
            ema.load_state_dict(checkpoint["ema"])
            ema.to(self.device)
            ema.copy_to()  # copy ema weights to model
            del ema

    # endregion: Lightning Hooks
    ############################################################

    ############################################################
    # region: other utilities

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        # CLEANUP: This method is to load old models that serialize rotary embedding buffers
        # Get the state_dict from the checkpoint (the key might be "state_dict")
        state_dict = checkpoint.get(
            "state_dict", checkpoint
        )  # entier lightning module's state_dict
        # Identify the keys corresponding to the rotary buffers. They will be in .model as well as .predictor.model
        ranked_logger.info(
            "Removing rotary embedding buffers from the state_dict."
        )
        keys_to_remove = [
            key
            for key in state_dict
            if "rotary_emb.sin" in key or "rotary_emb.cos" in key
        ]
        for key in keys_to_remove:
            state_dict.pop(key)

    def top_level_named_modules(
        self,
    ) -> Generator[Tuple[str, nn.Module], None, None]:
        yield "model", self.model

    def get_predictions_file_path(
        self,
        split: Literal["train", "val", "test"],
        dataloader_name: str,
        epoch: int,
        step: int,
    ) -> Path:
        return (
            self.predictions_dir
            / f"{split}/{dataloader_name}/{epoch=}_{step=}.jsonl"
        )

    # endregion: other utilities
    ############################################################

    ############################################################
    # region:  checkpoint management
    """Checkpoint and model weight management methods.

    This section provides methods for extracting, saving, loading, and managing
    model weights with proper EMA (Exponential Moving Average) handling.

    Key Constraints:
        - EMA state is managed by EMACallback during training
        - Harness cannot access EMA state from existing instance
        - EMA can ONLY be applied when loading from checkpoint file

    Method Categories:

    1. Helper Methods (Internal):
        - _extract_model_state_dict_from_lightning_state_dict()
        - _apply_ema_weights()

    2. Instance Methods (Current Model State):
        - extract_model_weights(): Get current model state dict
        - save_model_weights(path): Save to local file
        - load_model_weights(path): Load from local file
        - load_model_from_hub(repo_id): Load from HuggingFace Hub

    3. Class Methods (Create New Instance):
        - from_checkpoint(path, apply_ema=False): Load with optional EMA

    4. Hub Integration:
        - _save_pretrained(): Save to hub (via parent's push_to_hub())
        - _from_pretrained(): Not supported (raises NotImplementedError)

    Usage Examples:

        # Example 1: Extract model weights with EMA from existing training checkpoint and save them to a local file
        harness = Harness.from_checkpoint( # harness.model will have EMA weights
            "checkpoint.ckpt", # path to existing checkpoint
            apply_ema=True,
            cfg=cfg, # training config
            tokenizer=tokenizer,
            datamodule=datamodule
        )
        harness.save_model_weights("model_ema.pth")

        # Example 2: Push model weights with EMA to HuggingFace Hub
        harness = Harness.from_checkpoint(
            "checkpoint.ckpt",
            apply_ema=True,
            cfg=cfg,
            tokenizer=tokenizer,
            datamodule=datamodule
        )
        harness.push_to_hub(
            repo_id="username/my-model",
            commit_message="Upload model with EMA weights"
        )

        # Example 3: Load weights from hub into existing Harness
        harness = Harness(cfg, tokenizer=tokenizer, datamodule=datamodule)
        harness.load_model_from_hub("username/my-model")

        # Example 4: Load checkpoint without applying EMA (used for continued training from a checkpoint)
        harness = Harness.from_checkpoint(
            "checkpoint.ckpt",
            apply_ema=False,  # Use training weights, not EMA
            cfg=cfg,
            tokenizer=tokenizer,
            datamodule=datamodule
        )

        # Example 5: Extract and save without EMA (from in-memory model)
        harness = Harness(cfg, tokenizer=tokenizer, datamodule=datamodule)
        # ... train or modify model ...
        harness.save_model_weights("model.pth")

    Limitations:

        1. EMA Application:
           - Cannot apply EMA from existing Harness instance because we need to access EMA weights which are available on EMA callback object or checkpoint["ema"]. 
            - Cannot access self.trainer.callbacks reliably
            - Must use from_checkpoint(apply_ema=True) to get EMA weights

        2. HuggingFace Hub:
           - from_pretrained() not supported (NotImplementedError)
           - Cannot reconstruct Harness from hub config alone
           - Requires full Lightning setup (Hydra config, datamodule, tokenizer)
           - Use load_model_from_hub() to load weights into existing instance

        3. Instance Methods:
           - save_model_weights(), load_model_weights(), load_model_from_hub()
             work on current model state only
           - No EMA access from these methods

        4. Workflow for EMA:
           - Step 1: Load from checkpoint with apply_ema=True
           - Step 2: Save/push using instance methods
           - Cannot skip step 1 if EMA weights are needed

    File Formats:
        - Local files: PyTorch .pth format
        - HuggingFace Hub: SafeTensors format + config files
    """

    @staticmethod
    def _extract_model_state_dict_from_lightning_state_dict(
        state_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract model weights from full Lightning state dict.

        Args:
            state_dict: Full Lightning module state dict

        Returns:
            Model state dict with "model." prefix removed
        """
        model_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("model."):
                # Remove "model." prefix
                model_state_dict[key[6:]] = value
        return model_state_dict

    def _apply_ema_weights(self, ema_state_dict: Dict[str, Any]) -> None:
        """Apply EMA weights to the model from an EMA state dict.

        Args:
            ema_state_dict: EMA state dict from checkpoint["ema"]
        """
        from torch_ema import ExponentialMovingAverage

        decay = ema_state_dict.get("decay", 0.9999)
        num_updates = ema_state_dict.get("num_updates", None)
        use_num_updates = num_updates is not None

        logger.info(
            f"Applying EMA weights: decay={decay}, use_num_updates={use_num_updates}"
        )

        # Create EMA object with model's trainable parameters
        ema = ExponentialMovingAverage(
            [p for p in self.parameters() if p.requires_grad],
            decay=decay,
            use_num_updates=use_num_updates,
        )

        # Load the saved EMA state
        ema.load_state_dict(ema_state_dict)
        ema.to(self.device)

        # Apply EMA weights to the model (overwrites current weights)
        ema.copy_to()
        logger.info("EMA weights applied successfully")
        del ema

    def extract_model_weights(self) -> Dict[str, Any]:
        """Extract current model state dict.

        Returns:
            Model state dict (self.model.state_dict())
        """
        return self.model.state_dict()

    def save_model_weights(
        self, path: Union[str, Path], overwrite: bool = False
    ) -> None:
        """Save current model weights to local file.

        Args:
            path: Path to save the model weights
            overwrite: Whether to overwrite existing file

        Raises:
            ValueError: If file exists and overwrite is False
        """
        path = Path(path)
        if path.exists() and not overwrite:
            raise ValueError(
                f"Model weights file already exists at {path}. Use overwrite=True to overwrite."
            )

        path.parent.mkdir(parents=True, exist_ok=True)
        model_state_dict = self.extract_model_weights()
        torch.save(model_state_dict, path)
        logger.info(f"Model weights saved to {path}")

    def load_model_weights(
        self, path: Union[str, Path], strict: bool = True
    ) -> None:
        """Load model weights from local file into self.model.

        Args:
            path: Path to the model weights file
            strict: Whether to strictly enforce that the keys match
        """
        path = Path(path)
        if not path.exists():
            raise ValueError(f"Model weights file not found at {path}")

        logger.info(f"Loading model weights from {path}")
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=strict)
        logger.info("Model weights loaded successfully")

    def load_model_from_hub(
        self,
        repo_id: str,
        revision: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        force_download: bool = False,
        token: Optional[Union[str, bool]] = None,
        strict: bool = True,
        **kwargs,
    ) -> None:
        """Download and load model weights from HuggingFace Hub into self.model.

        This method downloads the model weights from the hub and loads them into
        the existing model. It does NOT reconstruct a new Harness instance.

        Args:
            repo_id: HuggingFace Hub repository ID (e.g., "username/model")
            revision: Git revision (branch, tag, or commit)
            cache_dir: Directory to cache downloaded files
            force_download: Force re-download even if cached
            token: HuggingFace Hub token for private repos
            strict: Whether to strictly enforce that the keys match
            **kwargs: Additional arguments for hf_hub_download
        """
        from huggingface_hub import hf_hub_download, constants
        import os

        logger.info(f"Downloading model weights from hub: {repo_id}")

        # Get token from environment if not provided
        if token is None:
            token = os.getenv("HF_HUB_KEY")

        # Download the model weights file
        weights_path = hf_hub_download(
            repo_id=repo_id,
            filename=constants.SAFETENSORS_SINGLE_FILE,
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            token=token,
            **kwargs,
        )

        logger.info(f"Loading model weights from {weights_path}")

        # Load weights using safetensors
        from safetensors.torch import load_model as load_model_as_safetensor

        load_model_as_safetensor(self.model, weights_path, strict=strict)
        logger.info("Model weights loaded successfully from hub")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        cfg: Optional[DictConfig] = None,
        tokenizer: Optional[Tokenizer] = None,
        datamodule: Optional[BaseDataModule] = None,
        apply_ema: bool = False,
        map_location: str = "cpu",
        **kwargs,
    ) -> "Harness":
        """Load Harness from Lightning checkpoint with optional EMA application.

        This is the ONLY method that can apply EMA weights, as it has direct access
        to the checkpoint file containing the EMA state.

        Args:
            checkpoint_path: Path to the Lightning checkpoint file
            cfg: Optional config to override checkpoint config
            tokenizer: Optional tokenizer instance
            datamodule: Optional datamodule instance
            apply_ema: Whether to apply EMA weights from checkpoint
            map_location: Device to load checkpoint to
            **kwargs: Additional arguments for load_from_checkpoint

        Returns:
            Harness instance with loaded weights (and EMA applied if requested)

        Example:
            # Load with EMA weights applied
            harness = Harness.from_checkpoint(
                "checkpoint.ckpt",
                apply_ema=True,
                cfg=cfg,
                tokenizer=tokenizer,
                datamodule=datamodule
            )
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint not found at {checkpoint_path}")

        logger.info(f"Loading Harness from checkpoint: {checkpoint_path}")

        # Prepare kwargs for load_from_checkpoint
        load_kwargs = {
            "checkpoint_path": checkpoint_path,
            "map_location": map_location,
        }
        if cfg is not None:
            load_kwargs["cfg"] = cfg
        if tokenizer is not None:
            load_kwargs["tokenizer"] = tokenizer
        if datamodule is not None:
            load_kwargs["datamodule"] = datamodule
        load_kwargs.update(kwargs)

        # Load the Lightning module
        harness = cls.load_from_checkpoint(**load_kwargs)

        # Apply EMA if requested
        if apply_ema:
            logger.info("Loading EMA state from checkpoint")
            checkpoint = torch.load(
                checkpoint_path, map_location=map_location, weights_only=False
            )

            if "ema" not in checkpoint:
                raise ValueError(
                    f"EMA state not found in checkpoint at {checkpoint_path}. "
                    "Cannot apply EMA weights."
                )

            ema_state_dict = checkpoint["ema"]
            harness._apply_ema_weights(ema_state_dict)
            logger.info("EMA weights applied to model")

        return harness

    # endregion:  checkpoint management
    ############################################################

    ############################################################
    # region: Hugging Face Hub methods

    def _save_pretrained(self, save_directory: str, **kwargs):
        """Save model weights and config to directory for HuggingFace Hub.

        This saves the current model state. If you need EMA weights, load from
        checkpoint first: Harness.from_checkpoint(path, apply_ema=True).

        Args:
            save_directory: Directory to save model and configs
            **kwargs: Additional arguments (unused)
        """
        from huggingface_hub import constants
        from safetensors.torch import save_model as save_model_as_safetensor

        logger.info(
            f"Saving model to {Path(save_directory) / constants.SAFETENSORS_SINGLE_FILE}"
        )

        # Save model weights using SafeTensors
        save_model_as_safetensor(
            self.model,
            str(Path(save_directory) / constants.SAFETENSORS_SINGLE_FILE),
        )  # type: ignore [arg-type]

        # Extract and save configs
        cfg = self.config

        # Save full config as YAML
        full_config = OmegaConf.to_yaml(cfg, resolve=False)
        config_path = Path(save_directory) / "full_config.yaml"
        with open(config_path, "w") as f:
            f.write(full_config)
        logger.info(f"Saved full config to {config_path}")

        # Save model config as JSON
        model_config = OmegaConf.to_container(cfg.model, resolve=True)
        model_config_path = Path(save_directory) / "config.json"
        with open(model_config_path, "w") as f:
            json.dump(model_config, f, indent=2)
        logger.info(f"Saved model config to {model_config_path}")

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: Optional[str],
        cache_dir: Optional[Union[str, Path]],
        force_download: bool,
        local_files_only: bool,
        token: Union[str, bool, None],
        map_location: str = "cpu",
        strict: bool = False,
        **model_kwargs,
    ):
        raise NotImplementedError(
            "Loading Harness via from_pretrained() is not supported. "
            "Harness requires complex Lightning setup (Hydra config, datamodule, tokenizer) "
            "that cannot be reconstructed from HF Hub config alone.\n\n"
            "Instead, use one of these workflows:\n"
            "1. Create Harness from checkpoint: harness = Harness.from_checkpoint(path, cfg=cfg, tokenizer=tokenizer, datamodule=datamodule)\n"
            "2. Load weights into existing Harness: harness.load_model_from_hub(repo_id)\n"
            "3. Instantiate normally then load: harness = Harness(cfg, ...); harness.load_model_from_hub(repo_id)"
        )

    # endregion: Hugging Face Hub methods
    ############################################################
