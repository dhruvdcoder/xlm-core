# %%
# change dir to the root of the project
# create the notebook inside the commands directory
import json
import os
from pathlib import Path
from typing import Any, Dict, Generator, cast

import torch

from xlm.harness import Harness
from xlm.utils.rich_utils import print_config_tree


if "PROJECT_ROOT" not in os.environ:
    os.environ["PROJECT_ROOT"] = "."
os.environ["HYDRA_FULL_ERROR"] = "1"

# region: Import necessary modules
import hydra
from omegaconf import DictConfig
from lightning import seed_everything
from xlm.utils.rank_zero import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)

# endregion


def generate(cfg: DictConfig):
    print_config_tree(cfg, resolve=True, save_to_file=False)
    if cfg.get("seed"):
        logger.info(f"Seed everything with seed {cfg.seed}")
        seed_everything(cfg.seed)

    # checkpoint pickup
    generation_ckpt_path = cfg.generation.ckpt_path
    generation_output_dir = cfg.generation.get("output_dir", None)
    generation_output_dir = (
        Path(generation_output_dir)
        if generation_output_dir is not None
        else None
    )
    generation_output_file_name = cfg.generation.get("output_file_name", None)
    if (
        generation_output_file_name is None
        and generation_output_dir is not None
    ):
        raise ValueError(
            "output_file_name is required when output_dir is provided"
        )
    generation_output_file_path = None
    if generation_output_file_name is not None:
        generation_output_file_path = (
            generation_output_dir / generation_output_file_name
        )

    if generation_output_file_path is not None:
        logger.info(
            f"Writing generation output to {generation_output_file_path}"
        )

    # prepare generation dataloader
    if cfg.generation.get("datamodule", None) is not None:
        datamodule = hydra.utils.instantiate(cfg.generation.datamodule)
        tokenizer = datamodule.tokenizer
    else:
        datamodule = hydra.utils.instantiate(cfg.datamodule)
        tokenizer = datamodule.tokenizer
    datamodule.prepare_data()
    datamodule.setup("predict")
    dataloader = datamodule.predict_dataloader()

    # instantiate the model
    torch.set_float32_matmul_precision("medium")
    if generation_ckpt_path is not None:
        module_cls = hydra.utils.get_class(cfg.lightning_module._target_)
        lightning_module = module_cls.load_from_checkpoint(
            checkpoint_path=generation_ckpt_path,
            tokenizer=tokenizer,
            datamodule=datamodule,
            cfg=cfg,  # chance to override the config of the checkpoint
        )
    else:
        lightning_module = hydra.utils.instantiate(
            cfg.lightning_module,
            tokenizer=tokenizer,
            datamodule=datamodule,
            cfg=cfg,
            _recursive_=False,
        )
    lightning_module = cast(Harness, lightning_module)
    lightning_module = lightning_module.to("cuda")
    lightning_module.eval()

    # check if we have model only checkpoint (replicating train functionality)
    model_only_ckpt_path = None
    if cfg.generation.get("model_only_checkpoint_path", None) is not None:
        if generation_ckpt_path is not None:
            logger.error(
                "generation.model_only_checkpoint_path and generation.ckpt_path cannot both be provided. "
                "We will use generation.ckpt_path for the model weights as well."
            )
        else:
            if not os.path.isfile(cfg.model_only_checkpoint_path):
                raise ValueError(
                    f"The model only checkpoint path {cfg.model_only_checkpoint_path} does not exist."
                )
            model_only_ckpt_path = cfg.model_only_checkpoint_path

    if model_only_ckpt_path is not None:
        message = lightning_module.model.load_state_dict(
            torch.load(model_only_ckpt_path)
        )
        logger.warning(
            f"Loading weights for `model` from a pretrained model at {model_only_ckpt_path} before generation"
        )
        logger.warning(message)

    # function to convert preds to serializable dict depending on the task.
    input_field_to_display = cfg.generation.get(
        "input_field_to_display", "input_ids"
    )

    def to_dict(preds, batch, batch_idx, dataloader_idx, dataloader_name):
        # we will implement a version for unconditional generation here.
        input_seqs = lightning_module.tokenizer.batch_decode(
            batch[input_field_to_display]
        )
        out_dicts = lightning_module.predictor.to_dict(
            batch,
            preds,
            batch_idx,
            dataloader_idx,
            dataloader_name,
        )
        assert len(out_dicts) == len(input_seqs)
        for i in range(len(out_dicts)):
            out_dicts[i]["input_text"] = input_seqs[i]
        return out_dicts

    prepare_input_batch_for_prediction = cfg.generation.get(
        "prepare_input_batch_for_prediction", False
    )
    if prepare_input_batch_for_prediction:
        # check if the model has a _prepare_input_batch_for_prediction method
        if not hasattr(lightning_module, "_prepare_input_batch_for_predict"):
            raise ValueError(
                "Model does not have a _prepare_input_batch_for_prediction method"
            )

    def sample_generator() -> Generator[Dict[str, Any], None, None]:
        # generates batches of predictions form the model
        for i, batch in enumerate(dataloader):
            batch = {
                k: (v.to("cuda") if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            if i == 0:
                if hasattr(datamodule, "print_batch"):
                    datamodule.print_batch(batch, "predict", 0)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if prepare_input_batch_for_prediction:
                    _batch = lightning_module._prepare_input_batch_for_predict(
                        batch
                    )
                else:
                    _batch = batch
                predicted_dict = lightning_module.predictor.predict(
                    _batch,
                    i,
                    dataloader_idx=None,
                    dataloader_name=(
                        dataloader.name
                        if hasattr(dataloader, "name")
                        else None
                    ),
                )
            yield from to_dict(
                predicted_dict,
                batch,
                i,
                dataloader_idx=None,
                dataloader_name=(
                    dataloader.name if hasattr(dataloader, "name") else None
                ),
            )

    fields_to_keep_in_output = cfg.generation.get(
        "fields_to_keep_in_output", ["input_text", "text"]
    )
    checked_fields = False

    def output_filter(out_dict: Dict[str, Any]) -> Dict[str, Any]:
        nonlocal checked_fields
        if not checked_fields:
            for field in fields_to_keep_in_output:
                if field not in out_dict:
                    raise ValueError(f"Field {field} not found in output")
            checked_fields = True
        return {
            k: v for k, v in out_dict.items() if k in fields_to_keep_in_output
        }

    # Do some evaluation here

    # pprint_template = """
    # --------------------------------
    # input_text: {input_text}\n
    # output_text: {text}\n
    # output_text_with_spl_tokens: {text_with_spl_tokens}\n
    # --------------------------------
    # """
    pprint_template = """
    --------------------------------
    input_text: {input_text}\n
    output_text: {text}\n
    --------------------------------
    """

    f = None
    max_examples = cfg.generation.get("max_examples", None)
    try:
        if generation_output_file_path is not None:
            generation_output_file_path.parent.mkdir(
                parents=True, exist_ok=True
            )
            f = open(
                generation_output_file_path,
                "w",
            )

        with torch.no_grad():
            for i, out_dict in enumerate(sample_generator()):
                if max_examples is not None and i > max_examples:
                    logger.info(f"Generated {i} predictions")
                    logger.info(
                        f"Stopping generation at {max_examples} examples"
                    )
                    break

                if i % 100 == 0:
                    logger.info(f"Generated {i} predictions")

                if f is not None:
                    f.write(
                        json.dumps(
                            output_filter(out_dict),
                        )
                        + "\n"
                    )
                    f.flush()
                else:
                    print(
                        pprint_template.format(
                            **out_dict,
                        )
                    )
    finally:
        if f is not None:
            f.flush()
            f.close()
        logger.info("Generation complete")
