#!/usr/bin/env python3
import shlex
import sys
from typing import Dict, List
import re
from pathlib import Path
from typing import cast
import hydra
from omegaconf import DictConfig
from simple_slurm import Slurm
import omegaconf
import subprocess
import json
import tempfile
import os
import traceback

# steal the raw args before Hydra's decorator runs
RAW_ARGS = sys.argv[1:]
if "---" in RAW_ARGS:
    sep = RAW_ARGS.index("---")
    INNER_ARGS = RAW_ARGS[sep + 1 :]
    OUTER_ARGS = RAW_ARGS[:sep]
else:
    INNER_ARGS = []
    OUTER_ARGS = RAW_ARGS

# rewrite sys.argv so Hydra only sees the outer bits
sys.argv = [sys.argv[0]] + OUTER_ARGS

# Hydra configuration parameters
_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": str(Path("../configs") / "slurm"),
    "config_name": "generate_sbatch.yaml",
}


# resolvers
def _parse_gpu_count(gres: str) -> int:
    # pattern: gpu:4
    match = re.search(r"gpu:(\d+)", gres)
    if match:
        return int(match.group(1))
    # add other patterns here
    raise ValueError(f"Invalid gres: {gres}")


def _ckpt_path_to_output_dir(ckpt_path: str) -> str:
    ckpt_path_ = Path(ckpt_path)
    # filename without extension
    ckpt_name = ckpt_path_.stem
    output_dir = ckpt_path_.parent.parent / "predictions" / ckpt_name
    return output_dir


def _determine_trainer_strategy(ntasks_per_node: int, nodes: int) -> str:
    if ntasks_per_node > 1:
        if nodes > 1:
            return "ddp_multinode"
        else:
            return "ddp"
    return "single_device"


omegaconf.OmegaConf.register_new_resolver("parse_gpu_count", _parse_gpu_count)
omegaconf.OmegaConf.register_new_resolver(
    "determine_trainer_strategy", _determine_trainer_strategy
)
omegaconf.OmegaConf.register_new_resolver(
    "ckpt_path_to_output_dir", _ckpt_path_to_output_dir
)


def validate_config(cfg: DictConfig) -> None:
    if cfg.generate.trainer_strategy in ["ddp_multinode", "ddp"]:
        raise ValueError("generation is not supported for multi-node")
    if cfg.generate.devices > 1:
        raise ValueError("generation is not supported for multi-GPU")
    if cfg.generate.num_nodes > 1:
        raise ValueError("generation is not supported for multi-node")


def determine_prediction_params(cmd: List[str]) -> Dict[str, str]:
    # Copy and modify the command to set job_type=print_predictor_params
    new_cmd = []
    job_type_set = False
    for arg in cmd:
        if arg.startswith("job_type="):
            new_cmd.append("job_type=print_predictor_params")
            job_type_set = True
        else:
            new_cmd.append(arg)
    if not job_type_set:
        new_cmd.append("++job_type=print_predictor_params")
    # Use a temp file for output
    with tempfile.NamedTemporaryFile(mode="r", delete=False) as tf:
        out_path = tf.name
    new_cmd.append(f"++predictor_json_out={out_path}")
    try:
        subprocess.run(
            new_cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        print("Subprocess failed!")
        print("Command:", e.cmd)
        print("Return code:", e.returncode)
        print("Stdout:\n", e.stdout)
        print("Stderr:\n", e.stderr)
        print("Stack trace:")
        traceback.print_exc()
        os.remove(out_path)
        raise
    with open(out_path, "r") as f:
        result = json.load(f)
    os.remove(out_path)
    return result


def prediction_params_to_filename(prediction_params: Dict[str, str]) -> str:
    sanitized = {}
    for k, v in prediction_params.items():
        if (
            k in ["max_length", "max_steps"]
            or "sampling_method"
            in k  # sampling_method and second_sampling_method
            or k.endswith("top")  # top and second_top
            or k.endswith("_k")  #
            or k.endswith("_p")  # second_p
            or k.endswith("p")  # p and second_p
            or "temp" in k  # temperature
            or "step" in k  # use_first_step_factor
            or "use" in k
        ):
            sanitized[k] = v

    return "__".join([f"{k}_{v}" for k, v in sanitized.items()]) + ".jsonl"


@hydra.main(**_HYDRA_PARAMS)
def main(cfg: DictConfig) -> None:
    """Main function to configure and submit SLURM job."""
    # Collect overrides for the inner script
    validate_config(cfg)
    # determine the logs folder
    run_dir = Path(cfg.paths.run_dir)
    output_dir = Path(cfg.paths.output_dir)
    # slurm_output_file = logs_dir / "%x.out"
    slurm_output_file = run_dir / "%x.out"
    slurm_config = cast(
        Dict, omegaconf.OmegaConf.to_container(cfg.slurm, resolve=True)
    )
    slurm_config["output"] = str(slurm_output_file)
    # Configure SLURM settings from config
    slurm = Slurm(**slurm_config)
    # add job_name

    # Set environment variables using slurm.add_cmd
    for key, value in cfg.env.items():
        slurm.add_cmd(f"export {key}={value}")

    # Get wandb job ID from command line args or use SLURM_JOB_NAME
    job_name = cfg.job_name

    # Print GPU info
    # slurm.add_cmd(
    #    "python -c 'import torch; print(\"num_gpus: \", torch.cuda.device_count())'"
    # )

    # Main training command with srun
    if cfg.generate.debug is None:
        cmd = [
            "python",
            "-O",
            "src/xlm/commands/lightning_main.py",
            f"job_name={job_name}",
            f"job_type={cfg.generate.job_type}",
            f"experiment={cfg.generate.experiment}",
            f"generation.ckpt_path={cfg.generate.ckpt_path}",
            f"datamodule.train_dataloader_kwargs.batch_size={cfg.generate.batch_size}",
            f"trainer_strategy={cfg.generate.trainer_strategy}",
            f"trainer.devices={cfg.generate.devices}",
            f"trainer.num_nodes={cfg.generate.num_nodes}",
            f"++trainer.precision={cfg.generate.precision}",
            f"compile={cfg.generate.compile}",
        ]
    else:
        cmd = [
            "python",
            "-O",
            "src/xlm/commands/lightning_main.py",
            f"job_name={job_name}",
            f"job_type={cfg.generate.job_type}",
            f"experiment={cfg.generate.experiment}",
            f"debug={cfg.generate.debug}",
        ]

    if INNER_ARGS:
        cmd += INNER_ARGS

    # Add srun command with the training command
    quoted = [shlex.quote(arg) for arg in cmd]

    # add the location of the output
    if (
        cfg.generate.output_dir is None
        or cfg.generate.output_file_name is None
    ):
        prediction_params = determine_prediction_params(quoted)
        output_file_name = prediction_params_to_filename(prediction_params)
        if cfg.generate.output_dir is None:
            quoted.append(shlex.quote(f"generation.output_dir={output_dir}"))
        else:
            quoted.append(
                shlex.quote(f"generation.output_dir={cfg.generate.output_dir}")
            )
        if cfg.generate.output_file_name is None:
            quoted.append(
                shlex.quote(f"generation.output_file_name={output_file_name}")
            )
    else:
        quoted.append(
            shlex.quote(
                f"generation.output_file_name={cfg.generate.output_file_name}"
            )
        )
        quoted.append(
            shlex.quote(f"generation.output_dir={cfg.generate.output_dir}")
        )

    slurm.add_cmd("srun " + " ".join(quoted))
    script = slurm.script()
    # Print the generated bash script
    print("Generated SLURM script:")
    # always print the script to the console
    print(script)

    # Save the generated bash script to a file
    # Submit the job
    if cfg.do == "submit":
        script_file = run_dir / "sbatch.sh"
        script_file.parent.mkdir(parents=True, exist_ok=True)
        with open(script_file, "w") as f:
            f.write(script)
        print(f"\nSLURM script saved to: {script_file}")

        job_id = slurm.sbatch()
        print(f"Submitted job with ID: {job_id}")


if __name__ == "__main__":
    main()
