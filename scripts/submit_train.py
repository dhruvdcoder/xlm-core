#!/usr/bin/env python3
# fmt: off
import dotenv
# read env variables before anything else is imported
dotenv.load_dotenv(
    override=True
)  # set env variables from .env file, override=True is important
found_secretes = dotenv.load_dotenv(".secrets.env", override=True)
if not found_secretes:
    print("Warning: .secrets.env not found")
# fmt: on

import shlex
import sys
from typing import Dict
import re
from pathlib import Path
from typing import cast
import hydra
from omegaconf import DictConfig
from simple_slurm import Slurm
import omegaconf

# steal the raw args before Hydra’s decorator runs
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
    "config_name": "train_sbatch.yaml",
}


# resolvers
def _parse_gpu_count(gres: str) -> int:
    # pattern: gpu:4
    match = re.search(r"gpu:(\d+)", gres)
    if match:
        return int(match.group(1))
    # add other patterns here
    raise ValueError(f"Invalid gres: {gres}")


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


def validate_config(cfg: DictConfig) -> None:
    if cfg.train.debug is not None:
        # check the trainer_strategy, devices, num_nodes, precision, compile
        if cfg.train.trainer_strategy in ["ddp_multinode", "ddp"]:
            raise ValueError(
                "debug mode is not supported for multi-node training"
            )
        if cfg.train.devices > 1:
            raise ValueError(
                "debug mode is not supported for multi-GPU training"
            )
        if cfg.train.num_nodes > 1:
            raise ValueError(
                "debug mode is not supported for multi-node training"
            )


@hydra.main(**_HYDRA_PARAMS)
def main(cfg: DictConfig) -> None:
    """Main function to configure and submit SLURM job."""
    # Collect overrides for the inner script
    validate_config(cfg)
    # determine the logs folder
    logs_dir = Path(cfg.paths.log_dir) / cfg.job_name
    run_dir = Path(cfg.paths.run_dir)
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
    if cfg.train.debug is None:
        cmd = [
            "python",
            "-O",
            "src/xlm/commands/lightning_main.py",
            f"job_name={job_name}",
            f"job_type={cfg.train.job_type}",
            f"experiment={cfg.train.experiment}",
            f"per_device_batch_size={cfg.train.batch_size}",
            f"trainer_strategy={cfg.train.trainer_strategy}",
            f"trainer.devices={cfg.train.devices}",
            f"trainer.num_nodes={cfg.train.num_nodes}",
            f"++trainer.precision={cfg.train.precision}",
            f"compile={cfg.train.compile}",
            "+loggers.wandb.resume=allow",
            f"+loggers.wandb.id={job_name}",
        ]
    else:
        cmd = [
            "python",
            "-O",
            "src/xlm/commands/lightning_main.py",
            f"job_name={job_name}",
            f"job_type={cfg.train.job_type}",
            f"experiment={cfg.train.experiment}",
            f"debug={cfg.train.debug}",
        ]

    if INNER_ARGS:
        cmd += INNER_ARGS

    # Add srun command with the training command
    quoted = [shlex.quote(arg) for arg in cmd]
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
