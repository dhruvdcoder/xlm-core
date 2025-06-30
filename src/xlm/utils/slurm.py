import os
import subprocess
from typing import Optional


def get_slurm_job_id() -> Optional[str]:
    return os.environ.get("SLURM_JOB_ID", None)


def get_output_file_path() -> Optional[str]:
    job_id = get_slurm_job_id()
    if job_id:
        cmd = f"scontrol show job {job_id} | grep StdOut | awk -F= '{{print $2}}'"
        output_file = subprocess.check_output(
            cmd, shell=True, text=True
        ).strip()
        return output_file if os.path.exists(output_file) else None
    else:
        return None


def get_slurm_job_out_filename():
    if "SLURM_JOB_OUT" in os.environ:
        return os.environ["SLURM_JOB_OUT"]
    else:
        return None


def print_slurm_info():
    print("SLURM INFO")
    print("-" * 80)
    out_file = get_output_file_path()
    if out_file is not None:
        print(f"SLURM_JOB_OUT: {out_file}")
    else:
        print("SLURM_JOB_OUT: None")
    print("-" * 80)
