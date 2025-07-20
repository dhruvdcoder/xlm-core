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


def get_slurm_vars():
    slurm_vars = {}
    for k, v in os.environ.items():
        if k.startswith("SLURM_"):
            slurm_vars[k] = v
    return slurm_vars


def print_slurm_info():
    """Typical output:
    SLURM INFO
    --------------------------------------------------------------------------------
    SLURM_STEP_NUM_TASKS: 1
    SLURM_JOB_USER: dhruveshpate_umass_edu
    SLURM_TASKS_PER_NODE: 1
    SLURM_JOB_UID: 31803
    SLURM_TASK_PID: 3169823
    SLURM_JOB_GPUS: 0
    SLURM_LOCALID: 0 # not needed
    SLURM_SUBMIT_DIR: /home/dhruveshpate_umass_edu # not needed
    SLURM_JOB_START_TIME: 1752936994 # not needed
    SLURM_STEP_NODELIST: gpu027
    SLURM_SPACK_ROOT: /usr # not needed
    SLURM_CLUSTER_NAME: unity # not needed
    SLURM_JOB_END_TIME: 1752951394 # not needed
    SLURM_PMI2_SRUN_PORT: 41735
    SLURM_CPUS_ON_NODE: 12
    SLURM_JOB_CPUS_PER_NODE: 12
    SLURM_GPUS_ON_NODE: 1
    SLURM_GTIDS: 0 # not needed
    SLURM_JOB_PARTITION: gpu
    SLURM_TRES_PER_TASK: cpu=12
    SLURM_OOM_KILL_STEP: 0 # not needed
    SLURM_JOB_NUM_NODES: 1
    SLURM_STEPID: 4294967290
    SLURM_JOBID: 40112524
    SLURM_GPUS: 1
    SLURM_PTY_PORT: 43041 # not needed
    SLURM_LAUNCH_NODE_IPADDR: 10.100.1.2
    SLURM_JOB_QOS: short
    SLURM_PTY_WIN_ROW: 46 # not needed
    SLURM_PMI2_PROC_MAPPING: (vector,(0,1,1)) # not needed
    SLURM_PROCID: 0 # not needed
    SLURM_CPUS_PER_TASK: 12
    SLURM_TOPOLOGY_ADDR: .ib-core.tors-r1pac16.gpu027 # not needed
    SLURM_STEPMGR: gpu027
    SLURM_TOPOLOGY_ADDR_PATTERN: switch.switch.switch.node # not needed
    SLURM_SRUN_COMM_HOST: 10.100.1.2
    SLURM_SCRIPT_CONTEXT: prolog_task # not needed
    SLURM_MEM_PER_NODE: 40960
    SLURM_PTY_WIN_COL: 157 # not needed
    SLURM_NODELIST: gpu027
    SLURM_SRUN_COMM_PORT: 40067
    SLURM_STEP_ID: 4294967290
    SLURM_JOB_ACCOUNT: pi_mccallum_umass_edu # not needed
    SLURM_PRIO_PROCESS: 0 # not needed
    SLURM_NNODES: 1
    SLURM_SUBMIT_HOST: login2
    SLURM_JOB_ID: 40112524
    SLURM_NODEID: 0
    SLURM_STEP_NUM_NODES: 1
    SLURM_INCLUDE_DIR: /usr/include # not needed
    SLURM_STEP_TASKS_PER_NODE: 1
    SLURM_MPI_TYPE: pmi2 # not needed
    SLURM_PMI2_STEP_NODES: gpu027 # not needed
    SLURM_CONF: /var/spool/slurm/slurmd/conf-cache/slurm.conf # not needed
    SLURM_JOB_NAME: interactive
    SLURM_LIB_DIR: /usr/lib/x86_64-linux-gnu # not needed
    SLURM_STEP_LAUNCHER_PORT: 40067
    SLURM_JOB_GID: 31803
    SLURM_JOB_NODELIST: gpu027
    SLURM_JOB_OUT: None
    --------------------------------------------------------------------------------
    """
    ignore = {
        "SLURM_LOCALID",
        "SLURM_SUBMIT_DIR",
        "SLURM_JOB_START_TIME",
        "SLURM_SPACK_ROOT",
        "SLURM_CLUSTER_NAME",
        "SLURM_JOB_END_TIME",
        "SLURM_GTIDS",
        "SLURM_OOM_KILL_STEP",
        "SLURM_PTY_PORT",
        "SLURM_PTY_WIN_ROW",
        "SLURM_PMI2_PROC_MAPPING",
        "SLURM_PROCID",
        "SLURM_TOPOLOGY_ADDR",
        "SLURM_TOPOLOGY_ADDR_PATTERN",
        "SLURM_SCRIPT_CONTEXT",
        "SLURM_PTY_WIN_COL",
        "SLURM_JOB_ACCOUNT",
        "SLURM_PRIO_PROCESS",
        "SLURM_INCLUDE_DIR",
        "SLURM_MPI_TYPE",
        "SLURM_PMI2_STEP_NODES",
        "SLURM_CONF",
        "SLURM_LIB_DIR",
    }
    try:
        print("SLURM INFO")
        print("-" * 80)
        slurm_vars = get_slurm_vars()
        for k, v in slurm_vars.items():
            if k not in ignore:
                print(f"{k}: {v}")
        out_file = get_output_file_path()
        if out_file is not None:
            print(f"SLURM_JOB_OUT: {out_file}")
        else:
            print("SLURM_JOB_OUT: None")
        print("-" * 80)
    except Exception as e:
        print(f"Error getting SLURM info: {e}")
