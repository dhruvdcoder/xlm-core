import os


def get_num_processes() -> int:
    return len(os.sched_getaffinity(0))  # type: ignore


def is_notebook() -> bool:
    try:
        shell = str(type(get_ipython()))
    except:
        shell = ""
    return "zmqshell.ZMQInteractiveShell" in shell
