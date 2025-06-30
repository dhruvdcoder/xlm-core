import signal
import logging

logger = logging.getLogger(__name__)


def print_signal_handlers(prefix: str = ""):
    """Just print information about existing handlers for SIGTERM, SIGINT, SIGCONT, USR1 and USR2 signals."""

    for sig in [
        signal.SIGTERM,
        signal.SIGINT,
        signal.SIGCONT,
        signal.SIGUSR1,
        signal.SIGUSR2,
    ]:
        logger.warning(f"{prefix}: Signal {sig} has handler {signal.getsignal(sig)}")


def remove_handlers(signals, prefix: str = ""):
    # Set handlers to their default values
    for sig in signals:
        logger.warning(f"{prefix}: Removing handler for signal {sig}")
        signal.signal(sig, signal.SIG_DFL)


# See: https://github.com/dhruvdcoder/relational_icl/wiki/submitit_slurm

# Example init function for child processes
# def worker_init_function(datapipe: IterDataPipe, worker_info: Any):
#    print_signal_handlers(prefix=f"Worker {worker_info.worker_id}:")
#    remove_handlers(
#        [signal.SIGTERM], prefix=f"Worker {worker_info.worker_id}:"
#    )
#    return datapipe
