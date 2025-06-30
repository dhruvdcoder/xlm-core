from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
from torch.distributed.algorithms.ddp_comm_hooks import powerSGD_hook
from xlm.utils.rank_zero import RankedLogger

logger = RankedLogger(name=__name__, rank_zero_only=True)


def get_powerSGD_hook():
    logger.info("Using PowerSGD communication hook for distributed training")
    return powerSGD_hook.powerSGD_hook


def get_powerSGD_state():
    # https://pytorch.org/docs/2.5/ddp_comm_hooks.html#powersgd-state
    powerSGD_state = powerSGD_hook.PowerSGDState(
        process_group=None,
        matrix_approximation_rank=2,
        start_powerSGD_iter=10000,  # before this no gradient compression is applied.
    )
    return powerSGD_state


def get_fp16_compression_hook():
    logger.info("Using FP16 compression hook for distributed training")
    return default_hooks.fp16_compress_hook


def get_bf16_compression_hook():
    logger.info("Using BF16 compression hook for distributed training")
    return default_hooks.bf16_compress_hook


def get_ddp_precision_hook(precision: str):
    if precision in ["16-mixed", "16-true", "16", "fp16"]:
        return get_fp16_compression_hook()
    elif precision in ["bf16-mixed", "bf16", "bf16-true"]:
        return get_bf16_compression_hook()
    else:
        logger.info(
            f"No precision hook for precision: {precision} for distributed training"
        )
        return None


def get_fp16_compression_wrapper():
    logger.info("Using FP16 compression wrapper for distributed training")
    return default_hooks.fp16_compress_wrapper


def get_bf16_compression_wrapper():
    logger.info("Using BF16 compression wrapper for distributed training")
    return default_hooks.bf16_compress_wrapper


def get_ddp_precision_wrapper(precision: str):
    if precision in ["16-mixed", "16-true", "16", "fp16"]:
        return get_fp16_compression_wrapper()
    elif precision in ["bf16-mixed", "bf16", "bf16-true"]:
        return get_bf16_compression_wrapper()
    else:
        logger.info(
            f"No precision wrapper for precision: {precision} for distributed training"
        )
