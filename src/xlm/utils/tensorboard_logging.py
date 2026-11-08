import lightning as L
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger


class LogGradientsToTensorBoard(L.pytorch.Callback):
    def __init__(self, log_every_n_steps: int = 10, log_norms: bool = False):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.log_norms = log_norms

    # The accumulated gradient is the one the optimizer steps on.
    # on_after_backward fires every microbatch, while global_step stays put.
    def on_before_optimizer_step(
        self, trainer: L.Trainer, pl_module: L.LightningModule, optimizer
    ) -> None:
        del optimizer
        global_step = trainer.global_step
        if global_step % self.log_every_n_steps != 0:
            return
        tb_logger = _tensorboard_logger(trainer)
        if not trainer.is_global_zero:
            return
        for top_name, module in pl_module.top_level_named_modules():
            for name, param in module.named_parameters():
                tb_logger.experiment.add_histogram(
                    f"{top_name}/{name}",
                    param.detach().float(),
                    global_step,
                )
                if param.requires_grad and param.grad is not None:
                    tb_logger.experiment.add_histogram(
                        f"{top_name}/{name}_grad",
                        param.grad.detach().float(),
                        global_step,
                    )
                if self.log_norms:
                    tb_logger.experiment.add_scalar(
                        f"{top_name}/{name}_norm",
                        param.detach().float().norm(),
                        global_step,
                    )
                    if param.requires_grad and param.grad is not None:
                        tb_logger.experiment.add_scalar(
                            f"{top_name}/{name}_grad_norm",
                            param.grad.detach().float().norm(),
                            global_step,
                        )


def _tensorboard_logger(trainer: L.Trainer) -> TensorBoardLogger:
    for logger in trainer.loggers:
        if isinstance(logger, TensorBoardLogger):
            return logger
    raise RuntimeError(
        "LogGradientsToTensorBoard requires a TensorBoardLogger in trainer.loggers."
    )
