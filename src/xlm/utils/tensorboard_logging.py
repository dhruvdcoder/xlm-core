import lightning as L


class LogGradientsToTensorBoard(L.pytorch.Callback):
    def __init__(self, log_every_n_steps: int = 10):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps

    # Ref: https://github.com/Lightning-AI/pytorch-lightning/issues/2660
    def on_after_backward(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        global_step = trainer.global_step
        assert isinstance(pl_module.logger, L.pytorch.loggers.tensorboard.TensorBoardLogger)  # type: ignore
        if global_step % self.log_every_n_steps == 0:
            for top_name, module in pl_module.top_level_named_modules():
                for name, param in module.named_parameters():
                    pl_module.logger.experiment.add_histogram(
                        f"{top_name}/{name}", param, global_step
                    )
                    if param.requires_grad:
                        if param.grad is not None:
                            pl_module.logger.experiment.add_histogram(
                                f"{top_name}/{name}_grad",
                                param.grad,
                                global_step,
                            )
