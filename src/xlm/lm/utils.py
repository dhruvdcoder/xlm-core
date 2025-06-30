import json
from typing import (
    Any,
    Dict,
    Literal,
)


from xlm.utils.rank_zero import rank_zero_only


class DefaultPredictionsLogger:

    def prepare_predictions(
        self,
        predictor,
        batch: Dict[str, Any],
        preds: Dict[str, Any],
        split: Literal["train", "val", "test"],
        dataloader_name: str,
    ):
        step = self.trainer.global_step or 0
        epoch = self.trainer.current_epoch or 0
        file_path = self.get_predictions_file_path(
            split, dataloader_name, epoch, step
        )
        file_path.parent.mkdir(parents=True, exist_ok=True)
        logger_ = [
            l_ for l_ in self.trainer.loggers if hasattr(l_, "log_text")
        ]
        text = []  # list of rows

        with open(file_path, "a") as f:
            for dict_ in self.predictor.to_dict(
                batch, preds, dataloader_name=dataloader_name
            ):
                text.append(dict_["text"])
                f.write(json.dumps(dict_) + "\n")
        # only log one set of predictions per eval run.
        if (
            self.trainer.global_step
            > self.last_global_step_logged_at_which_logged_predictions
        ):
            n_rows = 10
            for logger_ in logger_:
                logger_.log_text(
                    f"{split}/{dataloader_name}",
                    ["text"],  # column names
                    [[_text] for _text in text[:n_rows]],  # rows
                    step=self.trainer.global_step,
                )
            self.last_global_step_logged_at_which_logged_predictions = (
                self.trainer.global_step
            )


@rank_zero_only
def log_predictions(
    self,
    batch: Dict[str, Any],
    preds: Dict[str, Any],
    split: Literal["train", "val", "test"],
    dataloader_name: str,
):
    step = self.trainer.global_step or 0
    epoch = self.trainer.current_epoch or 0
    file_path = self.get_predictions_file_path(
        split, dataloader_name, epoch, step
    )
    file_path.parent.mkdir(parents=True, exist_ok=True)
    logger_ = [l_ for l_ in self.trainer.loggers if hasattr(l_, "log_text")]
    text = []  # list of rows

    with open(file_path, "a") as f:
        for dict_ in self.predictor.to_dict(
            batch, preds, dataloader_name=dataloader_name
        ):
            text.append(dict_["text"])
            f.write(json.dumps(dict_) + "\n")
    # only log one set of predictions per eval run.
    if (
        self.trainer.global_step
        > self.last_global_step_logged_at_which_logged_predictions
    ):
        n_rows = 10
        for logger_ in logger_:
            logger_.log_text(
                f"{split}/{dataloader_name}",
                ["text"],  # column names
                [[_text] for _text in text[:n_rows]],  # rows
                step=self.trainer.global_step,
            )
        self.last_global_step_logged_at_which_logged_predictions = (
            self.trainer.global_step
        )
