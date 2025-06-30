# Reference: https://github.com/allenai/allennlp/blob/80fb6061e568cb9d6ab5d45b661e86eb61b92c82/allennlp/common/tqdm.py#L85
import logging
import sys
from time import time
from typing import Optional

try:
    SHELL = str(type(get_ipython()))  # type:ignore # noqa: F821
except:  # noqa: E722
    SHELL = ""

if "zmqshell.ZMQInteractiveShell" in SHELL:
    from tqdm import tqdm_notebook as _tqdm
else:
    from tqdm import tqdm as _tqdm

from lightning.pytorch.callbacks.progress.tqdm_progress import Tqdm
from lightning.pytorch.callbacks.progress.tqdm_progress import Tqdm as pl_tqdm
from lightning.pytorch.callbacks.progress.tqdm_progress import TQDMProgressBar

# This is necessary to stop tqdm from hanging
# when exceptions are raised inside iterators.
# It should have been fixed in 4.2.1, but it still
# occurs.
# https://github.com/tqdm/tqdm/issues/469
_tqdm.monitor_interval = 0


logger = logging.getLogger("tqdm")
logger.propagate = False


def replace_cr_with_newline(message: str) -> str:
    """
    TQDM and requests use carriage returns to get the training line to update for each batch
    without adding more lines to the terminal output. Displaying those in a file won't work
    correctly, so we'll just make sure that each batch shows up on its one line.
    """
    # In addition to carriage returns, nested progress-bars will contain extra new-line
    # characters and this special control sequence which tells the terminal to move the
    # cursor one line up.
    message = message.replace("\r", "").replace("\n", "").replace("[A", "")
    if message and message[-1] != "\n":
        message += "\n"
    return message


class TqdmToLogsWriter(object):
    def __init__(self):
        self.last_message_written_time = 0.0

    def write(self, message):
        file_friendly_message = replace_cr_with_newline(message)
        if file_friendly_message.strip():
            sys.stdout.write(file_friendly_message)

    def flush(self):
        sys.stderr.flush()


def create_ff_tqdm(*args, **kwargs):
    # Use a slower interval when FILE_FRIENDLY_LOGGING is set.

    new_kwargs = {
        "file": TqdmToLogsWriter(),
        **kwargs,
    }

    return pl_tqdm(*args, **new_kwargs)


class FileFriendlyTQDMProgressBar(TQDMProgressBar):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.refresh_rate < 100:
            logger.warning(
                f"Got refresh_rate={self.refresh_rate}."
                " This is too low to be file-friendly."
            )

    def init_sanity_tqdm(self) -> Tqdm:
        bar = create_ff_tqdm(
            desc=self.sanity_check_description,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=True,
            bar_format=self.BAR_FORMAT,
        )
        return bar

    def init_train_tqdm(self) -> Tqdm:
        return create_ff_tqdm(
            desc=self.train_description,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            smoothing=0,
            bar_format=self.BAR_FORMAT,
        )

    def init_predict_tqdm(self) -> Tqdm:
        return create_ff_tqdm(
            desc=self.predict_description,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            smoothing=0,
            bar_format=self.BAR_FORMAT,
        )

    def init_validation_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for validation."""
        # The train progress bar doesn't exist in `trainer.validate()`
        has_main_bar = self.trainer.state.fn != "validate"
        return create_ff_tqdm(
            desc=self.validation_description,
            position=(2 * self.process_position + has_main_bar),
            disable=self.is_disabled,
            leave=not has_main_bar,
            dynamic_ncols=True,
            bar_format=self.BAR_FORMAT,
        )

    def init_test_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for testing."""
        return create_ff_tqdm(
            desc="Testing",
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            bar_format=self.BAR_FORMAT,
        )


class TqdmToLogsWriter2(object):
    def __init__(self):
        self.last_message_written_time = 0.0

    def write(self, message):
        file_friendly_message: Optional[str] = None
        file_friendly_message = replace_cr_with_newline(message)
        if file_friendly_message.strip():
            sys.stderr.write(file_friendly_message)

        # Every 10 seconds we also log the message.
        now = time()
        if now - self.last_message_written_time >= 10 or "100%" in message:
            if file_friendly_message is None:
                file_friendly_message = replace_cr_with_newline(message)
            for message in file_friendly_message.split("\n"):
                message = message.strip()
                if len(message) > 0:
                    logger.info(message)
                    self.last_message_written_time = now

    def flush(self):
        sys.stderr.flush()


class Tqdm:
    @staticmethod
    def tqdm(*args, **kwargs):
        # Use a slower interval when FILE_FRIENDLY_LOGGING is set.
        # default_mininterval = 2.0 if FILE_FRIENDLY_LOGGING else 0.1

        new_kwargs = {
            "file": TqdmToLogsWriter2(),
            # "mininterval": default_mininterval,
            **kwargs,
        }

        return _tqdm(*args, **new_kwargs)

    @staticmethod
    def set_lock(lock):
        _tqdm.set_lock(lock)

    @staticmethod
    def get_lock():
        return _tqdm.get_lock()
