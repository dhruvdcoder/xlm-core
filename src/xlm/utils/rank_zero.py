from functools import lru_cache
from typing import Any, Mapping, Optional, Union
from lightning_utilities.core.rank_zero import (
    rank_zero_info,
    rank_zero_warn,
    rank_zero_debug,
    rank_zero_only,
    rank_prefixed_message,
)
import logging


class RankedLogger(logging.LoggerAdapter):
    """A multi-GPU-friendly python command line logger."""

    def __init__(
        self,
        name: str = __name__,
        rank_zero_only: bool = False,
        extra: Optional[Mapping[str, object]] = None,
        stacklevel: int = 2,
    ) -> None:
        """Initializes a multi-GPU-friendly python command line logger that logs on all processes
        with their rank prefixed in the log message.

        :param name: The name of the logger. Default is ``__name__``.
        :param rank_zero_only: Whether to force all logs to only occur on the rank zero process. Default is `False`.
        :param extra: (Optional) A dict-like object which provides contextual information. See `logging.LoggerAdapter`.
        :param stacklevel: The stack level to use for finding the caller. Default is 2.
        """
        logger = logging.getLogger(name)
        super().__init__(logger=logger, extra=extra)
        self.rank_zero_only = rank_zero_only
        self.stacklevel = stacklevel

    def log(
        self, level: int, msg: str, *args, rank: Optional[int] = None, **kwargs
    ) -> None:
        """Delegate a log call to the underlying logger, after prefixing its message with the rank
        of the process it's being logged from. If `'rank'` is provided, then the log will only
        occur on that rank/process.

        :param level: The level to log at. Look at `logging.__init__.py` for more information.
        :param msg: The message to log.
        :param rank: The rank to log at.
        :param args: Additional args to pass to the underlying logging function.
        :param kwargs: Any additional keyword args to pass to the underlying logging function.
        """
        if self.isEnabledFor(level):
            msg, kwargs = self.process(msg, kwargs)
            current_rank = getattr(rank_zero_only, "rank", None)
            if current_rank is None:
                raise RuntimeError(
                    "The `rank_zero_only.rank` needs to be set before use"
                )
            msg = rank_prefixed_message(msg, current_rank)

            if self.rank_zero_only:
                if current_rank == 0:
                    self.logger.log(
                        level, msg, *args, stacklevel=self.stacklevel, **kwargs
                    )
            else:
                if rank is None:
                    self.logger.log(
                        level, msg, *args, stacklevel=self.stacklevel, **kwargs
                    )
                elif current_rank == rank:
                    self.logger.log(
                        level, msg, *args, stacklevel=self.stacklevel, **kwargs
                    )


def get_rank_zero_logger(
    name: str,
) -> RankedLogger:
    """Get a multi-GPU-friendly python command line logger.

    :param name: The name of the logger.
    :return: A multi-GPU-friendly python command line logger.
    """
    return RankedLogger(name=name, rank_zero_only=True)


@lru_cache(maxsize=None)
def warn_once(logger, msg: str):
    logger.warning(msg)
