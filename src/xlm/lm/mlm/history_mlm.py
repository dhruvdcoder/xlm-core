from pathlib import Path
import uuid
from typing import Any, Dict, List, Optional, TypedDict
from jaxtyping import Float, Integer, Bool
from torch import Tensor as TT
import torch

from xlm import flags as debug

import logging

logger = logging.getLogger(__name__)


class StepResults(TypedDict):
    """
    Args:
        x: The input token ids.
        attention_mask: The mask for the input token ids.
        position_ids: The ids of the positions in the sequence.
        logits: The logits for the next token.
        t: The time step.
        current_step: The current step.
        change: The change in the logits.
        input_end_positions: The end positions of the input tokens.
    """

    x: Integer[TT, "batch seq_len"]
    attention_mask: Bool[TT, "batch seq_len"]
    position_ids: Integer[TT, "batch seq_len"]
    logits: Float[TT, "batch seq_len vocab_size"] | None
    t: Float[TT, " batch"]
    current_step: int
    change: Bool[TT, " batch seq_len"]
    input_end_positions: Integer[TT, "batch"]
    inp: Optional[Integer[TT, "batch seq"]]  # only present during debugging


class HistoryPluginBase:
    max_new_tokens: Optional[int]

    # region: utils
    def x(self, step_results):
        return step_results["x"]

    def logits(self, step_results):
        if step_results.get("logits") is None:
            return None
        return step_results["logits"].to(torch.float32)

    def t(self, step_results):
        return step_results["t"]

    def change(self, step_results):
        return step_results["change"]

    def current_step(self, step_results):
        return step_results["current_step"]

    def out_x(self, step_results):
        assert self.max_new_tokens is not None
        return step_results["x"][:, -self.max_new_tokens :]

    def out_inp(self, step_results):
        assert self.max_new_tokens is not None
        return step_results["inp"][:, -self.max_new_tokens :]

    def out_p(self, step_results):
        assert self.max_new_tokens is not None
        if (
            step_results["p"] is None
        ):  # before first step, or for some algorithms always
            return None
        return step_results["p"][:, -self.max_new_tokens :].to(torch.float32)

    def out_logits(self, step_results):
        assert self.max_new_tokens is not None
        if step_results["logits"] is None:  # before first step
            return None
        return step_results["logits"][:, -self.max_new_tokens :].to(
            torch.float32
        )

    # endregion: utils

    # region: interface
    def update_history(self, step_results: Any) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        """Called between examples."""
        raise NotImplementedError

    def to_dict(self) -> List[Dict]:
        raise NotImplementedError

    # TODO: support some common interface for querying
    # endregion: interface


class HistoryTopKPlugin(HistoryPluginBase):
    """We will dump the top k tokens and probs as tensors in separate files for each step."""

    def __init__(
        self,
        output_dir: str,
        k: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.output_dir = Path(output_dir)
        self.max_new_tokens = max_new_tokens
        # state
        self.unique_id = None
        self.k = k
        self.bs = None
        self.reset()
        # only needed while deserializing
        self.full_id = None  # including batch number
        # check if the right flag is set
        if not debug.DEBUG_VIZ_LOGITS:
            logger.warning("VIZ_LOGITS must be set to True")

    @classmethod
    def from_serialized(cls, output_dir: str, full_id: str):
        instance = cls(output_dir=output_dir)
        instance.full_id = full_id
        return instance

    def reset(self) -> None:
        # print a message if debugging
        if debug.DEBUG_VIZ_LOGITS and self.unique_id is not None:
            for item in self.output_dir.iterdir():
                if item.is_dir():
                    if item.name[:5] == self.unique_id[:5]:
                        print(
                            f"History dumped in {self.output_dir / item.name}"
                        )
        # do the reset
        self.unique_id = None
        self.bs = None

    def update_history(self, step_results: Any) -> None:
        out_x = self.out_x(
            step_results=step_results
        )  # shape (batch_size, max_new_tokens)
        probs = self.out_p(step_results=step_results)
        if probs is None:
            out_logits = self.out_logits(
                step_results=step_results
            )  # shape (batch_size, max_new_tokens, vocab_size)
            # out_inp = self.out_inp(step_results=step_results) # shape (batch_size, max_new_tokens)

            probs = (
                torch.softmax(out_logits, dim=-1)
                if out_logits is not None
                else None
            )
        k = self.k
        if k is None:
            if probs is not None:
                k = probs.shape[-1]
        top_probs, top_ids = (
            torch.topk(probs, k=k, dim=-1)
            if probs is not None
            else (None, None)
        )  # shape (batch_size, max_new_tokens, top_k)
        current_step = step_results["current_step"]
        # set state if empty
        if self.bs is None:
            self.unique_id = uuid.uuid4().hex
            self.bs = len(out_x)
        elif self.bs != len(out_x):
            raise ValueError("batchsize changed. Create a new instance.")
        for b in range(self.bs):
            dirname = f"{self.unique_id}_{b}"
            (self.output_dir / dirname).mkdir(parents=True, exist_ok=True)
            # save
            if top_probs is not None:
                torch.save(
                    top_probs[b],
                    self.output_dir / dirname / f"{current_step}_p.pt",
                )
                assert top_ids is not None
                torch.save(
                    top_ids[b],
                    self.output_dir / dirname / f"{current_step}_pids.pt",
                )
            torch.save(
                out_x[b], self.output_dir / dirname / f"{current_step}_x.pt"
            )

    def to_dict(self) -> List[Dict]:
        if self.bs is None or self.unique_id is None:
            raise ValueError
        return [{"fid": f"{self.unique_id}_{b}"} for b in range(self.bs)]

    def finalize() -> None:
        pass

    def get_num_steps(self):
        if self.full_id is None:
            raise RuntimeError
        dir: Path = self.output_dir / self.full_id
        # go through all the files in dir of "{step}_x.pt"
        last_step: int = list(
            sorted(map(lambda x: int(x.name[:-5]), dir.glob("*_x.pt")))
        )[-1]
        return last_step

    def get(self, step: int):
        if self.full_id is None:
            raise RuntimeError
        dir = self.output_dir / self.full_id
        p = (
            torch.load(dir / f"{step}_p.pt", weights_only=True)
            if (dir / f"{step}_p.pt").exists()
            else None
        )
        pids = (
            torch.load(dir / f"{step}_pids.pt", weights_only=True)
            if (dir / f"{step}_pids.pt").exists()
            else None
        )
        x = (
            torch.load(dir / f"{step}_x.pt", weights_only=True)
            if (dir / f"{step}_x.pt").exists()
            else None
        )
        return {"p": p, "pids": pids, "x": x}
