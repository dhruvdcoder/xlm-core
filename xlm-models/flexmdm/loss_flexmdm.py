from typing import Optional, cast

import torch
import torch.nn.functional as F
from .noise_flexmdm import FlexMDMNoiseSchedule
from .types_flexmdm import FlexMDMBatch, FlexMDMLossDict, FlexMDMModel
from xlm.harness import LossFunction, Harness
from xlm.datamodule import Tokenizer
from xlm.utils.nn import masked_mean


def sample_time(batch_size: int, device: torch.device) -> torch.Tensor:
    eps = 1e-6
    interval = 1.0 - eps
    interval_size = interval / batch_size
    u = torch.rand(batch_size, device=device)
    return (
        torch.arange(batch_size, device=device, dtype=u.dtype) + u
    ) * interval_size


def get_noised_sequence(
    ids: torch.Tensor,
    batch_size: int,
    max_seq_len: int,
    t: torch.Tensor,
    fixed: torch.Tensor,
    noise_schedule: FlexMDMNoiseSchedule,
    tokenizer: Tokenizer,
) -> torch.Tensor:

    eps = 1e-6
    insertion_time = noise_schedule.insertion_noise_schedule.sample(
        (batch_size, max_seq_len), device=t.device
    )
    insertion_time = eps + (1 - eps) * insertion_time  # ensure t1 is not zero
    unmasking_time = noise_schedule.unmasking_noise_schedule.sample_truncated(
        insertion_time, (batch_size, max_seq_len), device=t.device
    )
    
    # for all fixed tokens, set corresponding times to 1 so that they are not noised
    insertion_time = torch.where(fixed.bool(), 0, insertion_time)
    unmasking_time = torch.where(fixed.bool(), 0, unmasking_time)

    x1 = ids  # original sequence
    pad_token_id = tokenizer.pad_token_id
    mask_token_id = tokenizer.mask_token_id
    eos_token_id = tokenizer.eos_token_id
    
    clean_tokens = x1.ne(pad_token_id)
    deleted_tokens = clean_tokens & (t[:, None] < insertion_time)
    masked_tokens = (
        clean_tokens
        & (t[:, None] >= insertion_time)
        & (t[:, None] < unmasking_time)
    )

    xt = torch.where(
        deleted_tokens,
        pad_token_id,  # for deletion, change to pad token
        torch.where(
            masked_tokens,
            mask_token_id,  # for masking, change to mask token
            x1,
        ),
    )

    # st: original positions of the non-deleted tokens
    st = xt.ne(pad_token_id).argsort(dim=1, descending=True, stable=True)
    xt = torch.gather(xt, 1, st)  # squeeze together the non-deleted tokens
    st[xt == pad_token_id] = 0

    x1_len = (x1 != pad_token_id).sum(dim=1)
    xt_len = (xt != pad_token_id).sum(dim=1)

    # gaps: will hold gap length for the gap that is left of the non-deleted token
    temp = st.clone()

    pad_front = (
        temp.new_zeros((temp.shape[0], 1)) - 1
    )  # -1 for the front padding
    # don't need pad_back because EOS is added at the end and never deleted
    # pad_back = temp.new_zeros((temp.shape[0], 1))
    # temp = torch.cat([pad_front, temp, pad_back], dim=1)  # Add a leading zero
    # temp.scatter_(
    #    1, xt_len.unsqueeze(1) + 1, x1_len.unsqueeze(1)
    # )  # Fill the last position with x1_len
    temp = torch.cat([pad_front, temp], dim=1)  # Add a leading -1

    gaps = temp[:, 1:] - temp[:, :-1] - 1
    gaps = torch.clamp(gaps, min=0)

    idx = torch.arange(gaps.size(1), device=xt.device).unsqueeze(
        0
    )  # shape [1, max_gap]
    # mask = idx <= xt_len.unsqueeze(1)
    mask = idx < xt_len.unsqueeze(1)
    gaps[~mask] = 0

    return x1, xt, st, gaps, mask


class FlexMDMLoss(LossFunction[FlexMDMBatch, FlexMDMLossDict]):
    def __init__(
        self,
        loss_on_padding: bool = False,
        loss_on_visible_tokens: bool = False,
        noise_schedule: FlexMDMNoiseSchedule = None,
        model: Optional[FlexMDMModel] = None,
        tokenizer: Optional[Tokenizer] = None,
        insert_loss_fn: str = "distribution", # choices: 'expectation', 'distribution'
    ):
        self.loss_on_padding = loss_on_padding
        self.loss_on_visible_tokens = loss_on_visible_tokens
        self.noise_schedule = noise_schedule
        self.model = model
        self.tokenizer = tokenizer
        self.insert_loss_fn = insert_loss_fn
        self.mask_token_id_tensor = None

    def configure(self, pl_module: Harness):
        self.mask_token_id_tensor = torch.tensor(  # type: ignore
            self.tokenizer.mask_token_id,
            dtype=torch.long,
            device=pl_module.device,
        )

    def __call__(
        self,
        batch: FlexMDMBatch,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> FlexMDMLossDict:
        loss_dict = self.loss_fn(
            batch, batch_idx, dataloader_idx, dataloader_name
        )
        return loss_dict

    def loss_fn(
        self,
        batch: FlexMDMBatch,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> FlexMDMLossDict:

        ids = batch["input_ids"]
        fixed = batch["fixed"]
        
        batch_size, max_length = ids.shape

        # TODO sample noise levels t, noise sequence
        t = sample_time(ids.shape[0], ids.device)
        token_weight = self.noise_schedule.unmasking_noise_schedule.rate_scale_factor(t)
        length_weight = self.noise_schedule.insertion_noise_schedule.rate_scale_factor(t)

        x1, xt, st, gaps, gaps_mask = get_noised_sequence(
            ids,
            batch_size,
            max_length,
            t,
            fixed,
            noise_schedule=cast(FlexMDMNoiseSchedule, self.noise_schedule),
            tokenizer=self.tokenizer,
        )

        attention_mask = (xt != self.tokenizer.pad_token_id).bool().to(xt.device)

        model = cast(FlexMDMModel, self.model)
        with torch.autocast(
            device_type="cuda",
            enabled=True,
            dtype=torch.float32,
        ):
            token_logits, length_logits =  model(xt, t, attention_mask)

        unmask_weight = token_weight[:, None].expand(-1, x1.shape[1])

        unmasked = torch.gather(x1, 1, st)

        scale_factor = x1.shape[0] * max_length

        mask_indices = (xt == self.tokenizer.mask_token_id)
        unmask_loss = unmask_weight[mask_indices] * F.cross_entropy(
            token_logits[mask_indices],
            unmasked[mask_indices],
            reduction="none",
        )
        unmask_loss = unmask_loss.sum() / scale_factor

        insert_weight = length_weight[:, None].expand(-1, x1.shape[1])

        match self.insert_loss_fn:
            case "expectation":                                
                expected_gaps = length_logits
                eps = 1e-6
                x_safe = torch.clamp(gaps[gaps_mask], min=eps)
                y_safe = torch.clamp(expected_gaps[gaps_mask], min=eps)
                bregman = y_safe - x_safe + x_safe * (torch.log(x_safe) - torch.log(y_safe))
                insertion_loss = insert_weight[gaps_mask] * bregman
                insertion_loss = insertion_loss.sum() / scale_factor
            case "distribution":
                #expected_gaps = (length_logits.softmax(-1) * torch.arange(0, max_length, device=length_logits.device).view(1, 1, -1)).sum(-1)
                insertion_loss = insert_weight[gaps_mask] * F.cross_entropy(length_logits[gaps_mask], gaps[gaps_mask])
                #eps = 1e-6
                #x_safe = torch.clamp(gaps[gaps_mask], min=eps)
                #y_safe = torch.clamp(expected_gaps[gaps_mask], min=eps)
                #bregman = y_safe - x_safe + x_safe * (torch.log(x_safe) - torch.log(y_safe))
                #insertion_loss = insert_weight[gaps_mask] * bregman
                insertion_loss = insertion_loss.sum() / scale_factor

        return {"loss": unmask_loss + insertion_loss, "unmask_loss": unmask_loss.detach(), "insertion_loss": insertion_loss.detach()}
