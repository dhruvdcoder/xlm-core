from typing import Any, Dict, List, Optional, Tuple, cast, Literal, Callable
from itertools import cycle
from functools import partial
import torch
from jaxtyping import Bool, Integer
from xlm import flags
from xlm.datamodule import Tokenizer
from torch import Tensor as TT
from xlm.harness import Predictor
from xlm.noise import NoiseSchedule
from .types_indigo import IndigoBatch, IndigoPredictionDict

import time


###############################################################
# region: Predictors

#import functions from utils
from xlm.lm.indigo.utils import (
    get_absolute_position_matrix,
    get_tertiary_relative_position_matrix,
    get_left_pointer_posisiotn,
    get_right_pointer_position
)


class IndigoPredictor(
    torch.nn.Module,
    Predictor[IndigoBatch, IndigoPredictionDict],
):

    def __init__(
        self,
        max_steps: int,
        max_length: int,
        tokenizer: Optional[Tokenizer] = None,
        noise_schedule: Optional[NoiseSchedule] = None,
        model, 
    ):
        """Constructor for IndigoPredictor."""
        if tokenizer is None:
            raise ValueError("tokenizer is required")
        super().__init__()
        self.tokenizer = tokenizer
        self.max_steps = max_steps
        self.max_length = max_length
        self.noise_schedule = noise_schedule
        
        self.register_buffer("_devref", torch.tensor(0), persistent=False)

        self.pad_id: int = getattr(tokenizer, "pad_token_id", 0)
        self.bos_id: int = getattr(
            tokenizer, "bos_token_id",
            getattr(tokenizer, "cls_token_id", self.pad_id)
        )
        self.eos_id: int = getattr(
            tokenizer, "eos_token_id",
            getattr(tokenizer, "sep_token_id", self.pad_id)
        )
        self.eod_id: Optional[int] = getattr(tokenizer, "eod_token_id", None)
        
        self.model = model

    @torch._dynamo.disable()
    def predict(
        self,
        batch: Dict[str, Any],  # type: ignore
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        # TODO (URV): Implement the predictor.
        device = self._devref.device
        
        t0 = time.time()
        
        if isinstance(batch.get('input_ids', None), torch.Tensor):
            x = batch['input_ids'].to(device=device, dtype=torch.long)
            if isinstance(batch.get('attention_mask', None), torch.Tensor):
                attention_mask = batch['attention_mask'].to(device=device, dtype=torch.bool)
            else:
                attention_mask = ( x != self.pad_id)
                
            need_bos = (x.size(1) == 0) or x[:, 0].ne(self.bos_id).any()
            if need_bos:
                x = torch.cat(
                    [torch.full((x.size(0), 1), self.bos_id, device=device, dtype=torch.long), x],
                    dim=1,
                )
                attention_mask = torch.cat(
                    [torch.ones((attention_mask.size(0), 1), device=device, dtype=torch.bool), attention_mask],
                    dim=1,
                )
        else:
            B = 1
            for v in batch.values():
                if isinstance(v, torch.Tensor) and v.dim() > 0:
                    B = v.size(0)
                    break
            x = torch.tensor([[self.bos_id, self.eos_id]] * B, device=device, dtype=torch.long)
            attention_mask = torch.ones((B, 2), device=device, dtype=torch.bool)
            
        if x.size(1) > self.max_length:
            x = x[:, : self.max_length]
            attention_mask = attention_mask[:, : self.max_length]
            
        positions = torch.arange(x.size(1), device = device, dtype=torch.long).unsqueeze(0).expand(s.zie(0), -1)
        
        state: Dict[str, TT] = {
            'x': x,
            'positions': positions,
            'attention_mask': attention_mask.bool(),
            'finished': torch.zeros(s.size(0), dtype=torch.bool, device=device),
        }
        
        step = 0
        while not self.stop(state, step):
            state = self.predict_step(state)
            step += 1
            
        texts, texts_with, final_ids, final_mask, final_pos = self.decode(state)
        
        return {
            'text': texts,
            'text_w_token': texts_with,
            'ids': final_ids,
            'attention_mask': final_mask,
            'positions': final_pos,
            'loss': None,
            'time_taken': [time.time() - t0] * len(texts),
        }
        
    @torch.no_grad()
    def predict_step(self, state: Dict[str, TT]) -> Dict[str, TT]:
        """
        build R from absolute position, forward model, predict next token, choose pointer, compute z,
        update
        """
        x = state['x']
        position = state['positions']
        amask = state['attention_mask']
        finished = state['finished']
        B, L = x.shape
        device = x.device
        
        R = get_tertiary_relative_position_matrix(positions)
        M = get_absolute_position_matrix(R)
        H, vocab_logits = self._call_model(x, R, amask)
        
        next_token_logits = vocab_logits[:, -1, :]
        for tok in (self.bos_id, self.pad_id):
            if 0 <= tok < next_token_logits.size(-1):
                next_token_logits[:, tok] = -torch.inf
        next_token = torch.argmax(next_token_logits, dim = - 1)
        
        model = self.model
        q = torch.matmul(h_t, model.E.weight.T)
        w_y = model.embed_tokens(next_tok)
        left_keys  = torch.matmul(H, model.C.weight.T)
        right_keys = torch.matmul(H, model.D.weight.T)

        keys = torch.cat([left_keys, right_keys], dim = 1)
        ptr_logits = torch.einsum('bd, bkd->bk', q, keys)
        
        def _unique_index_per_row(tok_id: int) -> Optional[TT]:
            where = (x == tok_id).nonzero(as_tuple=False)
            if where.numel() == 0:
                return None
            counts = torch.bincount(where[:0], minlenght = B)
            if not torch.all(counts == 1):
                return None
            idx = torch.empty(B, dtype=torch.long, device=device)
            idx[where[:, 0]] = where{:, 1}
            return idx
        
        bos_seq_idx = _unique_index_per_row(self.bos_id)
        eos_seq_idx = _unique_index_per_row(self.eos_id)
        arangeB = torch.arange(B, device=device)
        if bos_seq_idx is not None:
            bos_left_slot = bos_seq_idx * 2 # left of bos
            ptr_logits[arangeB, bos_left_slot] = -torch.inf
        if eos_seq_idx is not None:
            eos_right_slot = eos_seq_idx * 2 + 1
            ptr_logits[arangeB, eos_right_slot] = -torch.inf 
            
        slot = torch.argmax(ptr_logits, dim=-1)
        anchor_idx = slot // 2
        side_right = (slot % 2 == 1) # B false = left, true = right
        
        try:
            left_abs = get_left_pointer_posisiotn(M)
            right_abs = get_right_pointer_position(M)
            z_anchor_left = left_abs.gather(1, anchor_idx.unsqueeze(1)).squeeze(1)
            a_anchor_right = right_abs.gather(1, anchor_idx.unsqueeze(1)).squeeze(1)
            insert_abs = torch.where(side_right, z_anchor_right, z_anchor_left)
        except Exception :
            anchor_abs = positions.gather(1, anchor_idx.unsqueeze(1)).squeeze(1)
            insert_abs = anchor_abs + side_right.long()
            
        can_grow = (~finished) & (amask.sum(dim=1) < self.max_length) # [B]
        appended = torch.where(can_grow, next_token, torch.full_like(next_token, self.pad_id))
        x_new = torch.cat([x, appended.unsqueeze(1)], dim=1) # [B, L+1]
        amask_new = torch.cat([amask, can_grow.unsqueeze(1)], dim=1) # [B, L+1]

        # shift existing abs positions >= insert_abs by +1, then append z_new
        shift_mask = positions >= insert_abs.unsqueeze(1) 
        positions_shifted = positions + shift_mask.to(positions.dtype)
        positions_new = torch.cat([positions_shifted, insert_abs.unsqueeze(1)], dim=1) # [B, L+1]

        if self.eod_id is not None:
            ended_now = can_grow & (next_token == self.eod_id)
        else:
            ended_now = can_grow & (next_token == self.eos_id)
        finished_new = finished | ended_now

        return {
            "x": x_new,
            "positions": positions_new,
            "attention_mask": amask_new,
            "finished": finished_new,
        }   

    def stop(self, state: Dict[str, TT], step: int) -> bool:
        if step >= self.max_steps:
            return True
        if torch.all(state["finished"]):
            return True
        if torch.all(state["attention_mask"].sum(dim=1) >= self.max_length):
            return True
        return False

    def decode(
        self, state: Dict[str, TT]
    ) -> Tuple[List[str], List[str], Integer[TT, "B L"], Bool[TT, "B L"], Integer[TT, "B L"]]:
        """
        Sort tokens by absolute positions, strip specials, and decode.
        Returns (clean text, text with specials, ids, mask, positions) in sorted order.
        """
        x = state["x"]                  
        pos = state["positions"]        
        amask = state["attention_mask"]  

        # reorder by absolute positions
        sorted_pos, sort_idx = torch.sort(pos, dim=1)      
        x_sorted = torch.gather(x, 1, sort_idx)            
        mask_sorted = torch.gather(amask, 1, sort_idx)     

        # keep = mask & not special
        keep = mask_sorted.clone()
        for tok in (self.pad_id, self.bos_id, self.eos_id):
            keep &= (x_sorted != tok)

        cleaned_ids = torch.where(keep, x_sorted, torch.full_like(x_sorted, self.pad_id))

        if hasattr(self.tokenizer, "batch_decode"):
            out_with = self.tokenizer.batch_decode(cleaned_ids.tolist(), skip_special_tokens=False)
            out = self.tokenizer.batch_decode(cleaned_ids.tolist(), skip_special_tokens=True)
        else:
            def _to_txt(ids_row, keep_row):
                return " ".join(str(t) for t, k in zip(ids_row, keep_row) if k)
            out_with = [_to_txt(r, k.tolist()) for r, k in zip(cleaned_ids.tolist(), keep)]
            out = out_with

        return out, out_with, cleaned_ids, mask_sorted, sorted_pos

    def _resolve_model(self):
        return self.model

    def _call_model(
        self,
        x_t: Integer[TT, "B L"],
        rel_matrix: Integer[TT, "B L L"],
        attention_mask: Optional[Bool[TT, "B L"]] = None,
    ) -> Tuple[TT, TT]:
        """
        Calls IndigoModel.forward(x_t, rel_matrix, attention_mask) -> (H, logits).
        """
        H, logits = self.model(x_t, rel_matrix, attention_mask)
        if isinstance((H, logits), tuple) and H is not None and logits is not None:
            return H, logits
        raise RuntimeError("IndigoModel must return (hidden_states, vocab_logits).")


# endregion: Predictors
###############################################################
