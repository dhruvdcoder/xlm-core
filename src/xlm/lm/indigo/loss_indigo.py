import torch
import torch.nn.functional as F
from typing import Any, Dict, Optional
from xlm.harness import Harness, LossFunction
from xlm.utils.rank_zero import RankedLogger
from .types_indigo import IndigoBatch, IndigoLossDict

logger = RankedLogger(__name__, rank_zero_only=True)


class IndigoLoss(LossFunction[IndigoBatch, IndigoLossDict]):
    """
    Indigo Loss = word prediction CE + position prediction CE

    Each term is computed per step and masked. We report total loss, 
    individual components, perplexity, and accuracies.
    """

    def __init__(self, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer  # type: ignore

    def configure(self, pl_module: Harness):
        pass

    def __call__(
        self,
        batch: Dict[str, Any],
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self.loss_fn(batch)

    def loss_fn(self, batch: IndigoBatch) -> IndigoLossDict:
        assert self.model is not None, "model must be attached before training"

        # Inputs
        input_ids = batch["input_ids"]                      # (bs, seq_len)
        attention_mask = batch["attention_mask"]            # (bs, seq_len)
        word_labels = batch["word_labels"]                  # (bs, steps)
        word_labels_mask = batch["word_labels_mask"]        # (bs, steps)
        pointer_labels = batch["pointer_labels"]            # (bs, steps)
        pointer_labels_mask = batch["pointer_labels_mask"]  # (bs, steps)
        rel_matrix = batch["relative_matrix"]               # (bs, L, L)
        abs_pos = batch["absolute_positions"]               # (bs, steps)

        bs, steps = word_labels.shape
        vocab_size = self.model.output_layer.linear.out_features
        max_slots = rel_matrix.size(1) * 2  # Maximum pointer slots

        # Forward pass
        hidden_states, vocab_logits = self.model(
            x_t=input_ids,
            rel_matrix=rel_matrix,
            attention_mask=attention_mask,
        )  # vocab_logits: (bs, seq_len, V)

        gather_idx = abs_pos.unsqueeze(-1).expand(-1, -1, vocab_size)
        step_token_logits = torch.gather(vocab_logits, 1, gather_idx)  # (bs, steps, V)

        # --- Word (token) cross-entropy ---
        word_ce = F.cross_entropy(
            step_token_logits.view(-1, vocab_size),
            word_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view(bs, steps)
        word_ce = word_ce * word_labels_mask.float()
        word_loss = word_ce.sum() / word_labels_mask.float().sum()

        # --- Word accuracy ---
        with torch.no_grad():
            pred_tokens = step_token_logits.argmax(-1)  # (bs, steps)
            word_correct = (pred_tokens == word_labels) & word_labels_mask
            word_acc = word_correct.sum().float() / word_labels_mask.sum()

        # --- Pointer logits ---
        gt_word_embeds = self.model.embed_tokens(word_labels)  # (bs, steps, d_model)
        gather_h_idx = abs_pos.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1))
        step_hidden = torch.gather(hidden_states, 1, gather_h_idx)  # (bs, steps, d_model)

        pointer_logits_full = self.model.get_position(
            H=step_hidden,
            embed_matrix=gt_word_embeds,
        )  # (bs, steps, 2*seq_len)
        pointer_logits = pointer_logits_full[:, :, :max_slots]  # clip

        # --- Pointer (position) cross-entropy ---
        pointer_ce = F.cross_entropy(
            pointer_logits.view(-1, max_slots),
            pointer_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view(bs, steps)
        pointer_ce = pointer_ce * pointer_labels_mask.float()
        position_loss = pointer_ce.sum() / pointer_labels_mask.float().sum()

        # --- Pointer accuracy ---
        with torch.no_grad():
            pred_slots = pointer_logits.argmax(-1)
            pointer_correct = (pred_slots == pointer_labels) & pointer_labels_mask
            pointer_acc = pointer_correct.sum().float() / pointer_labels_mask.sum()

        # --- Aggregate ---
        total_loss = word_loss + position_loss
        batch_loss = (word_ce + pointer_ce).detach()
        ppl = torch.exp(word_loss)

        return {
            "loss": total_loss,
            "batch_loss": batch_loss,
            "word_loss": word_loss.detach(),
            "position_loss": position_loss.detach(),
            "word_acc": word_acc.detach(),
            "pointer_acc": pointer_acc.detach(),
            "ppl": ppl.detach(),
        }
