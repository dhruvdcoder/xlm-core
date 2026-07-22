"""LLaDA diffusion predictor.

LLaDA's reference sampler (``generate.py`` in https://github.com/ML-GSAI/LLaDA)
fills a fixed canvas of ``mask_token_id`` (126336) and iteratively commits the
most confident predictions per semi-autoregressive block ("low-confidence
remasking"). ``DreamPredictor`` already implements exactly this machinery, so
``LLaDAPredictor`` reuses it; the reference behavior corresponds to:

- ``confidence: top_prob``  == LLaDA's ``remasking='low_confidence'``
  (``confidence: null`` gives ``remasking='random'``)
- ``block_size``            == LLaDA's ``block_length`` (semi-AR blocks)
- ``max_steps``             == LLaDA's ``steps``
- ``max_new_tokens``        == LLaDA's ``gen_length``
- ``temperature: 0.0``      == LLaDA's greedy ``temperature=0.``

Unlike Dream, do NOT configure a ``logits_hook`` (Dream needs
``LogitsShiftBy1`` for its next-token alignment; LLaDA predicts in place).
Classifier-free guidance (``cfg_scale``) from the reference sampler is not
implemented.
"""

from dream.predictor_dream import DreamPredictor


class LLaDAPredictor(DreamPredictor):
    """Masked-diffusion predictor for LLaDA (see module docstring)."""

    pass
