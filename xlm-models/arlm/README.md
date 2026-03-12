# ARLM - Auto-Regressive Language Model

ARLM (Auto-Regressive Language Model) is a transformer-based language model implementation for the XLM framework.

## Overview

ARLM implements standard causal language modeling where the model predicts the next token given the previous tokens. It features:

- **Rotary Position Embeddings**: Uses rotary embeddings for better position encoding
- **Causal Attention**: Standard left-to-right attention pattern
- **Flexible Architecture**: Configurable layers, heads, and dimensions
- **Seq2Seq Support**: Can be used for both language modeling and sequence-to-sequence tasks
