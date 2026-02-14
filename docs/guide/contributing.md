# Adding a new language model architecture to XLM

For scaffolding a new external model, see [External Models](external-models.md). This document outlines the process for adding a new language model architecture to the XLM codebase. The code follows a modular design with four main components that work together to provide a complete language modeling solution. You need to implement all of them in order to add a new working model.

## Overview of Main Components

### 1. **LossFunction**
The `LossFunction` is responsible for computing the training loss during model training, validation and optionally test time.

**Key Responsibilities:**
- Compute loss between model predictions and ground truth targets
- Return a dictionary with "loss" key and any other additional values that you want to track. 

**Interface:**
```python
class LossFunction(Generic[T_in, T_out], Protocol):
    model: Any
    tokenizer: Tokenizer
    
    def loss_fn(self, batch: T_in, ...) -> T_out: ...
    def configure(self, pl_module: "Harness"): ...
        """Converts scalar to tensor such that loss_fn becomes compile friendly. If you don't want to compile you don't need to implement this.
        """
```

Examples: `xlm-models/arlm/loss_arlm.py`

### 2. **Predictor**
The `Predictor` handles generating output sequences from the model. 

**Key Responsibilities:**
- Run the model (typically in a loop) to produce a sequence of tokens.
- Convert generated token_ids to text.

**Interface:**
```python
class Predictor(Generic[T_in, T_out_pred], Protocol):
    tokenizer: Tokenizer
    noise_schedule: NoiseSchedule
    model: Any
    
    def predict(self, batch: T_in, ...) -> T_out_pred: ...
    def to_dict(self, batch: T_in, preds: T_out_pred, ...) -> List[Dict[str, Any]]: ...
```

Examples: `xlm-models/arlm/predictor_arlm.py`

### 3. **Collator**
The `Collator` prepares batches of data for training and inference. It handles data preprocessing, padding, and batching.

**Key Responsibilities:**
- It receives raw token_ids and converts them to a dict which is passed in as a batch to the model.
- Handle padding and truncation.

**Interface:**
```python
class Collator(Protocol):
    """For pre-training the model on language modeling."""
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]: ...

class Seq2SeqCollator(Collator):
    """For training the model on seq2seq tasks."""
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]: ...

class Seq2SeqCollatorPrediction(Collator):
    """For generating predictions for seq2seq tasks."""
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]: ...
```

Examples: `xlm-models/arlm/datamodule_arlm.py`

### 4. **Model**
The `Model` is the bare neural network architecture for your LM. It defines the forward pass and model parameters.

**Key Responsibilities:**
- Define the neural network architecture.
- Implement the forward pass.

**Interface:**
```python
class Model:
    def forward(self, input_ids: Tensor, attention_mask: Optional[Tensor] = None, ...) -> Tensor: ...
```

Examples: `xlm-models/arlm/model_arlm.py`

All these four components are designed to be aware of each other, and are only expected to run with each other for the same LM and not with any other LM. This is a key design choice that allows one to implement really esoteric models without worrying about how to abstract them such that their dataflow becomes compatible with other LMs.

## Step-by-Step Guide

See the [External Models](external-models.md) guide for the recommended workflow using `xlm-scaffold`. The scaffold creates the directory structure, type definitions, and config stubs. You then implement the four components (Model, Loss, Predictor, Collator) and register them in Hydra configuration.

Key config locations (paths may vary for external models in `xlm-models/`):

- `configs/lightning_train/collator/` – Collator configs
- `configs/lightning_train/datamodule/` – Datamodule configs
- `configs/lightning_train/model/` – Model (neural network) configs
- `configs/lightning_train/model_type/` – Loss, predictor, metrics
- `configs/lightning_train/experiment/` – Experiment configs

## Design

### Why is there so much nesting in the datamodule config?

The main component of the datamodule are the `dataset_managers`. Each `dataset_manager` will generate its own `dataloader` with its own `collator` and processing functions. This design allows:

1. Chaining arbitrary number of "eval" tasks/datasets during validation or testing
2. Injecting new eval tasks post-training

## Best Practices

- **Type Safety**: Use `jaxtyping` for tensor type annotations; define clear interfaces with `TypedDict` and `Protocol`
- **Modularity**: Keep components loosely coupled; use dependency injection through configuration
- **Testing**: Use debug mode configs (`debug=overfit`, `debug=small_data`) for quick testing
- **Documentation**: Document all public interfaces; include examples in docstrings

## Troubleshooting

1. **Hydra Errors**: If you see `Unable to find or instantiate abc.xyz.MyClass`, try importing manually: `python -c "from abc.xyz import MyClass"`.
2. **Unable to implement a component**: Check existing models in `xlm-models/` (arlm, mlm, ilm, mdlm) for reference.
