# Adding a new language model architecture to XLM

This document outlines the process for adding a new language model architecture to the XLM codebase. 
The code follows a modular design with four main components that work together to provide a complete language modeling solution. You need to implement all of them in order to add a new working model.

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

Examples:
1. `lm/arlm/loss_arlm.py`

### 2. **Predictor**
The `Predictor` handles geneating output sequences from the model. 

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
        """
        This is the main function that generates output sequences from the model.
        """
    def to_dict(self, batch: T_in, preds: T_out_pred, ...) -> List[Dict[str, Any]]: ...
        """
        Converts the output of the predict function (a batch) to a dictionary that can be logged to wandb or saved to a json file.
        """
```

Examples:
1. `lm/arlm/predictor_arlm.py`

### 3. **Collator**
The `Collator` prepares batches of data for training and inference. It handles data preprocessing, padding, and batching.

**Key Responsibilities:**
- It recieves raw token_ids and converts them to a dict which is passed in as a batch to the model.
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

Examples:
1. `lm/arlm/datamodule_arlm.py`

### 4. **Model**
The `Model` is the bare neural network architecture for your lm. 
It defines the forward pass and model parameters.

**Key Responsibilities:**
- Define the neural network architecture.
- Implement the forward pass.

**Interface:**
```python
class Model:
    def forward(self, input_ids: Tensor, attention_mask: Optional[Tensor] = None, ...) -> Tensor: ...
```

Examples:
1. `lm/arlm/model_arlm.py`


All these four components are designed to be aware of each other, and are only expected to run with each other for the same lm and not with any other lm. For example, ARLMLoss and ARLMPredictor expect the batches in certain from and ARLMCollator froms the batches in that form. But ARLMCollator's batches won't work with ILMLoss or ILMPredictor. This is a key design choice that allows one to implement really esoteric models without worrying about how to abstract them such that their dataflow becomes compatible with other LMs. This may result in some code duplilcation but it is a small price to pay for fast development and complete freedom.


## Step-by-Step Guide for Adding a New Language Model

### Step 1: Create the Model Directory Structure

Create a new directory under `src/xlm/lm/` for your model:

```bash
mkdir src/xlm/lm/my_lm/
```

### Step 2: Define Type Definitions

Create `types_my_lm.py` to define the data structures:

```python
from typing import Optional, TypedDict
from jaxtyping import Float, Integer, Bool
from torch import Tensor as TT

class MyLMBatch(TypedDict):
    """Input batch for your model."""
    input_ids: Integer[TT, " batch seq_len"]
    attention_mask: Integer[TT, " batch seq_len"]
    # Add other fields as needed

class MyLMLossDict(TypedDict):
    """Output of the loss function."""
    loss: Float[TT, ""]
    batch_loss: Float[TT, " batch"]
    # Add other metrics as needed

class MyLMPredictionDict(TypedDict):
    """Output of the predictor."""
    text: List[str]
    ids: Integer[TT, " batch seq_len"]
    # Add other fields as needed
```

### Step 3: Implement the Collator

Create `datamodule_my_lm.py`:

```python
from typing import List, Dict, Any
from xlm.datamodule import Collator

class MyLMCollator(Collator):
    """Collator for your model."""
    
    def __init__(self, tokenizer, **kwargs):
        self.tokenizer = tokenizer
        # Initialize other parameters
    
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Implement your collation logic
        pass
```

### Step 4: Implement the Model

Create `model_my_lm.py`:

```python
import torch
import torch.nn as nn
from typing import Optional
from jaxtyping import Integer
from torch import Tensor as TT

class MyLMModel(nn.Module):
    """Your model implementation."""
    
    def __init__(self, config):
        super().__init__()
        # Initialize your model components
        pass
    
    def forward(
        self,
        input_ids: Integer[TT, " batch seq_len"],
        attention_mask: Optional[Integer[TT, " batch seq_len"]] = None,
        **kwargs
    ):
        # Implement your forward pass
        pass
```

### Step 5: Implement the Loss Function

Create `loss_my_lm.py`:

```python
from typing import Optional
import torch
from xlm.harness import LossFunction, Harness
from .types_my_lm import MyLMBatch, MyLMLossDict

class MyLMLoss(LossFunction[MyLMBatch, MyLMLossDict]):
    """Loss function for your model."""
    
    def __init__(self, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
    
    def loss_fn(self, batch: MyLMBatch, ...) -> MyLMLossDict:
        # Implement your loss computation
        pass
    
    def configure(self, pl_module: Harness):
        # Configure the loss function if needed
        pass
```

### Step 6: Implement the Predictor

Create `predictor_my_lm.py`:

```python
from typing import List, Dict, Any
from xlm.harness import Predictor
from .types_my_lm import MyLMBatch, MyLMPredictionDict

class MyLMPredictor(Predictor[MyLMBatch, MyLMPredictionDict]):
    """Predictor for your model."""
    
    def __init__(self, model=None, tokenizer=None, noise_schedule=None):
        self.model = model
        self.tokenizer = tokenizer
        self.noise_schedule = noise_schedule
    
    def predict(self, batch: MyLMBatch, ...) -> MyLMPredictionDict:
        # Implement your prediction logic
        pass
    
    def to_dict(self, batch: MyLMBatch, preds: MyLMPredictionDict, ...) -> List[Dict[str, Any]]:
        # Convert predictions to dictionary format
        pass
```

### Step 7: Implement Metrics (Optional)

If your model requires custom metrics, or custom update functions for existing metrics (more common) create `metrics_my_lm.py`. See `xlm/src/xlm/lm/arlm/metrics_arlm.py` for an example.


### Step 8: Register Components in Configuration

The necessary **logical** components for a typical new model can be described using the following tree. Through configs we built this logical three bottom up--meaning the higher nodes have a chance to override the config values set by the lower nodes.
```
experiment
├── data
│   └── datamodule
│       └── collators
└── model components
    ├── neural network
    ├── loss
    └── predictor
```
Unfortunately, the actual config files don't correspond exactly to the logical tree described above, but they will come pretty close.

First start with the data section.

#### Collator Configurations

A model can have different collators for three different batch formation tasks:

1. **Base (uncoditional) Training**: The dataset sends raw lists of tokenized ids as examples. Each example is a single tokenized sentence. In this collator, this sequence is converted to a batch for the model for training. For example, for MLM, this would mean adding some masks, creating target ids and padding.
2. **Seq2Seq Training**: The dataset sends two lists of tokenized ids per example--one for prompt/prefix and one for target/suffix. Typically, this collator will concat the prefix and suffix put any necessary special tokens in between, etc.

3. **Seq2Seq Prediction**: The collator either single list of tokenized ids per example corresponding to the prompt (true prediction scenario), or two lists of tokenized ids per example corresponding to the prompt and target (prediction eval scenario). So how is this different from the seq2seq training collator? Typically, this collator does not put the prefix and suffix together, instead it will only send out the prefix as "input_ids", the suffix, if present (prediction eval scenario) is kept separate as "target_ids", to be used by the corresponding metric.




Create `configs/lightning_train/collator/default_my_lm.yaml`:

```yaml
_target_: xlm.lm.my_lm.datamodule_my_lm.DefaultMyLMCollator
block_size: ${block_size}
tokenizer: ${global_components:tokenizer}
noise_schedule: ${global_components:noise_schedule}
```

We will now use the collators in the datamodule config.

#### Datamodule Configuration
We need `<dataset>_<model_type>.yaml` for each dataset.
Create `configs/lightning_train/datamodule/<dataset>_my_lm.yaml` where you will combine all the collators and other components for your model type for a specific dataset.

```yaml
# @package _global_
defaults:
  - default
  - /collator@datamodule.dataset_managers.train.lm.collator: default_my_lm
  - /collator@datamodule.dataset_managers.val.lm.collator: default_my_lm

datamodule:
  print_batch_fn: xlm.lm.my_lm.datamodule_my_lm.print_batch_my_lm

tags:
  dataset: my_lm
```


#### Model Configuration
Responsibilities:
  1. Define default parameters for the model (neural network architecture only)

Create `configs/lightning_train/model/my_lm.yaml`:

```yaml
# @package _global_

model:
  _target_: xlm.lm.my_lm.model_my_lm.MyLMModel
  num_embeddings: ${tokenizer:full_vocab_size}
  d_model: 768
  num_layers: 12
  nhead: 12
  padding_idx: ${tokenizer:pad_token_id}
  dim_feedforward: ${eval:${.d_model}*4}
  dropout: 0.1
  activation: "relu"
  layer_norm_eps: 1e-5
  max_length: ${predictor.max_length}
  force_flash_attn: false

tags:
  model: my_lm_small
```

#### Model Type Configuration

Responsibilities:
  1. Set-up the reported and diagnostic metrics, which will be present in all tasks/datasets, for the model type. Leave task/dataset specific metrics for the final experiment config.
  2. Provide loss function defaults.
  3. Provide predictor defaults.
  4. (Advanced) If you have you own harness, you can also set it up here under `lightning_module` section.

Create `configs/lightning_train/model_type/my_lm.yaml`:

```yaml
# @package _global_

defaults:
  - /metrics@reported_metrics.train.lm.accumulated_loss: accumulated_loss
  - /metrics@reported_metrics.val.lm.accumulated_loss: accumulated_loss
  - /metrics@reported_metrics.test.lm.accumulated_loss: accumulated_loss

lightning_module:
  _target_: xlm.harness.Harness

loss:
  _target_: xlm.lm.my_lm.loss_my_lm.MyLMLoss

predictor:
  _target_: xlm.lm.my_lm.predictor_my_lm.MyLMPredictor
  tokenizer: ${lightning_module:tokenizer}
  noise_schedule: ${lightning_module:noise_schedule}
  max_steps: ${block_size}
  max_length: ${eval:${block_size}+${oc.select:input_block_size,0}}
  sampling_method: sample_top_p
  p: 0.5

reported_metrics:
  train:
    lm:
      accumulated_loss:
        prefix: train/lm
        update_fn: xlm.lm.my_lm.metrics_my_lm.mean_metric_update_fn
  val:
    lm:
      accumulated_loss:
        prefix: val/lm
        update_fn: xlm.lm.my_lm.metrics_my_lm.mean_metric_update_fn
  test:
    lm:
      accumulated_loss:
        prefix: test/lm
        update_fn: xlm.lm.my_lm.metrics_my_lm.mean_metric_update_fn

tags:
  model_type: my_lm
```

#### Experiment Configuration
Create `configs/lightning_train/experiment/my_lm.yaml`:

```yaml
# @package _global_
defaults:
  - override /datamodule: my_lm
  - override /noise_schedule: dummy
  - override /model_type: my_lm
  - override /model: my_lm

per_device_batch_size: 64
global_batch_size: 512
block_size: 128

datamodule:
  print_batch_fn: xlm.lm.my_lm.datamodule_my_lm.print_batch_my_lm

trainer:
  max_steps: 1000_000
  val_check_interval: 50000
  num_sanity_val_steps: 3
```

### Step 9: Integration with Harness

The `Harness` class automatically integrates your components:

1. **Model Integration**: The harness instantiates your model from config
2. **Loss Integration**: Your loss function is automatically configured with the model and tokenizer
3. **Predictor Integration**: Your predictor is configured with model, tokenizer, and noise schedule
4. **Data Integration**: Your collator is used by the datamodule

### Step 10: Testing Your Implementation

Instead of writing formal tests, the codebase uses debug mode for testing. Create a debug configuration and run your model:

#### Run Debug Mode
```bash
# Debug overfit run
python src/xlm/commands/lightning_main.py "job_type=train" "job_name=my_lm_debug" "experiment=my_lm" "debug=overfit"

# Debug with small data
python src/xlm/commands/lightning_main.py "job_type=train" "job_name=my_lm_debug" "experiment=my_lm" "debug=small_data"
```

## Design

### Why is there so much nesting in the datamodule config?

The main component of the datamodule are the `dataset_managers`. Each `dataset_manager` will generate its own `dataloader` with its own `collator` and processing functions. This design choice is the crux of the whole codebase. This allows the following:

1. Chain arbirary number of "eval" tasks/datasets to be run during validation or testing (validation and testing can even have different eval tasks!)
2. Inject new eval tasks post-training.

TODO: We don't need stages to be present inside each dataset manager. We can rely on the outer keys to determine the stages.

```yaml
datamodule:
  _target_: xlm.datamodule.TextDataModule
  ...
  dataset_managers:
    train:
      lm:
        _target_: xlm.datamodule.DatasetManager
        collator:
          _target_: xlm.lm.ilm.datamodule_ilm.ILMSeq2SeqCollator
          ...
        full_name: dhruveshpatel/star-hard/train
        preprocess_function: xlm.tasks.star.preprocess_fn
        on_the_fly_processor: xlm.datamodule.token_ids_to_input_ids_and_prompt_ids
        on_the_fly_group_processor: null
        stages:
        - fit
        dataloader_kwargs:
          batch_size: 64
          ...
    val:
      lm:
        _target_: xlm.datamodule.DatasetManager
        collator:
          _target_: xlm.lm.ilm.datamodule_ilm.ILMSeq2SeqCollator
          ...
        full_name: dhruveshpatel/star-hard/validation
        preprocess_function: xlm.tasks.star.preprocess_fn
        on_the_fly_processor: xlm.datamodule.token_ids_to_input_ids_and_prompt_ids
        on_the_fly_group_processor: null
        stages:
        - fit
        - validate
        ...
        dataloader_kwargs:
          ...
      prediction:
        collator:
          _target_: xlm.lm.ilm.datamodule_ilm.ILMSeq2SeqPredCollator
          input_block_size: 116
          block_size: 28
          tokenizer: ${global_components:tokenizer}
          noise_schedule: ${global_components:noise_schedule}
        _target_: xlm.datamodule.DatasetManager
        full_name: dhruveshpatel/star-hard/validation
        full_name_debug: dhruveshpatel/star-hard/train
        preprocess_function: xlm.tasks.star.preprocess_fn
        on_the_fly_processor: xlm.datamodule.token_ids_to_input_ids_and_prompt_ids
        on_the_fly_group_processor: null
        columns_to_remove:
        - edge_list
        - source
        - goal
        - path
        stages:
        - fit
        - validate
        iterable_dataset_shards: null
        shuffle_buffer_size: null
        shuffle_seed: 42
        split_by_node: true
        dataloader_kwargs:
          batch_size: 64
          num_workers: 5
          shuffle: null
          pin_memory: true
          persistent_workers: false
          prefetch_factor: 5
          drop_last: true
        model_name: null
    test:
      lm:
        collator:
          _target_: xlm.lm.ilm.datamodule_ilm.ILMSeq2SeqCollator
          input_block_size: 116
          block_size: 28
          tokenizer: ${global_components:tokenizer}
          noise_schedule: ${global_components:noise_schedule}
        _target_: xlm.datamodule.DatasetManager
        full_name: dhruveshpatel/star-hard/test
        full_name_debug: dhruveshpatel/star-hard/train
        preprocess_function: xlm.tasks.star.preprocess_fn
        on_the_fly_processor: xlm.datamodule.token_ids_to_input_ids_and_prompt_ids
        on_the_fly_group_processor: null
        columns_to_remove:
        - edge_list
        - source
        - goal
        - path
        stages:
        - test
        iterable_dataset_shards: null
        shuffle_buffer_size: null
        shuffle_seed: 42
        split_by_node: true
        dataloader_kwargs:
          batch_size: 64
          num_workers: 5
          shuffle: null
          pin_memory: true
          persistent_workers: false
          prefetch_factor: 5
          drop_last: true
        model_name: null
      prediction:
        collator:
          _target_: xlm.lm.ilm.datamodule_ilm.ILMSeq2SeqPredCollator
          input_block_size: 116
          block_size: 28
          tokenizer: ${global_components:tokenizer}
          noise_schedule: ${global_components:noise_schedule}
        _target_: xlm.datamodule.DatasetManager
        full_name: dhruveshpatel/star-hard/test
        full_name_debug: dhruveshpatel/star-hard/train
        preprocess_function: xlm.tasks.star.preprocess_fn
        on_the_fly_processor: xlm.datamodule.token_ids_to_input_ids_and_prompt_ids
        on_the_fly_group_processor: null
        columns_to_remove:
        - edge_list
        - source
        - goal
        - path
        stages:
        - test
        iterable_dataset_shards: null
        shuffle_buffer_size: null
        shuffle_seed: 42
        split_by_node: true
        dataloader_kwargs:
          batch_size: 64
          num_workers: 5
          shuffle: null
          pin_memory: true
          persistent_workers: false
          prefetch_factor: 5
          drop_last: true
        model_name: null
  noise_schedule: ${global_components:noise_schedule}
  rewrite_manual_cache: false
  block_size: 28
  global_batch_size: 64
  num_dataset_workers: 5
  print_batch_fn: xlm.lm.ilm.datamodule_ilm.print_batch_ilm
reported_metrics:
  train:
    lm:
      accumulated_loss:
        _target_: xlm.metrics.MetricWrapper
        name: accumulated_loss
        metric:
          _target_: torchmetrics.MeanMetric
        prefix: train/lm
        update_fn: xlm.lm.ilm.metrics_ilm.mean_metric_update_fn
  val:
    lm:
      accumulated_loss:
        _target_: xlm.metrics.MetricWrapper
        name: accumulated_loss
        metric:
          _target_: torchmetrics.MeanMetric
        prefix: val/lm
        update_fn: xlm.lm.ilm.metrics_ilm.mean_metric_update_fn
    prediction:
      exact_match:
        _target_: xlm.metrics.MetricWrapper
        name: exact_match
        metric:
          _target_: xlm.metrics.ExactMatch
        prefix: val/prediction
        update_fn: xlm.metrics.seq2seq_exact_match_update_fn
      token_accuracy:
        _target_: xlm.metrics.MetricWrapper
        name: token_accuracy
        metric:
          _target_: xlm.metrics.TokenAccuracy
        prefix: val/prediction
        update_fn: xlm.metrics.seq2seq_token_accuracy_update_fn
  test:
    lm:
      accumulated_loss:
        _target_: xlm.metrics.MetricWrapper
        name: accumulated_loss
        metric:
          _target_: torchmetrics.MeanMetric
        prefix: test/lm
        update_fn: xlm.lm.ilm.metrics_ilm.mean_metric_update_fn
    prediction:
      exact_match:
        _target_: xlm.metrics.MetricWrapper
        name: exact_match
        metric:
          _target_: xlm.metrics.ExactMatch
        prefix: test/prediction
        update_fn: xlm.metrics.seq2seq_exact_match_update_fn
      token_accuracy:
        _target_: xlm.metrics.MetricWrapper
        name: token_accuracy
        metric:
          _target_: xlm.metrics.TokenAccuracy
        prefix: test/prediction
        update_fn: xlm.metrics.seq2seq_token_accuracy_update_fn
```


## Best Practices

### 1. **Type Safety**
- Use `jaxtyping` for tensor type annotations
- Define clear interfaces with `TypedDict` and `Protocol`
- Use generic types for better type checking

### 2. **Modularity**
- Keep components loosely coupled
- Use dependency injection through configuration
- Follow the single responsibility principle

### 3. **Error Handling**
- Validate inputs in your components
- Provide clear error messages
- Handle edge cases gracefully

### 4. **Documentation**
- Document all public interfaces
- Include examples in docstrings
- Update this guide with any new patterns

### 5. **Testing**
- Use debug mode configs for quick testing
- Write many assert statements. We skip assert in final training using `python -O` flag so there is no need to worry about performance.

## Example: ARLM Implementation

The ARLM (Auto-Regressive Language Model) implementation serves as a good example:

- **Model**: `ARLMModel` - Standard transformer for causal language modeling in `lm/arlm/model_arlm.py` and config in `configs/lightning_train/model/arlm.yaml`
- **Loss**: `ARLMLoss` - Cross-entropy loss with proper masking in `lm/arlm/loss_arlm.py` and config in `configs/lightning_train/model_type/arlm.yaml`
- **Predictor**: `ARLMPredictor` - Autoregressive text generation in `lm/arlm/predictor_arlm.py` and config in `configs/lightning_train/model_type/arlm.yaml`
- **Collators**: 
    - `DefaultARLMCollator` - Handles padding and target shifting in `lm/arlm/datamodule_arlm.py` and config in `configs/lightning_train/collator/default_arlm.yaml`
    - `Seq2SeqCollator` - For seq2seq tasks in `lm/arlm/datamodule_arlm.py` and config in `configs/lightning_train/collator/seq2seq_arlm.yaml`
    - `Seq2SeqCollatorPrediction` - For prediction in seq2seq tasks when there is an exact target to match in `lm/arlm/datamodule_arlm.py` and config in `configs/lightning_train/collator/seq2seq_pred_arlm.yaml`

- **List of files touched**:
  - `lm/arlm/model_arlm.py`
  - `lm/arlm/loss_arlm.py`
  - `lm/arlm/predictor_arlm.py`
  - `lm/arlm/datamodule_arlm.py`
  - `configs/lightning_train/model/rotary_transformer_small_arlm.yaml`
  - `configs/lightning_train/model_type/arlm_seq2seq.yaml`
  - `configs/lightning_train/collator/default_arlm.yaml`
  - `configs/lightning_train/collator/seq2seq_arlm.yaml`
  - `configs/lightning_train/collator/seq2seq_pred_arlm.yaml`
  - `configs/lightning_train/datamodule/star_arlm.yaml`
  - `configs/lightning_train/datamodule/star_easy_arlm.yaml`
  - `configs/lightning_train/experiment/star_easy_arlm.yaml`

## FAQs

## How do I train an existing model architecture on a new dataset?
TODO

## Troubleshooting

### Common Issues and Solutions

1. **Hydra Errors**: 
  - Error message like `Unable to find a package ...` by hydra. 
    - See the name of the package in the error message, for example `Unable to find or instantiate abc.xyz.MyClass` then first try to import is manually in the python interpreter `python -c "from abc.xyz import MyClass"`.
  - Reach out on slack if you cannot fix a config error in 10 minutes.

2. **Unable to determine how to implement a component of a new lm**:
  - Check existing lms for reference
  - Reach out on slack.


# Adding a new entrypoint script to the cli

TODO