# Adding temporary diagnostic metrics

Sometimes you want to quickly log a value from your loss function for debugging purposes without setting up a full `MetricWrapper`. You can do this by adding the value to the dictionary returned by your loss function and configuring `log_strs` in `Harness`.

### 1. Update your loss function

In your `loss_fn` (e.g., in `FlexMDMLoss.loss_fn`), add the desired metric to the returned dictionary. Make sure to `.detach()` it to avoid memory leaks.

```python:flexmdm_correction/loss_flexmdm_correction.py
    def loss_fn(self, batch, ...):
        # ... your loss computation ...
        
        loss = ...
        my_debug_metric = ...

        return {
            "loss": loss,
            "unmask_loss": unmask_loss.detach(),
            "insertion_loss": insertion_loss.detach(),
            "my_metric": my_debug_metric.detach(),  # Add your new metric here
        }
```

### 2. Configure `log_strs`

In your experiment configuration (e.g., `experiment/star_easy_flexmdm.yaml`), add an entry to `lightning_module.log` (which maps to `log_strs` in `Harness`). 

The keys are the names you want to see in the logger (e.g., Weights & Biases or TensorBoard), and the values are the keys in the dictionary returned by your loss function.

```yaml
# experiment/star_easy_flexmdm_correction.yaml

lightning_module:
  log:
    "train/loss": "loss"
    "train/unmask_loss": "unmask_loss"
    "train/insertion_loss": "insertion_loss"
    "train/my_custom_name": "my_metric"  # Map "my_metric" from loss_dict to "train/my_custom_name"
```

### How it works in `Harness`

The `Harness` class (in `lib/xlm-core/src/xlm/harness.py`) iterates over `self.log_strs` during the training step and logs each entry:

```python:821:832:lib/xlm-core/src/xlm/harness.py
        if stage == "train":
            for k, v in self.log_strs.items():
                self.log(
                    k,
                    loss_dict[v].detach(),
                    on_step=True,
                    on_epoch=False,
                    prog_bar=True,
                    sync_dist=False,
                    rank_zero_only=True,
                    logger=True,
                    add_dataloader_idx=False,
                )
```

By default, `prog_bar=True` is set, so these metrics will also appear in your progress bar during training.
