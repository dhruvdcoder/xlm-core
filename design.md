# Common Harness and Callbacks

In huggingface, the trainer acts like a harness. There is not equivalent of LightningModule in hf.
The question is how do we provide common harness that can work with both lightning and hf?
For HF, we will have to create our own custom trainer that inherits from the HF trainer and sends calls like
`training_step` etc to our harness. I think this will be a non-trivial task. So this should be done later when we need it.
The HF trainer also does all the logging by itself.


As for the callbacks, the hooks on HF callbacks take in a lot of arguments, while lightning callback only takes trainer and lightning module.
It will be non-trivial to create a full-map of arguments across interfaces. So our callbacks should be mixins that implement the core functionality of the
methods like `_on_train_batch_start(self, arg1, arg2, ..., arg_k)` and `_on_train_batch_end(self, arg1, arg2, ... arg_n)` etc.
where each method will have a unique, fully fixed signature with the exact arguments, i.e. if the callback needs current training step,
it will specify it as an explicit argument `_on_train_batch_start(self, current_step)`. Then for each framework, we will create a child class
inheriting from the mixin and implementing the necessary handshake to pass in the arguments needed by the core method.




```python
    """
    A class for objects that will inspect the state of the training loop at some events and take some decisions. At
    each of those events the following arguments are available:

    Args:
        args ([`TrainingArguments`]):
            The training arguments used to instantiate the [`Trainer`].
        state ([`TrainerState`]):
            The current state of the [`Trainer`].
        control ([`TrainerControl`]):
            The object that is returned to the [`Trainer`] and can be used to make some decisions.
        model ([`PreTrainedModel`] or `torch.nn.Module`):
            The model being trained.
        tokenizer ([`PreTrainedTokenizer`]):
            The tokenizer used for encoding the data. This is deprecated in favour of `processing_class`.
        processing_class ([`PreTrainedTokenizer` or `BaseImageProcessor` or `ProcessorMixin` or `FeatureExtractionMixin`]):
            The processing class used for encoding the data. Can be a tokenizer, a processor, an image processor or a feature extractor.
        optimizer (`torch.optim.Optimizer`):
            The optimizer used for the training steps.
        lr_scheduler (`torch.optim.lr_scheduler.LambdaLR`):
            The scheduler used for setting the learning rate.
        train_dataloader (`torch.utils.data.DataLoader`, *optional*):
            The current dataloader used for training.
        eval_dataloader (`torch.utils.data.DataLoader`, *optional*):
            The current dataloader used for evaluation.
        metrics (`Dict[str, float]`):
            The metrics computed by the last evaluation phase.

            Those are only accessible in the event `on_evaluate`.
        logs  (`Dict[str, float]`):
            The values to log.

            Those are only accessible in the event `on_log`.

    The `control` object is the only one that can be changed by the callback, in which case the event that changes it
    should return the modified version.

    The argument `args`, `state` and `control` are positionals for all events, all the others are grouped in `kwargs`.
    You can unpack the ones you need in the signature of the event using them. As an example, see the code of the
    simple [`~transformers.PrinterCallback`].

    Example:

    ```python
    class PrinterCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            _ = logs.pop("total_flos", None)
            if state.is_local_process_zero:
                print(logs)
    ```"""
```