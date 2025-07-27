To train the model in interactive mode we need to specify the `job_type=train`, `job_name` and `experiment`. Following is an example command to train ILM model on openwebtext.

```bash
xlm "job_type=train" "job_name=owt_ilm" "experiment=owt_ilm"
```

# Debugging

We have various debugging config overrides in the `configs/lightning_train/debug` directory. The most useful one is `debug=overfit` which will overfit the model on a single batch. To use it, simply add it to the command line arguments. For example:

```bash
xlm "job_type=train" "job_name=owt_ilm" "experiment=owt_ilm" "debug=overfit"
```
You can create your own debug configs and use them see `configs/lightning_train/debug/` directory for examples.