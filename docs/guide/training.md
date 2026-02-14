# Training

To train the model in interactive mode, specify `job_type=train`, `job_name`, and `experiment`. Example command to train the ILM model on OpenWebText:

```bash
xlm "job_type=train" "job_name=owt_ilm" "experiment=owt_ilm"
```

## Debugging

We have various debugging config overrides in the `configs/lightning_train/debug` directory. The most useful one is `debug=overfit`, which overfits the model on a single batch. To use it, add it to the command line arguments:

```bash
xlm "job_type=train" "job_name=owt_ilm" "experiment=owt_ilm" "debug=overfit"
```

You can create your own debug configs and use them. See the `configs/lightning_train/debug/` directory for examples.
