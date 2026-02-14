# Generation

To perform unconditional generation, specify `job_type=generate`, `job_name`, and `experiment`. The example below shows how to generate from an ILM training checkpoint:

```bash
xlm "job_type=generate" \
  "job_name=owt_ilm" \
  "experiment=owt_ilm" \
  "debug=[overfit,print_predictions]" \
  "+generation.ckpt_path=logs/owt_ilm5/checkpoints/40-422500.ckpt" \
  "datamodule.dataset_managers.predict.unconditional_prediction.num_examples=5" \
  "predictor.stopping_threshold=0.9"
```

## Demo

```bash
python src/xlm/commands/cli_demo.py "job_type=demo" "job_name=owt_ilm_demo" "experiment=owt_ilm" predictor.stopping_threshold=0.9 +global_flags.DEBUG_PRINT_PREDS=true +hub/checkpoint=ilm_owt
```
