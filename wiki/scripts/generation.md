In order to perform unconditional generation we need to specify the `job_type=generate`, `job_name` and `experiment`. The example below shows how to generate from an ILM training checkpoint. 
   
```bash
xlm "job_type=generate" \
"job_name=owt_ilm" \
"experiment=owt_ilm" \
"debug=[overfit,print_predictions]" \
"+generation.ckpt_path=logs/owt_ilm5/checkpoints/40-422500.ckpt" \
"datamodule.dataset_managers.predict.unconditional_prediction.num_examples=5" \
"predictor.stopping_threshold=0.9"
```

# Demo

Interactive prompt → generation (`job_type=demo`). See [docs/guide/demo.md](../../docs/guide/demo.md).

```bash
# OWT ILM
python -m xlm.commands.cli_demo \
  job_type=demo job_name=owt_ilm_hub_demo experiment=owt_ilm \
  +hub.repo_id=dhruveshpatel/ilm-owt +hub.revision=step-800000 \
  +trainer.precision=32-true compile=false model.force_flash_attn=false \
  predictor.stopping_threshold=0.9 predictor.max_steps=1024 \
  paths.output_dir=/tmp/xlm_cli_demo_ilm_owt

# OWT FlexMDM
python -m xlm.commands.cli_demo \
  job_type=demo job_name=owt_flexmdm_hub_demo experiment=owt_flexmdm \
  +hub.repo_id=dhruveshpatel/flexmdm-owt +hub.revision=step-800000 \
  +trainer.precision=32-true compile=false model.force_flash_attn=false \
  predictor.max_steps=1024 ++predictor.top_p=0.95 \
  paths.output_dir=/tmp/xlm_cli_demo_flexmdm_owt
```
