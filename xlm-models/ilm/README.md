# ILM - Infilling Language Model

> **Migration Note**: This model has been migrated from `xlm.lm.ilm` to be an independent external package. 
> All import paths have been updated from `xlm.lm.ilm.*` → `ilm.*` and config targets updated accordingly.
> The functionality remains exactly the same.

## Installation

```bash
# Install from xlm-models
pip install xlm-models[ilm]

# Or install directly
pip install ./ilm

# Development installation
pip install -e ./ilm
```

## Usage with New Import Paths

```yaml
# Model configuration (updated target path)
model:
  _target_: ilm.model_ilm.RotaryTransformerILMModel

# Loss configuration (updated target path)  
loss:
  _target_: ilm.loss_ilm.ILMLossWithMaskedCE

# Predictor configuration (updated target path)
predictor:
  _target_: ilm.predictor_ilm.ILMPredictor
```

---

# Original ILM Documentation

# Conditional Generation

```bash
python src/xlm/commands/cli_demo.py "job_type=demo" "job_name=owt_ilm_demo" "experiment=owt_ilm" predictor.stopping_threshold=0.9 +hub/checkpoint=ilm_owt
```

# Unconditional Generation
```bash
python src/xlm/commands/lightning_main.py "job_type=generate" "job_name=owt_ilm" "experiment=owt_ilm" "debug=[overfit,print_predictions]" "+generation.ckpt_path=logs/owt_ilm5/checkpoints/57-600000.ckpt" datamodule.dataset_managers.predict.unconditional_prediction.num_examples=5
```


# Evaluate

```bash
xlm "job_type=eval" "job_name=owt_ilm_eval" "experiment=[owt_ilm,gpt2_generative_perplexity]" "++eval.checkpoint_path=logs/owt_ilm5/checkpoints/75-800000.ckpt" "debug=eval_unconditional_preds" +predictor.use_high_precision=true predictor.p=0.9 trainer.limit_val_batches=20 ~datamodule.dataset_managers.val.lm datamodule.dataset_managers.val.unconditional_prediction.num_examples=100
```