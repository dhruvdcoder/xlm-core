

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
xlm "job_type=eval" "job_name=owt_ilm_eval" "experiment=[owt_ilm,gpt2_generative_perplexity]" "++eval.checkpoint_path=logs/owt_ilm5/checkpoints/66-702500.ckpt" "debug=eval_unconditional_preds" +predictor.use_high_precision=true predictor.p=0.9 trainer.limit_val_batches=20 ~datamodule.dataset_managers.val.lm datamodule.dataset_managers.val.unconditional_prediction.num_examples=100
```