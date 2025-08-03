

# Demo

```bash
python src/xlm/commands/cli_demo.py "job_type=demo" "job_name=owt_ilm_demo" "experiment=owt_ilm" predictor.stopping_threshold=0.9 +hub/checkpoint=ilm_owt
```

# Evaluate

```bash
xlm "job_type=eval" "job_name=owt_ilm_eval" "experiment=[owt_ilm,gpt2_generative_perplexity]" "++eval.checkpoint_path=logs/owt_ilm
5/checkpoints/66-702500.ckpt" "debug=eval_unconditional_preds" +predictor.use_high_precision=true predictor.p=0.9
```