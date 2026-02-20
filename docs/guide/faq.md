# FAQ

## General

**What is xLM?**  
xLM is a unified framework for developing and comparing non-autoregressive language models. It provides modular components for models, losses, predictors, and data collation.

**Which models are available?**  
The framework includes MLM, ILM, ARLM, MDLM, and IDLM. See the [Quick Start](quickstart.md) for the full list.

## Usage

**How do I train on a new dataset?**  
Use the appropriate experiment config for your model and dataset. For example: `xlm job_type=train job_name=my_run experiment=lm1b_ilm`.

**How do I debug training?**  
Add `debug=overfit` to overfit on a single batch, or use other debug configs from `configs/lightning_train/debug/`.

## Contributing

**How do I add a new model?**  
See the [Contributing Guide](contributing.md) for a complete walkthrough of the four components: Model, Loss, Predictor, and Collator.
