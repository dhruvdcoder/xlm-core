# IDLM - Iterative Diffusion Language Model

> **Migration Note**: This model has been migrated from `xlm.lm.idlm` to be an independent external package. 
> All import paths have been updated from `xlm.lm.idlm.*` → `idlm.*` and config targets updated accordingly.
> The functionality remains exactly the same.

## Installation

```bash
# Install from xlm-models
pip install xlm-models[idlm]

# Or install directly
pip install ./idlm

# Development installation
pip install -e ./idlm
```

## Usage with New Import Paths

```yaml
# Model configuration (updated target path)
model:
  _target_: idlm.model_idlm.DDITIDLMModel

# Loss configuration (updated target path)  
loss:
  _target_: idlm.loss_idlm.IdlmLoss

# Predictor configuration (updated target path)
predictor:
  _target_: idlm.predictor_idlm.IdlmPredictor
```

---

# Original IDLM Documentation

1. Create a new environment using conda.

```
conda create -n venv_idlm python=3.11.10 pip ipykernel -y
conda activate venv_idlm
```
## Install hydra from source
If you want dynamic help messages, we need to build the latest hydra from source. To do this you will need jdk installed.
If you are on a cluster, you may be able to load java using `module load jdk` or something similar.
If your system does not have `java`, open `build_hydra.sh` and uncomment the lines that install jdk (don't worry, it will be removed after hydra is installed.)
```bash
cd requirements && chmod +x build_hydra.sh && ./build_hydra.sh && cd ..
```

## Install hydra from pypi
This is the easier option but you will not get the dynamic help messages.
```bash
pip install hydra-core
```

## Install the rest of the requirements
```
pip install -r requirements/core_requirements.txt && \
pip install -r requirements/test_requirements.txt && \
pip install -r requirements/lint_requirements.txt && \
pip install -r requirements/extra_requirements.txt && \
pip install -r requirements/plot_requirements.txt && \
pip install -e .
```


If you are working on Unity in a scratch space, I recommend creating an environment in your `/work/pi_*/user/*` directory or in the scratch workspace itself and not in your home directory.




#### Star Graphs

The script `python src/xlm/commands/generate_star_graphs.py` generates the star graphs datasets. The ones used in the paper are availabe on huggingface hub `dhruveshpatel/star-<easy/medium/hard>`.


```bash
########################################################
# region: Star Easy
# debug run on an interactive node
python src/xlm/commands/lightning_main.py "job_type=train" "job_name=star_easy_idlm" "experiment=star_easy_idlm" "debug=overfit"
# final run
python src/xlm/commands/lightning_main.py "job_type=train" "job_name=star_easy_idlm" "experiment=star_easy_idlm"
# submit a  slurm job for final run
python scripts/submit_train.py "do=submit" "job_name=star_easy_idlm" "train.experiment=star_easy_idlm" "hardware=1_node_1_gpu" "slurm.constraint=\"vram40,bf16\"" --- "compile=false" "trainer.precision=bf16-mixed" "trainer.num_nodes=1" "trainer.devices=1" "trainer_strategy=single_device"
# If you don't want to submit a job but just want generate the sbatch script, change "do=submit" to "do=print"
python scripts/submit_train.py "do=print" "job_name=star_easy_idlm" "train.experiment=star_easy_idlm" "hardware=1_node_1_gpu" "slurm.constraint=\"vram40,bf16\"" --- "compile=false" "trainer.precision=bf16-mixed" "trainer.num_nodes=1" "trainer.devices=1" "trainer_strategy=single_device"
#endregion: Star Easy
########################################################

########################################################
#Star Medium: Just change the names
python scripts/submit_train.py "do=submit" "job_name=star_medium_idlm" "train.experiment=star_medium_idlm" "hardware=1_node_1_gpu" "slurm.constraint=\"vram40,bf16\"" --- "compile=false" "trainer.precision=bf16-mixed" "trainer.num_nodes=1" "trainer.devices=1" "trainer_strategy=single_device"

# to evaluate an existing checkpoint, just add "job_type=eval" and "++eval.checkpoint_path=path/to/checkpoint.ckpt" and "++eval.split=test"
# for example 
python src/xlm/commands/lightning_main.py "job_type=eval" "job_name=star_medium_eval" "experiment=star_medium_idlm" "++eval.checkpoint_path=logs/idlm_star_medium/checkpoints/last.ckpt" "++eval.split=validation"

########################################################

########################################################
# region: Star Hard
python scripts/submit_train.py "do=submit" "job_name=star_hard_idlm" "train.experiment=star_hard_idlm" "hardware=1_node_1_gpu" "slurm.constraint=\"vram40,bf16\"" --- "compile=false" "trainer.precision=bf16-mixed" "trainer.num_nodes=1" "trainer.devices=1" "trainer_strategy=single_device"
# endregion: Star Hard
########################################################
```
#### LM1B
```bash
python src/xlm/commands/lightning_main.py "job_type=prepare_data" "job_name=lm1b_prepare_data"
python src/xlm/commands/lightning_main.py "job_type=train" "job_name=lm1b" "experiment=lm1b_idlm" "debug=overfit"
# final run
python scripts/submit_train.py "do=submit" "job_name=lm1b_idlm" "train.experiment=lm1b_idlm" "train.batch_size=128" "train.compile=true" "train.precision=bf16-mixed" "hardware=ddp_4_node_1_gpu" "slurm.constraint=\"vram80,bf16,ib\"" "++slurm.exclude=gpu016"
```


#### OpenWebText

We filter out sequences with length > 1024 using the GPT2 tokenizer using the script `python src/xlm/commands/split_owt.py `. The filtered dataset is available on huggingface hub `dhruveshpatel/owt-gpt2-1024-split`.

```bash
# prepare data
python src/xlm/commands/lightning_main.py "job_type=prepare_data" "job_name=owt_prepare_data" "experiment=owt_idlm"

# interactive debug overfit
python src/xlm/commands/lightning_main.py "job_type=train" "job_name=owt_idlm" "experiment=owt_idlm" "debug=overfit"

# multinode debug on 2 nodes with compile on.
python scripts/submit_train.py "do=submit" "job_name=owt_idlm_debug_multinode_compile" "train.experiment=owt_idlm" "train.batch_size=32" "hardware=ddp_2_node_1_gpu_debug" "slurm.constraint=\"vram80,bf16,ib\"" --- "compile=true" "trainer.precision=bf16-mixed" "trainer.num_nodes=2" "trainer.devices=1" "trainer_strategy=ddp_multinode" "debug=multinode"

# final run
NUM_NODES=4 # or 8
python scripts/submit_train.py "do=submit" "job_name=owt_idlm" "train.experiment=owt_idlm" "train.batch_size=32" "train.compile=false" "train.precision=bf16-mixed" "hardware=ddp_${NUM_NODES}_node_1_gpu" "slurm.constraint=\"vram80,bf16,ib\"" "++slurm.exclude=gpu016"

python scripts/submit_train.py "do=submit" "job_name=owt_idlm5" "train.experiment=owt_idlm" "train.batch_size=32" "hardware=ddp_4_node_1_gpu" "slurm.constraint=\"vram80,bf16,ib\""  "++slurm.exclude=gpu016" --- "compile=true" "trainer.precision=bf16-mixed" "trainer.num_nodes=4" "trainer.devices=1" "trainer_strategy=ddp_multinode" "optimizer.lr=0.0001"

# generate from a checkpoint
python src/xlm/commands/lightning_main.py "job_type=generate" "job_name=owt_idlm" "experiment=owt_idlm" "debug=[overfit,print_predictions]" "+generation.ckpt_path=logs/owt_idlm5/checkpoints/19-200000.ckpt" datamodule.dataset_managers.predict.unconditional_prediction.num_examples=2

# push to hub
python src/xlm/commands/push_to_hub.py "job_type=push_to_hub" "job_name=owt_ilm_hub" "experiment=owt_ilm" "+hub_checkpoint_path=logs/owt_ilm5/checkpoints/40-422500.ckpt" +hub.repo_id="dhruveshpatel/ilm-owt"

# run the demo
python src/xlm/commands/cli_demo.py "job_type=demo" "job_name=owt_ilm_demo" "experiment=owt_ilm" predictor.stopping_threshold=0.9 +global_flags.DEBUG_PRINT_PREDS=true +hub/checkpoint=ilm_owt
```
# Demo

```bash
python src/xlm/commands/cli_demo.py "job_type=demo" "job_name=owt_ilm_demo" "experiment=owt_ilm" predictor.stopping_threshold=0.9 +hub/checkpoint=ilm_owt
```

# Evaluate

```bash
xlm "job_type=eval" "job_name=owt_ilm_eval" "experiment=[owt_ilm,gpt2_generative_perplexity]" "++eval.checkpoint_path=logs/owt_ilm
5/checkpoints/66-702500.ckpt" "debug=eval_unconditional_preds" +predictor.use_high_precision=true predictor.p=0.9
```

# Logs
- Faulty nodes. Some runs are 3x slower for no apparent reason. 
  - In all these runs there is on faulty GPU with lower utilization (92%) compared to the rest (99%).
  - 07/04: node016 is currently faulty.
  - I need to use custom [WandbLogger](https://github.com/Lightning-AI/pytorch-lightning/issues/20774) to log system metrics from all ranks.

- ILM OWT training fails after 60k steps due to NaN loss probably due to stopping loss. 


1. Implement MLM, MDLM with all the necessary predictors (get help)
  * (WIP) Implementing predictor for MLM.
  *  Test them on star graphs
2. Implement ARLM (get help)
  * test on star graphs
3. Implement InDIGO (difficult)
4. Implement Slidinging Insertion (difficult)