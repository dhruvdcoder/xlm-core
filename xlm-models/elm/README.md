

# ELM


1. Create a new environment using conda.

```
conda create -n venv_xlm python=3.11.10 pip ipykernel -y
conda activate venv_xlm
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
pip install -e .
```


If you are working on Unity in a scratch space, I recommend creating an environment in your `/work/pi_*/user/*` directory or in the scratch workspace itself and not in your home directory.


## Using wandb
Add the wandb account/entity and project name in the .env file.

```bash
WANDB_ENTITY=ilm-extensions
WANDB_PROJECT=elm
```


#### Star Graphs

The script `python src/xlm/commands/generate_star_graphs.py` generates the star graphs datasets. The ones used in the paper are availabe on huggingface hub `dhruveshpatel/star-<easy/medium/hard>`.


```bash
########################################################
# region: Star Easy
# debug run on an interactive node
python src/xlm/commands/lightning_main.py "job_type=train" "job_name=star_easy_elm" "experiment=star_easy_elm" "debug=overfit"
# final run
python src/xlm/commands/lightning_main.py "job_type=train" "job_name=star_easy_elm" "experiment=star_easy_elm"
# submit a  slurm job for final run
python scripts/submit_train.py "do=submit" "job_name=star_easy_elm" "train.experiment=star_easy_elm" "hardware=1_node_1_gpu" "slurm.constraint=\"vram40,bf16\"" --- "compile=false" "trainer.precision=bf16-mixed" "trainer.num_nodes=1" "trainer.devices=1" "trainer_strategy=single_device"
# If you don't want to submit a job but just want generate the sbatch script, change "do=submit" to "do=print"
python scripts/submit_train.py "do=print" "job_name=star_easy_elm" "train.experiment=star_easy_elm" "hardware=1_node_1_gpu" "slurm.constraint=\"vram40,bf16\"" --- "compile=false" "trainer.precision=bf16-mixed" "trainer.num_nodes=1" "trainer.devices=1" "trainer_strategy=single_device"
#endregion: Star Easy
########################################################

########################################################
#Star Medium: Just change the names
python scripts/submit_train.py "do=submit" "job_name=ilm_star_medium" "train.experiment=star_medium_ilm" "hardware=1_node_1_gpu" "slurm.constraint=\"vram40,bf16\"" --- "compile=false" "trainer.precision=bf16-mixed" "trainer.num_nodes=1" "trainer.devices=1" "trainer_strategy=single_device"

# to evaluate an existing checkpoint, just add "job_type=eval" and "++eval.checkpoint_path=path/to/checkpoint.ckpt" and "++eval.split=test"
# for example 
python src/xlm/commands/lightning_main.py "job_type=eval" "job_name=star_medium_eval" "experiment=star_medium_ilm" "++eval.checkpoint_path=logs/ilm_star_medium/checkpoints/last.ckpt" "++eval.split=validation"

########################################################

########################################################
# region: Star Hard
python scripts/submit_train.py "do=submit" "job_name=ilm_star_hard" "train.experiment=star_hard_ilm" "hardware=1_node_1_gpu" "slurm.constraint=\"vram40,bf16\"" --- "compile=false" "trainer.precision=bf16-mixed" "trainer.num_nodes=1" "trainer.devices=1" "trainer_strategy=single_device"
# endregion: Star Hard
########################################################
```
#### LM1B

TODO


#### OpenWebText

TODO


We filter out sequences with length > 1024 using the GPT2 tokenizer using the script `python src/xlm/commands/split_owt.py `. The filtered dataset is available on huggingface hub `dhruveshpatel/owt-gpt2-1024-split`.

```bash
# prepare data
python src/xlm/commands/lightning_main.py "job_type=prepare_data" "job_name=owt_prepare_data" "experiment=owt_ilm"

# interactive debug overfit
python src/xlm/commands/lightning_main.py "job_type=train" "job_name=owt_ilm" "experiment=owt_ilm" "debug=overfit"

# multinode debug on 2 nodes with compile on.
python scripts/submit_train.py "do=submit" "job_name=owt_ilm_debug_multinode_compile" "train.experiment=owt_ilm" "train.batch_size=32" "hardware=ddp_2_node_1_gpu_debug" "slurm.constraint=\"vram80,bf16,ib\"" --- "compile=true" "trainer.precision=bf16-mixed" "trainer.num_nodes=2" "trainer.devices=1" "trainer_strategy=ddp_multinode" "debug=multinode"

# final run
NUM_NODES=4 # or 8
python scripts/submit_train.py "do=submit" "job_name=owt_ilm" "train.experiment=owt_ilm" "train.batch_size=32" "hardware=ddp_${NUM_NODES}_node_1_gpu" "slurm.constraint=\"vram80,bf16,ib\"" --- "compile=true" "trainer.precision=bf16-mixed" "trainer.num_nodes=$NUM_NODES" "trainer.devices=1" "trainer_strategy=ddp_multinode"

python scripts/submit_train.py "do=submit" "job_name=owt_ilm5" "train.experiment=owt_ilm" "train.batch_size=32" "hardware=ddp_4_node_1_gpu" "slurm.constraint=\"vram80,bf16,ib\""  "++slurm.exclude=gpu016" --- "compile=true" "trainer.precision=bf16-mixed" "trainer.num_nodes=4" "trainer.devices=1" "trainer_strategy=ddp_multinode" "optimizer.lr=0.0001"

# generate from a checkpoint
python src/xlm/commands/lightning_main.py "job_type=generate" "job_name=owt_ilm" "experiment=owt_ilm" "debug=[overfit,print_predictions]" "+generation.ckpt_path=logs/owt_ilm5/checkpoints/19-200000.ckpt" datamodule.dataset_managers.predict.unconditional_prediction.num_examples=2

# push to hub
python src/xlm/commands/push_to_hub.py "job_type=push_to_hub" "job_name=owt_ilm_hub" "experiment=owt_ilm" "+hub_checkpoint_path=logs/owt_ilm5/checkpoints/40-422500.ckpt" +hub.repo_id="dhruveshpatel/ilm-owt"

# run the demo
python src/xlm/commands/cli_demo.py "job_type=demo" "job_name=owt_ilm_demo" "experiment=owt_ilm" predictor.stopping_threshold=0.9 +global_flags.DEBUG_PRINT_PREDS=true +hub/checkpoint=ilm_owt
```



