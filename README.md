# ILM


1. Create a new environment using conda.

```
conda create -n venv_xlm python=3.11.10 pip ipykernel -y
conda activate venv_xlm
```

```
pip install -r requirements/core_requirements.txt && \
pip install -r requirements/test_requirements.txt && \
pip install -r requirements/lint_requirements.txt && \
pip install -e .
```


If you are working on Unity in a scratch space, I recommend creating an environment in your `/work/pi_*/user/*` directory or in the scratch workspace itself and not in your home directory.





#### Star Graphs

The script `python src/xlm/commands/generate_star_graphs.py` generates the star graphs datasets. The ones used in the paper are availabe on huggingface hub `dhruveshpatel/star-<easy/medium/hard>`.


```bash
########################################################
# region: Star Easy
# debug run on an interactive node
python src/xlm/commands/lightning_main.py "job_type=train" "job_name=star_easy" "experiment=star_easy_ilm" "debug=overfit"
# final run
python src/xlm/commands/lightning_main.py "job_type=train" "job_name=star_easy" "experiment=star_easy_ilm"
# submit a  slurm job for final run
python scripts/submit_train.py "do=submit" "job_name=ilm_star_easy" "train.experiment=star_easy_ilm" "hardware=1_node_1_gpu" "slurm.constraint=\"vram40,bf16\"" --- "compile=false" "trainer.precision=bf16-mixed" "trainer.num_nodes=1" "trainer.devices=1" "trainer_strategy=single_device"
# If you don't want to submit a job but just want generate the sbatch script, change "do=submit" to "do=print"
python scripts/submit_train.py "do=print" "job_name=ilm_star_easy" "train.experiment=star_easy_ilm" "hardware=1_node_1_gpu" "slurm.constraint=\"vram40,bf16\"" --- "compile=false" "trainer.precision=bf16-mixed" "trainer.num_nodes=1" "trainer.devices=1" "trainer_strategy=single_device"
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
```bash
python src/xlm/commands/lightning_main.py "job_type=prepare_data" "job_name=lm1b_prepare_data"
python src/xlm/commands/lightning_main.py "job_type=train" "job_name=lm1b" "experiment=lm1b_ilm" "debug=overfit"
```


#### OpenWebText

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






# Log

TODO: 
0. On exception checkpointing is not working. See line 1452 in `logs/owt_ilm/sbatch/2025-06-30_22-15-58/owt_ilm.out`.
  * (Done) possible solutions update fsspec to 2025.5.0. I've updated the environment but the run is already on its way. Will wait for the next run.

- owt training still fails towards the end of an epoch.
  - (fixed) reduced the number of shards per node to less than the batch size. Waiting to see if it works.

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