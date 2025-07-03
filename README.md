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
```

TODO: 
0. On exception checkpointing is not working. See line 1452 in `logs/owt_ilm/sbatch/2025-06-30_22-15-58/owt_ilm.out`.
  * possible solutions update fsspec to 2025.5.0. I've updated the environment but the run is already on its way. Will wait for the next run.
1. Implement MLM, MDLM with all the necessary predictors (get help)
  *  Test them on star graphs
2. Implement ARLM (get help)
  * test on star graphs
3. Implement InDIGO (difficult)
4. Implement Slidinging Insertion (difficult)



# Directory Structure

## Inside xlm/src

1. `commands`: Contains the scripts and their notebook versions.
2. `models`: Contains complete networks used for generation.
3. `modules`: Contains the building blocks of the networks.
4. `utils`: Contains utility functions.
5. `datamodule`: Contains one file for each dataset.
6. `diffusion`: Contains one file for each type of diffusion. The main object in each file a lightning module.

# Misc
Sparse tensor don't support division. So we need to avoid all division including the ones due to normalization in grad accum.
```python
# .venv_text_diffusion/lib/python3.11/site-packages/lightning/pytorch/loops/optimization/automatic.py:82
#closure_loss = closure_loss / normalize
# HACK: Can't divide sparse tensor with 
closure_loss = closure_loss * (1.0 / (normalize))
```


# New scripts
Note the arguments after `---` are passed to the inner `lightning_main_v2.py`.
```bash
# IDLM vstar medium diffusion weights loglinear
python scripts/submit_train.py "do=submit" "job_name=idlm_vstar_medium_diffusion_weights_loglinear" "train.experiment=idlm_vstar_medium_diffusion_weights_loglinear" "hardware=1_node_1_gpu" "slurm.constraint=\"vram80,bf16\"" --- "compile=false" "trainer.precision=bf16-mixed" "trainer.num_nodes=1" "trainer.devices=1" "trainer_strategy=single_device"

# IDLM vstar medium poisson with every diffusion weight
python scripts/submit_train.py "do=submit" "job_name=idlm_vstar_medium_poisson" "train.experiment=idlm_vstar_medium_poisson" "hardware=1_node_1_gpu" "slurm.constraint=\"vram80,bf16\"" --- "compile=false" "trainer.precision=bf16-mixed" "trainer.num_nodes=1" "trainer.devices=1" "trainer_strategy=single_device"

# CLEANUP: Don't need this anymore. We use the next command
# IDLM stories diffusion geometric
python scripts/submit_train.py "do=submit" "job_name=idlm_stories_geometric" "train.experiment=idlm_stories_diffusion_geometric" "hardware=ddp_4_node_1_gpu" "slurm.constraint=\"vram80,bf16,ib\"" --- "compile=true" "trainer.precision=bf16-mixed" "trainer.num_nodes=4" "trainer.devices=1" "trainer_strategy=ddp_multinode"

# IDLM stories
# Does not use diffusion weights in loss
python scripts/submit_train.py "do=submit" "job_name=idlm_stories" "train.experiment=idlm_stories" "hardware=ddp_4_node_1_gpu" "slurm.constraint=\"vram80,bf16,ib\"" --- "compile=true" "trainer.precision=bf16-mixed" "trainer.num_nodes=4" "trainer.devices=1" "trainer_strategy=ddp_multinode"

# IDLM stories v3
# Does not use diffusion weights in loss
# We pass time instead of total_noise to the model
python scripts/submit_train.py "do=submit" "job_name=idlm_stories_v3" "train.experiment=idlm_stories_v3" "hardware=ddp_4_node_1_gpu" "slurm.constraint=\"vram80,bf16,ib\"" --- "compile=true" "trainer.precision=bf16-mixed" "trainer.num_nodes=4" "trainer.devices=1" "trainer_strategy=ddp_multinode"

# CLEANUP: Don't need this anymore. We use the next command
# IDLM lm1b diffusion geometric
python scripts/submit_train.py "do=submit" "job_name=idlm_lm1b_geometric" "train.experiment=idlm_lm1b_diffusion_geometric" "hardware=ddp_4_node_1_gpu" "slurm.constraint=\"vram80,bf16,ib\"" --- "compile=true" "trainer.precision=bf16-mixed" "trainer.num_nodes=4" "trainer.devices=1" "trainer_strategy=ddp_multinode"

# IDLM lm1b
python scripts/submit_train.py "do=submit" "job_name=idlm_lm1b_v3" "train.experiment=idlm_lm1b_v4" "train.batch_size=64" "hardware=ddp_4_node_1_gpu" "slurm.constraint=\"vram80,bf16,ib\"" --- "compile=true" "trainer.precision=bf16-mixed" "trainer.num_nodes=4" "trainer.devices=1" "trainer_strategy=ddp_multinode"


# Prepare data for grammar dataset
python src/xlm/commands/lightning_main_v2.py "job_type=prepare_data" "job_name=idlm_grammar_prepare_data" "experiment=idlm_grammar"

# IDLM grammar poisson
python scripts/submit_train.py "do=submit" "job_name=idlm_grammar_poisson" "train.experiment=idlm_grammar" "hardware=1_node_1_gpu" "slurm.constraint=\"vram40,bf16\"" --- "compile=false" "trainer.precision=bf16-mixed" "trainer.num_nodes=1" "trainer.devices=1" "trainer_strategy=single_device"

# ILM grammar
python scripts/submit_train.py "do=submit" "job_name=ilm_grammar" "train.experiment=ilm_grammar" "hardware=1_node_1_gpu" "slurm.constraint=\"vram40,bf16\"" --- "compile=false" "trainer.precision=bf16-mixed" "trainer.num_nodes=1" "trainer.devices=1" "trainer_strategy=single_device"


# IT vstar medium
python scripts/submit_train.py "do=submit" "job_name=it_stochastic_vstar_medium" "train.experiment=it_stochastic_vstar_medium" "hardware=1_node_1_gpu" "slurm.constraint=\"vram80,bf16\"" --- "compile=false" "trainer.precision=bf16-mixed" "trainer.num_nodes=1" "trainer.devices=1" "trainer_strategy=single_device"

# IT vstar small
python scripts/submit_train.py "do=submit" "job_name=it_stochastic_vstar_small" "train.experiment=it_stochastic_vstar_small" "hardware=1_node_1_gpu" "slurm.constraint=\"vram80,bf16\"" --- "compile=false" "trainer.precision=bf16-mixed" "trainer.num_nodes=1" "trainer.devices=1" "trainer_strategy=single_device"

# IT star small
python scripts/submit_train.py "do=submit" "job_name=it_stochastic_star_small" "train.experiment=it_stochastic_star_small" "hardware=1_node_1_gpu" "slurm.constraint=\"vram80,bf16\"" --- "compile=false" "trainer.precision=bf16-mixed" "trainer.num_nodes=1" "trainer.devices=1" "trainer_strategy=single_device"

# IMPORTANT NOTE: These XLNet versions are trained with random output order, but use left-to-right generation at inference time. 
# XLNet vstar medium
python scripts/submit_train.py "do=submit" "job_name=xlnet_vstar_medium" "train.experiment=xlnet_vstar_medium" "hardware=1_node_1_gpu" "slurm.constraint=\"vram80,bf16\"" --- "compile=false" "trainer.precision=bf16-mixed" "trainer.num_nodes=1" "trainer.devices=1" "trainer_strategy=single_device"

# XLNet vstar small
python scripts/submit_train.py "do=submit" "job_name=xlnet_vstar_small" "train.experiment=xlnet_vstar_small" "hardware=1_node_1_gpu" "slurm.constraint=\"vram80,bf16\"" --- "compile=false" "trainer.precision=bf16-mixed" "trainer.num_nodes=1" "trainer.devices=1" "trainer_strategy=single_device"

# XLNet star small
python scripts/submit_train.py "do=submit" "job_name=xlnet_star_small" "train.experiment=xlnet_star_small" "hardware=1_node_1_gpu" "slurm.constraint=\"vram80,bf16\"" --- "compile=false" "trainer.precision=bf16-mixed" "trainer.num_nodes=1" "trainer.devices=1" "trainer_strategy=single_device"


# ILM Tiny2 vstar medium
python scripts/submit_train.py "do=submit" "job_name=ilm_tiny2_vstar_medium" "train.experiment=ilm_tiny2_vstar_medium" "hardware=1_node_1_gpu" "slurm.constraint=\"vram80,bf16\"" --- "compile=false" "trainer.precision=bf16-mixed" "trainer.num_nodes=1" "trainer.devices=1" "trainer_strategy=single_device"

# ILM XTiny2 vstar medium
python scripts/submit_train.py "do=submit" "job_name=ilm_xtiny2_vstar_medium" "train.experiment=ilm_xtiny2_vstar_medium" "hardware=1_node_1_gpu" "slurm.constraint=\"vram80,bf16\"" "slurm.time=06:00:00" --- "compile=false" "trainer.precision=bf16-mixed" "trainer.num_nodes=1" "trainer.devices=1" "trainer_strategy=single_device"

# ILM Tiny2 vstar small
python scripts/submit_train.py "do=submit" "job_name=ilm_tiny2_vstar_small" "train.experiment=ilm_tiny2_vstar_small" "hardware=1_node_1_gpu" "slurm.constraint=\"vram80,bf16\"" "slurm.time=06:00:00" --- "compile=false" "trainer.precision=bf16-mixed" "trainer.num_nodes=1" "trainer.devices=1" "trainer_strategy=single_device"

# ILM XTiny2 vstar small
python scripts/submit_train.py "do=submit" "job_name=ilm_xtiny2_vstar_small" "train.experiment=ilm_xtiny2_vstar_small" "hardware=1_node_1_gpu" "slurm.constraint=\"vram80,bf16\"" "slurm.time=06:00:00" --- "compile=false" "trainer.precision=bf16-mixed" "trainer.num_nodes=1" "trainer.devices=1" "trainer_strategy=single_device"

# ILM Tiny2 star small
python scripts/submit_train.py "do=submit" "job_name=ilm_tiny2_star_small" "train.experiment=ilm_tiny2_star_small" "hardware=1_node_1_gpu" "slurm.constraint=\"vram80,bf16\"" "slurm.time=06:00:00" --- "compile=false" "trainer.precision=bf16-mixed" "trainer.num_nodes=1" "trainer.devices=1" "trainer_strategy=single_device"

# ILM XTiny2 star small
python scripts/submit_train.py "do=submit" "job_name=ilm_xtiny2_star_small" "train.experiment=ilm_xtiny2_star_small" "hardware=1_node_1_gpu" "slurm.constraint=\"vram80,bf16\"" "slurm.time=06:00:00" --- "compile=false" "trainer.precision=bf16-mixed" "trainer.num_nodes=1" "trainer.devices=1" "trainer_strategy=single_device"
```

Generation
```bash
# IDLM stories
# Don't forget to change the ckpt path and predictor params
# predictor.sampling_method=sample_top_p 
# predictor.p=0.5
# predictor.second_sampling_method=sample_top_k
# predictor.second_top=1
# predictor.length_temperature=1.0
# predictor.use_first_step_factor=false # false generates longer sequences
python scripts/submit_generate.py "do=submit" "job_name=idlm_stories_generate" "generate.experiment=idlm_stories_generate" "generate.ckpt_path=/scratch3/workspace/dhruveshpate_umass_edu-text_diffusion/xlm/logs/idlm_stories_v2/checkpoints/23-100000.ckpt" --- "predictor.sampling_method=sample_top_p" "predictor.p=0.2" "predictor.second_sampling_method=sample_top_k" "predictor.second_top=1" "predictor.length_temperature=1.0" "predictor.use_first_step_factor=false"

python scripts/submit_generate.py "do=submit" "job_name=idlm_stories_generate" "generate.experiment=idlm_stories_generate" "generate.ckpt_path=/scratch3/workspace/dhruveshpate_umass_edu-text_diffusion/xlm/logs/idlm_stories_v2/checkpoints/23-100000.ckpt" --- "predictor.sampling_method=sample_top_p" "predictor.p=0.2" "predictor.second_sampling_method=sample_top_k" "predictor.second_top=1" "predictor.length_temperature=1.0" "predictor.use_first_step_factor=true"

python scripts/submit_generate.py "do=submit" "job_name=idlm_stories_generate_loglinear" "generate.experiment=idlm_stories_generate_loglinear" "generate.ckpt_path=/scratch3/workspace/dhruveshpate_umass_edu-text_diffusion/xlm/logs/idlm_stories_loglinear/checkpoints/23-100000.ckpt" --- "predictor.sampling_method=sample_top_p" "predictor.p=0.2" "predictor.second_sampling_method=sample_top_k" "predictor.second_top=1" "predictor.length_temperature=1.0" "predictor.use_first_step_factor=true"

# LM1B
# Don't forget to change the ckpt path and predictor params
# predictor.sampling_method=sample_top_p 
# predictor.p=0.5
# predictor.second_sampling_method=sample_top_k
# predictor.second_top=1
# predictor.length_temperature=1.0
# predictor.use_first_step_factor=false # false generates longer sequences
python scripts/submit_generate.py "do=submit" "job_name=idlm_lm1b_generate" "generate.experiment=idlm_lm1b_generate" "generate.ckpt_path=/scratch3/workspace/dhruveshpate_umass_edu-text_diffusion/xlm/logs/idlm_lm1b_v4/checkpoints/16-1000000.ckpt" --- "predictor.sampling_method=sample_top_p" "predictor.p=0.2" "predictor.second_sampling_method=sample_top_k" "predictor.second_top=1" "predictor.length_temperature=1.0" "predictor.use_first_step_factor=false"

python scripts/submit_generate.py "do=submit" "job_name=idlm_lm1b_generate" "generate.experiment=idlm_lm1b_generate" "generate.ckpt_path=/scratch3/workspace/dhruveshpate_umass_edu-text_diffusion/xlm/logs/idlm_lm1b_v4/checkpoints/16-1000000.ckpt" --- "predictor.sampling_method=sample_top_p" "predictor.p=0.2" "predictor.second_sampling_method=sample_top_k" "predictor.second_top=1" "predictor.length_temperature=1.0" "predictor.use_first_step_factor=true"
```
