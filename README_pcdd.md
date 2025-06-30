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


If you are working on Unity in a scratch space, I recommend creating an environment in your `/work/pi_*/user/*` directory.


Installing cusf for `hyp1f1` function.

```
module load gcc
module load cuda/12.6
module load boost
module load cmake
module load ninja
mkdir lib
cd lib
git clone https://lab.compute.dtu.dk/cusf/cusf.git
cd cusf
make all
```



## Downloading datasets
### Language Modeling
1. LM1B
2. BabyLM

#### Preprocessing LM1B



#### Preprocessing BabyLM

1. Train the tokenizer
```
export PROJECT_ROOT=.
python src/xlm/commands/train_tokenizer.py experiment=train_babylm_tokenizer
```


# Training


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

# To do

1. Remove EOS token for varlen lm1b. EOS has not value for varlen model.


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
