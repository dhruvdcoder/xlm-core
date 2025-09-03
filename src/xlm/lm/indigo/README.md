

# Indigo


1. Create a new environment using conda.

```
conda create -n venv_indigo python=3.11.10 pip ipykernel -y
conda activate venv_indigo
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

## Create `.env` file
Create a `.env` file in the root directory with the following contents. Make appropriate changes to the paths.
```bash
WANDB_ENTITY=ilm-extensions
WANDB_PROJECT=indigo
DATA_DIR=/work/pi_mccallum_umass_edu/dhruveshpate_umass_edu/indigo/data
HF_HOME=/work/pi_mccallum_umass_edu/dhruveshpate_umass_edu/indigo//hf_cache
LOG_DIR=/work/pi_mccallum_umass_edu/dhruveshpate_umass_edu/indigo/dhruvesh_logs
TOKENIZERS_PARALLELISM=false
PROJECT_ROOT=.
HYDRA_FULL_ERROR=1
OC_CAUSE=1
```


## Using wandb
Add the wandb account/entity and project name in the .env file.

```bash
WANDB_ENTITY=ilm-extensions
WANDB_PROJECT=indigo
```


#### Star Graphs

The script `python src/xlm/commands/generate_star_graphs.py` generates the star graphs datasets. The ones used in the paper are availabe on huggingface hub `dhruveshpatel/star-<easy/medium/hard>`.


```bash
########################################################
# region: Star Easy
# debug run on an interactive node
python src/xlm/commands/lightning_main.py "job_type=train" "job_name=star_easy_indigo" "experiment=star_easy_indigo" "debug=overfit"
# final run
python src/xlm/commands/lightning_main.py "job_type=train" "job_name=star_easy_indigo" "experiment=star_easy_indigo"
# submit a  slurm job for final run
python scripts/submit_train.py "do=submit" "job_name=star_easy_indigo" "train.experiment=star_easy_indigo" "hardware=1_node_1_gpu" "slurm.constraint=\"vram40,bf16\"" --- "compile=false" "trainer.precision=bf16-mixed" "trainer.num_nodes=1" "trainer.devices=1" "trainer_strategy=single_device"
# If you don't want to submit a job but just want generate the sbatch script, change "do=submit" to "do=print"
python scripts/submit_train.py "do=print" "job_name=star_easy_indigo" "train.experiment=star_easy_indigo" "hardware=1_node_1_gpu" "slurm.constraint=\"vram40,bf16\"" --- "compile=false" "trainer.precision=bf16-mixed" "trainer.num_nodes=1" "trainer.devices=1" "trainer_strategy=single_device"
#endregion: Star Easy
########################################################

########################################################
#Star Medium: Just change the names
python scripts/submit_train.py "do=submit" "job_name=star_medium_indigo" "train.experiment=star_medium_indigo" "hardware=1_node_1_gpu" "slurm.constraint=\"vram40,bf16\"" --- "compile=false" "trainer.precision=bf16-mixed" "trainer.num_nodes=1" "trainer.devices=1" "trainer_strategy=single_device"

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





