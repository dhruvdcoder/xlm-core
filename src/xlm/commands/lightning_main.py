# %%
# change dir to the root of the project
# create the notebook inside the commands directory
import os
import json

from xlm.utils.slurm import print_slurm_info

if "PROJECT_ROOT" not in os.environ:
    os.environ["PROJECT_ROOT"] = "."
os.environ["HYDRA_FULL_ERROR"] = "1"

# region: Import necessary modules
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
from xlm.utils import omegaconf_resolvers
from xlm.external_models import setup_external_models
import dotenv

# endregion


# Hydra configuration parameters
# Use absolute path to configs directory within the package
_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": str(
        Path(__file__).parent.parent / "configs" / "lightning_train"
    ),
    "config_name": "config.yaml",
}

# region: other global constants and functions
dotenv.load_dotenv(
    override=True
)  # set env variables from .env file, override=True is important
found_secretes = dotenv.load_dotenv(".secrets.env", override=True)
if not found_secretes:
    print("Warning: .secrets.env not found")
# Register a temporary resolver so that early calls to resolve() don't fail
omegaconf_resolvers.register_resolvers()
OmegaConf.register_new_resolver(
    "datamodule", lambda attr: "${datamodule:" + str(attr) + "}"
)
OmegaConf.register_new_resolver(
    "tokenizer", lambda attr: "${tokenizer:" + str(attr) + "}"
)
OmegaConf.register_new_resolver(
    "lightning_module", lambda attr: "${lightning_module:" + str(attr) + "}"
)
OmegaConf.register_new_resolver(
    "global_components", lambda attr: "${global_components:" + str(attr) + "}"
)
# endregion

# Setup external models before Hydra initialization
# This adds external model directories to sys.path for imports
external_model_dirs = setup_external_models()

# Register our SearchPathPlugin manually with Hydra
if external_model_dirs:
    from hydra.core.plugins import Plugins
    from hydra.plugins.search_path_plugin import SearchPathPlugin
    from hydra.core.config_search_path import ConfigSearchPath

    class ExternalModelsSearchPathPlugin(SearchPathPlugin):
        def manipulate_search_path(
            self, search_path: ConfigSearchPath
        ) -> None:
            for model_dir in external_model_dirs:
                config_dir = model_dir / "configs"
                if config_dir.exists():
                    search_path.append("file", str(config_dir))

    # Register the plugin
    Plugins.instance().register(ExternalModelsSearchPathPlugin)


@hydra.main(**_HYDRA_PARAMS)
def main(cfg: DictConfig) -> None:
    # Print predictor config as JSON and exit if requested
    if getattr(cfg, "job_type", None) == "print_predictor_params":
        out_path = getattr(cfg, "predictor_json_out", None)
        data = json.dumps(
            OmegaConf.to_container(cfg.get("predictor", {}), resolve=True)
        )
        if out_path:
            with open(out_path, "w") as f:
                f.write(data)
        else:
            print(data)
        return
    # INFO: delay the import until after main() to help with hydra auto-completion speed
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    from xlm.utils.rich_utils import print_config_tree

    if cfg.job_type == "name":
        job_name = cfg.job_name

        print_config_tree(cfg, resolve=True, save_to_file=False)
        print(f"job_name: \t{job_name}")
        return

    if cfg.job_type == "length_analysis":
        from xlm.commands.length_analysis import length_analysis

        # print_config_tree(cfg, resolve=True, save_to_file=False)
        length_analysis(cfg)
        return

    from xlm.utils.debug import set_flags
    from xlm.commands.lightning_train import train
    from xlm.commands.lightning_eval import evaluate
    from xlm.commands.lightning_generate import generate
    from xlm.commands.lightning_prepare_data import prepare_data

    if cfg.job_type == "train":
        import multiprocessing as mp

        # at some point I started getting worker error
        # I tried the workaround mentioned here: https://github.com/pytorch/pytorch/issues/119845 buy setting mp.set_start_method("spawn")
        # if the issue reappears, try uncommenting the line below
        # mp.set_start_method("spawn")
        print_config_tree(cfg, resolve=True, save_to_file=cfg.paths.run_dir)
        set_flags(cfg)
        print_slurm_info()
        train(cfg)
    elif cfg.job_type == "eval":
        set_flags(cfg)
        evaluate(cfg)
    elif cfg.job_type == "generate":
        generate(cfg)
    elif cfg.job_type == "prepare_data":
        print_config_tree(cfg, resolve=True)
        prepare_data(cfg)


if __name__ == "__main__":
    main()
