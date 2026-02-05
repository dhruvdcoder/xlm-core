from pathlib import Path
from typing import Literal, Optional, Union, Any
from lightning.pytorch.loggers import WandbLogger
from lightning.fabric.utilities.types import _PATH
from omegaconf import DictConfig
import omegaconf
from xlm.utils.slurm import get_output_file_path
from xlm.utils.rank_zero import rank_zero_only


class MyWandbLogger(WandbLogger):
    def __init__(
        self,
        name: Optional[str] = None,
        save_dir: _PATH = ".",
        version: Optional[str] = None,
        offline: bool = False,
        dir: Optional[_PATH] = None,
        id: Optional[str] = None,
        anonymous: Optional[bool] = None,
        project: Optional[str] = None,
        log_model: Union[Literal["all"], bool] = False,
        experiment: Union["Run", "RunDisabled", None] = None,
        prefix: str = "",
        checkpoint_name: Optional[str] = None,
        hydra_cfg: Optional[DictConfig] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            save_dir=save_dir,
            version=version,
            offline=offline,
            dir=dir,
            id=id,
            anonymous=anonymous,
            project=project,
            log_model=log_model,
            experiment=experiment,
            prefix=prefix,
            checkpoint_name=checkpoint_name,
            **kwargs,
        )
        self.hydra_cfg = hydra_cfg
        slurm_output_file_path = get_output_file_path()
        if slurm_output_file_path:
            slurm_output_file_path = Path(slurm_output_file_path)
            dir_ = slurm_output_file_path.parent
            # name_ = slurm_output_file_path.name
            self.experiment.save(
                glob_str=str(slurm_output_file_path), base_path=str(dir_)
            )
        if hydra_cfg is not None:
            run_dir = hydra_cfg.paths.get("run_dir", None)
            if run_dir is not None:
                run_dir = Path(run_dir)
                if run_dir.exists():
                    self.experiment.save(
                        glob_str=f"{run_dir}/*",
                        base_path=str(run_dir.parent),
                    )
                    hydra_dir = run_dir / ".hydra"
                    if hydra_dir.exists():
                        self.experiment.save(
                            glob_str=f"{hydra_dir}/*",
                            base_path=str(run_dir.parent),
                        )

    # CLEANUP: We may not need this but can leave it for now as reference
    # pl_module.log_hyperparams() will send the hydra config as params by calling to_container() already
    @rank_zero_only
    def log_hyperparams(self, params) -> None:
        super().log_hyperparams(params)
        # if self.hydra_cfg is not None:
        #    hydra_cfg = omegaconf.OmegaConf.to_container(
        #        self.hydra_cfg, resolve=True
        #    )
        #    self.experiment.config.update(
        #        {"cfg": hydra_cfg}, allow_val_change=True
        #    )
