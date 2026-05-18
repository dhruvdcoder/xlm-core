# Custom Commands

External models (directories or packages listed in `xlm_models.json` / `XLM_MODELS_PATH`) can extend the `xlm` CLI by adding `configs/commands.yaml` under the model root:

```yaml
# configs/commands.yaml
my_custom_command: my_awesome_model.commands.my_function
preprocess_data: my_awesome_model.commands.preprocess
```

Hydra must still compose a valid top-level `config` (model, datamodule, tags, …). The **job type name** must match a key in `commands.yaml`.

Usage:

```bash
xlm job_type=my_custom_command arg1=value1 arg2=value2
```

The command functions must accept an `omegaconf.DictConfig` containing the resolved configuration.

## Pattern: command-only `job_type` with a thin experiment overlay

Some custom commands never construct a trainer or datamodule: they only read a few fields from a dedicated Hydra group (paths, sizes, flags). A typical external model wires that as:

- `configs/commands.yaml` — maps `job_type` to a Python callable.
- `configs/<group>/default.yaml` — defaults for the group the command reads.
- `configs/experiment/<overlay>.yaml` — minimal `defaults` that pull in `<group>`, so operators add the overlay to the same `experiment=[...]` list they use for training.

Abstract example (names and keys are illustrative; real packages document the exact composition and overrides):

```bash
xlm job_type=export_checkpoint \
  experiment=[my_model, my_distributed_stack, export_overlay] \
  export_checkpoint.sharded_dir=/path/to/sharded_ckpt \
  export_checkpoint.output=/path/to/model.safetensors
```

You may still need extra Hydra overrides so the base configs compose, even when the command never touches data. Checkpoint layout, sharded export options, and how to point inference at a consolidated model are covered in [FSDP / LLM guide](./llms.md) and in each external model’s own docs.
