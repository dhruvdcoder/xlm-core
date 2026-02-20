# Custom Commands

Models can define custom commands that extend xLM's CLI by creating `configs/commands.yaml`:

```yaml
# configs/commands.yaml
my_custom_command: "my_awesome_model.commands.my_function"
preprocess_data: "my_awesome_model.commands.preprocess"
```

Usage:

```bash
xlm command=my_custom_command arg1=value1 arg2=value2
```

The command functions should accept an `omegaconf.DictConfig` parameter containing the Hydra configuration.
