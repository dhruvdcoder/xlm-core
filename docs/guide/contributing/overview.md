# Ways to Contribute

XLM welcomes contributions of all kinds: code, documentation, bug reports, feature ideas, and helping others in [Discussions](https://github.com/dhruvdcoder/xlm-core/discussions).

Start with the [Contributing hub](https://github.com/dhruvdcoder/xlm-core/blob/main/CONTRIBUTING.md) for environment setup, testing, and pull request workflow. Then follow the guide that matches what you want to do:

| Contribution type | When to use it | Guide |
|-------------------|----------------|-------|
| **Add a maintained model** | Ship a new model family inside this repo under `xlm-models/` | [Adding a model](adding-a-model.md) |
| **Add an external model** | Develop and publish a model in a separate repo | [External models](../external-models.md) |
| **Add a task or dataset** | Preprocess function + dataset YAMLs in `src/xlm/` | [Adding a task or dataset](../adding-a-task.md) |
| **Run your model on your task** | External task + model wiring via Hydra YAML | [Running your model on your task](adding-a-task-external.md) |
| **Core framework development** | Change training infra in `src/xlm/` (harness, datamodule, metrics, CLI) | [Core development](core-development.md) |
| **Docs, bugs, and issues** | Fix docs, report bugs, or propose features without code changes | [Docs and issues](docs-and-issues.md) |

## Before you start

1. **Search existing [Issues](https://github.com/dhruvdcoder/xlm-core/issues)** — someone may already be working on it.
2. **Open an issue** for non-trivial work (use the matching issue template when available).
3. **Set up your environment** — follow [Install from source](https://github.com/dhruvdcoder/xlm-core/blob/main/CONTRIBUTING.md#install-from-source) in the Contributing hub. For what each requirements file contains, see [Dependencies](../../developers/dependencies.md).

## Good first issues

Looking for a small starter task? See [Good first issue](https://github.com/dhruvdcoder/xlm-core/issues?q=state%3Aopen+label%3A%22good+first+issue%22).

## Pull requests

Push your branch to your fork and open a PR against `main`. Use the [pull request template](https://github.com/dhruvdcoder/xlm-core/blob/main/.github/PULL_REQUEST_TEMPLATE.md) and link related issues.
