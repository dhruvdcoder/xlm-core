# Contributing to xLM

We value and appreciate community contributions of all kinds, including:

* Code contributions
* Answering questions
* Helping others
* Improving the documentation
* Reporting bugs
* Suggesting new features
* Spreading the word through blogs and social media or simply by ⭐️ing the repository


**This guide was adapted from existing open source projects:** 
1. [scikit-learn guide to contributing](https://github.com/scikit-learn/scikit-learn/blob/main/CONTRIBUTING.md).
2. [Hugging Face Transformers guide to contributing](https://huggingface.co/docs/transformers/en/contributing).


## Contributing to code or documentation
The best way to do that is to open a Pull Request and link it to the issue that you'd like to work on. 

### Just want to start contributing?
If you don't know where to start, there is a special [Good First Issue](https://github.com/dhruvdcoder/xlm-core/issues?q=state%3Aopen%20label%3A%22good%20first%20issue%22) listing. 


### Fixing outstanding issues

If you notice an issue with the existing code and have a fix in mind, feel free to create a new issue, [start a PR referencing the issue](#create-a-pull-request).

### Submitting an issue (bug or feature request)

Please follow these guidelines when submitting a bug-related issue or a feature
request. 
It will make it easier for us to come back to you quickly and with good
feedback.

### Reporting a bug 🐞

* Before you report an issue, we would really appreciate it if you could **make sure the bug was not already reported** (use the search bar on GitHub under Issues). 
* Do your best to make sure your issue is related to bugs in the library itself, and not your code. If you're unsure whether the bug is in your code or the library, please ask in the [Discussions](https://github.com/dhruvdcoder/xlm-core/discussions/categories/q-a). 


### Feature request 🚀

Please open an issue and follow the "Feature request" template.



## Contributing a new model 

We maintain a small library of models in a separate package called `xlm-models` in the same repository.

There are two ways to contribute a new model:

1. **Contribute an external model**: External models live in their own repository and are not part of the `xlm-models` package. This mechanism is great for reproducing models from research papers because it allows you considerable flexibility in terms of implementation as we don't require you to follow any strict structure. Please see [External Models Guide](https://dhruveshp.com/xlm-core/latest/guide/external-models/) for the detailed instructions.

2. **Contribute a model to the `xlm-models` package**: TODO: Add guide for this.



# Create a Pull Request

You will need basic `git` proficiency to contribute.

You'll need Python 3.11 or above to contribute to xLM. Follow the steps below to start contributing:

1. Fork the [repository](https://github.com/dhruvdcoder/xlm-core) by
   clicking on the **[Fork](https://github.com/dhruvdcoder/xlm-core/fork)** button on the repository's page. This creates a copy of the code
   under your GitHub user account.

2. Clone your fork to your local disk, and add the base repository as a remote:

   ```bash
   git clone git@github.com:dhruvdcoder/xlm-core.git
   cd xlm-core
   git remote add upstream https://github.com/dhruvdcoder/xlm-core.git
   ```

3. Create a new branch to hold your development changes:

   ```bash
   git checkout -b a-descriptive-name-for-my-changes
   ```

   🚨 **Do not** work on the `main` branch!

4. Set up a development environment by running the following command in a virtual environment:

   ```bash
   pip install -e .
   pip install -r requirements/dev_requirements.txt
   pip install -r requirements/test_requirements.txt
   pip install -r requirements/docs_requirements.txt
   pip install -r requirements/lint_requirements.txt
   


5. Develop the features in your branch.

   As you work on your code, you should make sure the test suite
   passes. Run the tests impacted by your changes like this:
   TODO: Add the commands to run the tests.


   TODO: Create a single makefile for formatting and style checks.

   ```bash
   make style
   ```

   ```bash
   make check-repo
   ```

   To learn more about those checks and how to fix any issues with them, check out the
   [Checks on a Pull Request](https://huggingface.co/docs/transformers/pr_checks) guide.

   If you're modifying documents under the `docs/source` directory, make sure the documentation can still be built. This check will also run in the CI when you open a pull request. To run a local check
   make sure you install the [documentation builder](https://github.com/huggingface/doc-builder).

   ```bash
   pip install hf-doc-builder
   ```

   Run the following command from the root of the repository:

   ```bash
   doc-builder build transformers docs/source/en --build_dir ~/tmp/test-build
   ```

   This will build the documentation in the `~/tmp/test-build` folder where you can inspect the generated
   Markdown files with your favorite editor. You can also preview the docs on GitHub when you open a pull request.

   Once you're happy with your changes, add the changed files with `git add` and
   record your changes locally with `git commit`:

   ```bash
   git add modified_file.py
   git commit
   ```

   Please remember to write [good commit
   messages](https://chris.beams.io/posts/git-commit/) to clearly communicate the changes you made!

   To keep your copy of the code up to date with the original
   repository, rebase your branch on `upstream/branch` *before* you open a pull request or if requested by a maintainer:

   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

   Push your changes to your branch:

   ```bash
   git push -u origin a-descriptive-name-for-my-changes
   ```

   If you've already opened a pull request, you'll need to force push with the `--force` flag. Otherwise, if the pull request hasn't been opened yet, you can just push your changes normally.

6. Now you can go to your fork of the repository on GitHub and click on **Pull Request** to open a pull request. Make sure you tick off all the boxes on our [checklist](#pull-request-checklist) below. When you're ready, you can send your changes to the project maintainers for review.

7. It's ok if maintainers request changes, it happens to our core contributors
   too! So everyone can see the changes in the pull request, work in your local
   branch and push the changes to your fork. They will automatically appear in
   the pull request.

### Pull request checklist

☐ The pull request title should summarize your contribution.
☐ If your pull request addresses an issue, please mention the issue number in the pull
request description to make sure they are linked (and people viewing the issue know you
are working on it).
☐ To indicate a work in progress please prefix the title with `[WIP]`. These are
useful to avoid duplicated work, and to differentiate it from PRs ready to be merged.
☐ Make sure existing tests pass.
☐ If adding a new feature, also add tests for it.

- If you are adding a new model, make sure you use
     `ModelTester.all_model_classes = (MyModel, MyModelWithLMHead,...)` to trigger the common tests.
- If you are adding new `@slow` tests, make sure they pass using
     `RUN_SLOW=1 python -m pytest tests/models/my_new_model/test_my_new_model.py`.
- If you are adding a new tokenizer, write tests and make sure
     `RUN_SLOW=1 python -m pytest tests/models/{your_model_name}/test_tokenization_{your_model_name}.py` passes.
- CircleCI does not run the slow tests, but GitHub Actions does every night!

☐ All public methods must have informative docstrings (see
[`modeling_bert.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py)
for an example).
☐ Due to the rapidly growing repository, don't add any images, videos and other
non-text files that'll significantly weigh down the repository. Instead, use a Hub
repository such as [`hf-internal-testing`](https://huggingface.co/hf-internal-testing)
to host these files and reference them by URL. We recommend placing documentation
related images in the following repository:
[huggingface/documentation-images](https://huggingface.co/datasets/huggingface/documentation-images).
You can open a PR on this dataset repository and ask a Hugging Face member to merge it.

For more information about the checks run on a pull request, take a look at our [Checks on a Pull Request](https://huggingface.co/docs/transformers/pr_checks) guide.

### Tests

An extensive test suite is included to test the library behavior and several examples. Library tests can be found in
the [tests](https://github.com/huggingface/transformers/tree/main/tests) folder and examples tests in the
[examples](https://github.com/huggingface/transformers/tree/main/examples) folder.

We like `pytest` and `pytest-xdist` because it's faster. From the root of the
repository, specify a *path to a subfolder or a test file* to run the test:

```bash
python -m pytest -n auto --dist=loadfile -s -v ./tests/models/my_new_model
```

Similarly, for the `examples` directory, specify a *path to a subfolder or test file* to run the test. For example, the following command tests the text classification subfolder in the PyTorch `examples` directory:

```bash
pip install -r examples/xxx/requirements.txt  # only needed the first time
python -m pytest -n auto --dist=loadfile -s -v ./examples/pytorch/text-classification
```

In fact, this is actually how our `make test` and `make test-examples` commands are implemented (not including the `pip install`)!

You can also specify a smaller set of tests in order to test only the feature
you're working on.

By default, slow tests are skipped but you can set the `RUN_SLOW` environment variable to
`yes` to run them. This will download many gigabytes of models so make sure you
have enough disk space, a good internet connection or a lot of patience!

Remember to specify a *path to a subfolder or a test file* to run the test. Otherwise, you'll run all the tests in the `tests` or `examples` folder, which will take a very long time!

```bash
RUN_SLOW=yes python -m pytest -n auto --dist=loadfile -s -v ./tests/models/my_new_model
RUN_SLOW=yes python -m pytest -n auto --dist=loadfile -s -v ./examples/pytorch/text-classification
```

Like the slow tests, there are other environment variables available which are not enabled by default during testing:

- `RUN_CUSTOM_TOKENIZERS`: Enables tests for custom tokenizers.

More environment variables and additional information can be found in the [testing_utils.py](https://github.com/huggingface/transformers/blob/main/src/transformers/testing_utils.py).

🤗 Transformers uses `pytest` as a test runner only. It doesn't use any
`pytest`-specific features in the test suite itself.

This means `unittest` is fully supported. Here's how to run tests with
`unittest`:

```bash
python -m unittest discover -s tests -t . -v
python -m unittest discover -s examples -t examples -v
```

### Style guide

For documentation strings, 🤗 Transformers follows the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).
Check our [documentation writing guide](https://github.com/huggingface/transformers/tree/main/docs#writing-documentation---specification)
for more information.

### Develop on Windows

On Windows (unless you're working in [Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/) or WSL), you need to configure git to transform Windows `CRLF` line endings to Linux `LF` line endings:

```bash
git config core.autocrlf input
```

One way to run the `make` command on Windows is with MSYS2:

1. [Download MSYS2](https://www.msys2.org/), and we assume it's installed in `C:\msys64`.
2. Open the command line `C:\msys64\msys2.exe` (it should be available from the **Start** menu).
3. Run in the shell: `pacman -Syu` and install `make` with `pacman -S make`.
4. Add `C:\msys64\usr\bin` to your PATH environment variable.

You can now use `make` from any terminal (PowerShell, cmd.exe, etc.)! 🎉

### Sync a forked repository with upstream main (the Hugging Face repository)

When updating the main branch of a forked repository, please follow these steps to avoid pinging the upstream repository which adds reference notes to each upstream PR, and sends unnecessary notifications to the developers involved in these PRs.

1. When possible, avoid syncing with the upstream using a branch and PR on the forked repository. Instead, merge directly into the forked main.
2. If a PR is absolutely necessary, use the following steps after checking out your branch:

   ```bash
   git checkout -b your-branch-for-syncing
   git pull --squash --no-commit upstream main
   git commit -m ''
   git push --set-upstream origin your-branch-for-syncing
   ```

