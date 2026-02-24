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

   

6. When ready create a PR on for the main branch and add `dhruvdcoder, brozonoyer, sensai99, Durga-Prasad1` as a reviewers.


