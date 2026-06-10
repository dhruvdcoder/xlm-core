---
name: Bug report
about: Create a report to help us improve
title: "[BUG] ..."
labels: bug
assignees: ''

---

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Minimal repro: an `xlm` command or Python script.

```bash
# Example
xlm job_type=train job_name=repro experiment=... debug=overfit
```

```python
# Or minimal Python
```

**Expected behavior**
What you expected to happen.

**Actual behavior**
What happened instead (traceback, wrong metrics, etc.).

**Relevant config**
Hydra experiment and overrides (paste or link):

```yaml
experiment=...
# +overrides
```

**Environment**

- OS:
- Python:
- xlm-core: <!-- python -c "import xlm; print(xlm.__version__)" -->
- xlm-models (if applicable):
- PyTorch:
- Lightning:
- Transformers:
- Hydra:

**Screenshots**
If applicable.

**Additional context**
Logs, checkpoint paths, or links to related issues.
