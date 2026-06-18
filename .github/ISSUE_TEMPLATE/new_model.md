---
name: New model
about: Propose a new language model family for XLM
title: "[Model] ..."
labels: model
assignees: ''

---

**Model name**
<!-- Short name (e.g. `my_nar_lm`) -->

**Paper / reference**
<!-- arXiv link, OpenReview, project page, or brief description if no paper -->

**Where will it live?**

- [ ] Maintained in `xlm-models/` (this repo)
- [ ] External repository (separate package)

**Architecture summary**
<!-- How does generation differ from existing families (ARLM, ILM, MDLM, MLM, …)? -->

**Components**
<!-- Which of the four pieces need non-standard behavior? -->

- [ ] Model (forward pass)
- [ ] Loss
- [ ] Predictor
- [ ] Collator / datamodule

**Dependencies**
<!-- Any new optional extras beyond core xlm-core? -->

**Checkpoints**
<!-- Are there existing checkpoints that your code will allow loading or will you be providing checkpoints for your model? -->

- [ ] Existing checkpoints
- [ ] New checkpoints

