# Eval Module Design for xlm-core

## 1. Trace of the prd2 Eval Pipeline (from `README.md:52`)

The command being traced:

```bash
python src/dd/commands/main.py experiment=dream_math500_semi_ar \
  model_harness.predictor.max_new_tokens=512 \
  model_harness.predictor.diffusion_steps=128 \
  args.num_fewshot=1 debug=local args.limit=1 \
  job_name=local_debug model_harness.predictor.block_size=16
```

### 1.1 Entry Point: `src/dd/commands/main.py`

Hydra entry point with config root at `configs/evals/main.yaml`.

**What `_main(cfg)` does, step by step:**

1. **Setup**: accelerator, output dir, logging, wandb, seeds.
2. **TaskManager** (lm-eval): `hydra.utils.instantiate(cfg.task_manager)` → `lm_eval.tasks.TaskManager`.
3. **Task resolution**: `task_list = cfg.args.tasks.split(",")` then `task_manager.match_tasks(task_list)`.
4. **Model harness**: `hydra.utils.instantiate(cfg.model_harness)` → calls `Evaluator.manual_instantiate()`.
5. **Task dict**: `get_task_dict(task_names, task_manager)` → lm-eval loads the YAML task definition.
6. **Adjust configs**: Override `num_fewshot`, `repeats`, set seeds on each task object.
7. **Caching**: Optionally wraps model in `CachingLM` / `CustomCachingLM`.
8. **Evaluate**: Calls `lm_eval.evaluator.evaluate(lm=model_harness, task_dict=..., ...)`.
9. **Results**: Logs to wandb, saves samples/results, prints table.

### 1.2 Config Resolution for `experiment=dream_math500_semi_ar`

**`configs/evals/main.yaml`** (base config):
- Sets defaults: `model_harness: default`, `model_harness/model: llada_instruct`, etc.
- Searchpath includes `configs/common` (where `model_harness/` configs live).
- Defines: `args` (tasks, fewshot, limit, resume...), `eval_tracker`, `task_manager`.

**`configs/evals/experiment/dream_math500_semi_ar.yaml`** (experiment override):
```yaml
# @package _global_
defaults:
  - override /model_harness/model: dream
  - override /model_harness/tokenizer: dream
  - override /model_harness/predictor: semi_ar
  - override /task: math500
model_harness:
  apply_chat_template: false
  predictor:
    logits_hook:
      _target_: dd.diffusion.dream.modelling_dream.DreamLogitsHook
    add_bos_token: true
```

**Resolved component configs:**

| Component | Config file | `_target_` |
|---|---|---|
| model_harness | `configs/common/model_harness/default.yaml` | `dd.diffusion.base.lm_harness_eval.Evaluator.manual_instantiate` |
| model | `configs/common/model_harness/model/dream.yaml` | `dd.diffusion.dream.modelling_dream.DreamModel.from_pretrained` |
| tokenizer | `configs/common/model_harness/tokenizer/dream.yaml` | `dd.diffusion.dream.tokenization_dream.DreamTokenizer.from_pretrained` |
| predictor | `configs/common/model_harness/predictor/semi_ar.yaml` | `dd.diffusion.base.prediction.SemiARPredictor` |
| task | `configs/evals/task/math500.yaml` | sets `args.tasks=math500`, `args.num_fewshot=4`, points to custom tasks dir |

### 1.3 The Model Harness: `dd.diffusion.base.lm_harness_eval.Evaluator`

**File**: `src/dd/diffusion/base/lm_harness_eval.py`

This class extends `lm_eval.api.model.LM` and is the bridge between the diffusion model and lm-eval-harness.

**Key interface method — `generate_until(requests)`:**
```
for each batch of Instances:
    contexts, gen_args = unzip(req.arguments for req in batch)
    input_batch = self.predictor.prepare_batch_for_generation(contexts)
    decoded_results = self.predictor.generate(input_batch)
    responses = decoded_results["decoded"]  # List[str]
    # cache and collect responses
```

**What it does NOT implement** (raises `NotImplementedError`):
- `loglikelihood()` — not needed for generate_until tasks
- `loglikelihood_rolling()` — same

**`manual_instantiate()` classmethod** is called with `_recursive_=false` because it needs to:
1. Instantiate model, tokenizer, predictor separately
2. Pass model and tokenizer to predictor constructor
3. Return an `Evaluator(model, tokenizer, predictor, ...)`

### 1.4 The Predictor: `SemiARPredictor`

**File**: `src/dd/diffusion/base/prediction.py`

Inheritance: `SemiARPredictor → DefaultPredictor → _Predictor → Predictor(Protocol)`

**Key methods:**
- `prepare_batch_for_generation(prompts: List[str]) → Batch`: Tokenize, left-pad, append `[MASK]` tokens
- `generate(batch: Batch) → GenerationResult`: Iterative denoising loop over `diffusion_steps`
- `decode(step_results) → List[str]`: Decode token IDs to strings, optionally chopping at EOS

The `SemiARPredictor` uses block-based scheduling (`BlockAllowSchedule`) to unmask tokens in blocks of `block_size`, and uses a confidence-based selection within each block.

### 1.5 The Task: `math500` (lm-eval YAML task)

**File**: `src/dd/tasks/math500/math500.yaml`

```yaml
task: math500
dataset_path: HuggingFaceH4/MATH-500
process_docs: !function math_verify_utils.process_docs
output_type: generate_until
test_split: test
training_split: test              # fewshot examples come from test split
doc_to_text: "Problem: {{problem}}\nAnswer:"
doc_to_target: "{{answer}}"
num_fewshot: 4
fewshot_config:
  doc_to_target: "{{solution}}"   # fewshot examples show solutions
generation_kwargs:
  until: ["Problem:", "<|eot_id|>"]
  do_sample: false
  temperature: 0
filter_list:
  - filter:
      - function: custom
        filter_fn: !function math_verify_utils.extract_answer
    name: extract_answer
process_results: !function math_verify_utils.process_results
metric_list:
  - metric: math_equal_at_1
    aggregation: mean
    higher_is_better: true
```

**Supporting utils** (`math_verify_utils.py`):
- `process_docs(dataset)` → wraps answer in `$...$`
- `extract_answer(resps, docs)` → uses `math_verify.parse()` to extract answer from model response
- `process_results(doc, results)` → uses `math_verify.verify()` to check correctness → `{"math_equal_at_1": bool}`

### 1.6 lm-eval `evaluate()` Flow (the core loop we need to replicate)

**File**: `lm_eval/evaluator.py :: evaluate()`

```
Phase 1: BUILD REQUESTS
  for each task:
    task.build_all_requests(limit=..., ...)
      → for each doc in dataset:
          fewshot_ctx = task.fewshot_context(doc, num_fewshot, ...)
          instance = task.construct_requests(doc, ctx, ...)
            → For generate_until:
                arguments = (ctx, generation_kwargs)
                return Instance(request_type="generate_until", doc=doc, arguments=arguments, ...)

Phase 2: RUN MODEL
  for each request_type (just "generate_until" for us):
    clone requests for repeats
    resps = lm.generate_until(cloned_reqs)  → calls our Evaluator.generate_until()
    for resp, req in zip(resps, cloned_reqs):
      req.resps.append(resp)

Phase 3: POSTPROCESS
  for each task:
    task.apply_filters()  → runs extract_answer filter on raw resps → stored in req.filtered_resps
    for each doc:
      metrics = task.process_results(doc, [req.filtered_resps[filter_key] ...])
        → calls math_verify_utils.process_results → {"math_equal_at_1": True/False}
      task_output.sample_metrics[("math_equal_at_1", filter_key)].append(value)

Phase 4: AGGREGATE
  for each task:
    task_output.calculate_aggregate_metric(bootstrap_iters=...)
      → applies aggregation function (mean) over all sample_metrics
  consolidate_results(eval_tasks) → final results dict
```

---

## 2. What lm-eval-harness Provides (and what we use)

### Features we actually use:
1. **Dataset loading** (HuggingFace datasets)
2. **Fewshot context assembly** (sample N examples, format with Jinja2 templates)
3. **Instance/Request dispatch** (batch prompts → model → collect responses)
4. **Filters** (post-process raw model outputs, e.g., extract boxed answer)
5. **Metric computation** (per-sample `process_results` + aggregation like `mean`)
6. **Result logging** (save samples as JSONL)

### Features we DON'T use:
1. **Model registry** (`@register_model`) — we use Hydra instantiation
2. **Task registry / TaskManager** — we can load tasks from YAML directly
3. **`loglikelihood` / `multiple_choice` request types** — diffusion models only do `generate_until`
4. **Chat templates / multiturn fewshot** — not used for Dream base model
5. **Multi-GPU sharding** (padding_requests, accelerator) — handled separately
6. **Request caching** (sqlite-based) — we have our own caching in prd2
7. **Decontamination** — not used
8. **Complex group/subtask hierarchy** — we evaluate single tasks

---

## 3. Proposed Minimal Eval Module for xlm-core

### 3.1 Design Principles

1. **Hydra-native**: Use Hydra instantiation everywhere. No registration machinery.
2. **Single request type**: Only support `generate_until` (text generation).
3. **Minimal abstractions**: ~4 core classes, each < 200 lines.
4. **Reuse existing xlm-core Predictor protocol**: The `Predictor` in `xlm.harness` already has `predict()` and `generate()`. We add a thin `generate_from_prompts(List[str]) → List[str]` wrapper.
5. **Decouple task definition from framework**: Tasks are YAML/dataclass configs that describe dataset + prompt template + metric. No need for a base class hierarchy.

### 3.2 Core Components

```
src/xlm/eval/
├── __init__.py
├── task.py          # EvalTask: dataset loading, prompt formatting, fewshot
├── evaluator.py     # evaluate(): the main loop
├── filters.py       # answer extraction filters
├── metrics.py       # metric functions (math_equal, exact_match, etc.)
└── tasks/           # task YAML configs
    └── math500.yaml
```

#### A. `EvalTask` (replaces lm-eval's ConfigurableTask)

A simple dataclass-driven task that:
- Loads a HuggingFace dataset
- Formats prompts using Jinja2 templates (`doc_to_text`, `doc_to_target`)
- Assembles fewshot context by sampling from training/test split
- Yields `(prompt_str, doc)` pairs

```python
@dataclass
class EvalTaskConfig:
    task_name: str
    dataset_path: str
    dataset_name: Optional[str] = None
    test_split: str = "test"
    training_split: str = "test"    # for fewshot examples
    doc_to_text: str = ""           # Jinja2 template
    doc_to_target: str = ""         # Jinja2 template
    fewshot_doc_to_target: Optional[str] = None  # override for fewshot examples
    num_fewshot: int = 0
    target_delimiter: str = " "
    fewshot_delimiter: str = "\n\n"
    process_docs: Optional[Callable] = None       # dataset preprocessing
    generation_kwargs: Optional[dict] = None
    limit: Optional[int] = None

class EvalTask:
    def __init__(self, config: EvalTaskConfig): ...
    def load_dataset(self) -> datasets.Dataset: ...
    def format_prompt(self, doc: dict) -> str: ...
    def fewshot_context(self, doc: dict) -> str: ...
    def doc_iterator(self, limit=None) -> Iterator[Tuple[int, dict]]: ...
```

Key simplification: no `Instance` object. We just produce `List[str]` prompts and keep track of `doc` objects alongside.

#### B. `ResultProcessor` (replaces lm-eval's filter + process_results)

```python
@dataclass
class ResultProcessorConfig:
    extract_answer: Callable[[str], Any]    # raw_response → parsed_answer
    judge: Callable[[dict, Any], dict]      # (doc, parsed_answer) → {"metric_name": value}
    aggregation: Dict[str, Callable]        # {"metric_name": mean}
```

This is Hydra-instantiable. For math500:
```yaml
result_processor:
  _target_: xlm.eval.ResultProcessor
  extract_answer:
    _target_: xlm.eval.filters.math_verify_extract
  judge:
    _target_: xlm.eval.metrics.math_verify_judge
  aggregation:
    math_equal_at_1: mean
```

#### C. `evaluate()` (replaces lm-eval's evaluate function)

The main loop, ~100 lines:

```python
def evaluate(
    model: GenerativeModel,    # anything with generate_from_prompts(List[str]) → List[str]
    task: EvalTask,
    result_processor: ResultProcessor,
    batch_size: int = 1,
    limit: Optional[int] = None,
    log_samples: bool = True,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    # Phase 1: Build prompts
    prompts, docs = [], []
    for doc_id, doc in task.doc_iterator(limit=limit):
        prompt = task.fewshot_context(doc) + task.format_prompt(doc)
        prompts.append(prompt)
        docs.append(doc)

    # Phase 2: Generate (batched)
    all_responses = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        responses = model.generate_from_prompts(batch_prompts)
        all_responses.extend(responses)

    # Phase 3: Score
    per_sample_metrics = []
    for doc, response in zip(docs, all_responses):
        parsed = result_processor.extract_answer(response)
        metrics = result_processor.judge(doc, parsed)
        per_sample_metrics.append(metrics)

    # Phase 4: Aggregate
    aggregated = {}
    for metric_name, agg_fn in result_processor.aggregation.items():
        values = [m[metric_name] for m in per_sample_metrics]
        aggregated[metric_name] = agg_fn(values)

    # Phase 5: Log
    if log_samples and output_dir:
        save_samples(output_dir, docs, prompts, all_responses, per_sample_metrics)

    return {"results": aggregated, "samples": per_sample_metrics}
```

#### D. `GenerativeModel` protocol (bridge to xlm-core's Predictor)

```python
class GenerativeModel(Protocol):
    def generate_from_prompts(self, prompts: List[str]) -> List[str]: ...
```

We write a thin adapter that wraps the existing `Predictor`:
```python
class PredictorAdapter:
    def __init__(self, predictor: xlm.harness.Predictor, tokenizer, device="cuda"):
        self.predictor = predictor
        self.tokenizer = tokenizer
        self.device = device

    def generate_from_prompts(self, prompts: List[str]) -> List[str]:
        # tokenize prompts, create batch, call predictor.predict(batch)
        # decode and return list of strings
        ...
```

### 3.3 Hydra Config Structure

```
configs/eval/
├── main.yaml          # entry point config
├── task/
│   ├── math500.yaml
│   └── gsm8k.yaml
└── experiment/
    └── dream_math500.yaml
```

**`configs/eval/main.yaml`:**
```yaml
defaults:
  - _self_
  - task: math500
  - experiment: null

model:
  _target_: xlm.eval.PredictorAdapter
  predictor: ${predictor}      # from xlm-core's existing config
  tokenizer: ${tokenizer}

evaluator:
  batch_size: 1
  limit: null
  log_samples: true
  output_dir: ${paths.run_dir}/eval_results
```

**`configs/eval/experiment/dream_math500.yaml`:**
```yaml
# @package _global_
defaults:
  - override /task: math500

# Point to the Dream model/predictor
predictor:
  _target_: dream.predictor_dream.DreamPredictor
  diffusion_kwargs:
    max_new_tokens: 512
    diffusion_steps: 128
    block_size: 16
```

### 3.4 Comparison: lm-eval-harness vs xlm-core eval

| Aspect | lm-eval-harness | xlm-core eval (proposed) |
|---|---|---|
| Lines of code | ~5000 (task.py + evaluator.py + registry + filters + ...) | ~500 (4 files) |
| Model interface | `LM` base class with `loglikelihood`, `generate_until`, etc. | `GenerativeModel` protocol with single `generate_from_prompts` |
| Task definition | YAML + registration + ConfigurableTask (1800 lines) | YAML + EvalTaskConfig dataclass (~100 lines) |
| Request dispatch | `Instance` objects grouped by type | Simple `List[str]` prompts |
| Filters | `FilterEnsemble` with registry | Plain `Callable` |
| Metrics | Registry-based with aggregation functions | Dict of `Callable` |
| Fewshot | Complex sampler with caching | Simple random sampling |
| Multi-GPU | Built-in padding + gather | Not needed initially (single GPU) |
| Caching | SQLite-based request/response cache | Not needed initially |

### 3.5 Implementation Plan

**Phase 1 — Core (needed to validate the Dream port):**
1. `EvalTask` with dataset loading, Jinja2 prompts, fewshot assembly
2. `evaluate()` loop
3. `PredictorAdapter` wrapping xlm-core's Dream predictor
4. Math500 task config + `math_verify`-based metric
5. Entry point script

**Phase 2 — Polish:**
6. Add GSM8K task
7. Result saving (JSONL per-sample logs)
8. Wandb integration
9. CLI overrides for common params (limit, num_fewshot, etc.)

**Phase 3 — Optional:**
10. Caching (resume support)
11. Multi-GPU evaluation
12. More tasks (MMLU, HumanEval, etc.)

---

## 4. Key Decisions / Open Questions

1. **Should we support the lm-eval YAML task format?**
   - Pro: Can reuse existing task definitions (math500.yaml, gsm8k.yaml, etc.) from lm-eval's task library.
   - Con: Requires parsing `!function` tags, `filter_list`, etc.
   - **Recommendation**: No. Write our own simpler YAML format. The task definitions are small (< 30 lines each) and easy to port.

2. **Should the adapter wrap `Predictor.predict()` or `Predictor.generate()`?**
   - `predict()` expects a tokenized batch dict (from datamodule).
   - `generate()` on `_Predictor` (prd2) takes a `Batch` object.
   - **Recommendation**: The adapter should handle tokenization internally. Accept `List[str]` prompts, tokenize, call the appropriate method, decode and return `List[str]`.

3. **How to handle the `prepare_batch_for_generation` logic?**
   - In prd2, the `SemiARPredictor` handles left-padding + mask-token appending in `prepare_batch_for_generation()`.
   - In xlm-core, `DreamPredictor.predict()` expects a batch dict with `input_ids`.
   - **Decision**: The adapter needs to replicate the prompt→batch conversion. This is model-specific so it belongs in the adapter, not the eval framework.

4. **Fewshot from same split as test?**
   - math500 uses `training_split: test` (fewshot examples from the same test set).
   - Need to ensure we don't use the current test example as a fewshot example.
   - lm-eval handles this via `ContextSampler` which excludes the current doc.
   - **Decision**: Implement simple exclusion in our fewshot sampler.

---

## 5. Native Proposal: Reuse the Existing `post_hoc_evaluator` Pipeline

Instead of building a standalone eval module (Section 3), we can reuse xlm-core's
existing `post_hoc_evaluator` infrastructure in `Harness`. This has two major
advantages:

1. **Evals run during training with the exact same execution path** as
   checkpoint-then-evaluate. No divergence between training-time validation
   metrics and standalone evaluation.
2. **No new components in the library** — only new evaluator classes and
   dataset preprocessing functions, following the same patterns already
   established by molgen (`DeNovoEval`, `FragmentEval`,
   `genmol_fragment_preprocess_fn`).

### 5.1 How the Existing Pipeline Works

The current flow (used by molgen):

```
on_validation_batch_end / on_test_batch_end
  └─ for each "prediction" dataloader:
       predictor.predict(batch) → preds
       predictor.to_dict(batch, preds) → List[Dict]
       log_predictions writes per-batch JSONL

on_validation_epoch_end / on_test_epoch_end
  └─ for each "prediction" dataloader:
       compute_post_hoc_metrics(split, dataloader_name, epoch, step)
         1. log_predictions.read() → all predictions from JSONL
         2. post_hoc_evaluator.eval(predictions, tokenizer) → (predictions, metrics)
         3. self.log() each metric
         4. (optionally) update_predictions writes metrics back to JSONL
```

Key interface: `post_hoc_evaluator.eval(predictions, tokenizer)` returns
`(updated_predictions, aggregated_metrics)`. This is generic enough for any
evaluation task.

### 5.2 Required Changes

#### A. `CompositePostHocEvaluator` — routing wrapper

Currently `self.post_hoc_evaluator` is a single instance. With multiple eval
datasets (math500, gsm8k, arc, etc.), we need different evaluators per
dataloader.

```python
class CompositePostHocEvaluator:
    """Routes to task-specific evaluators based on dataloader name.

    Config example:
        post_hoc_evaluator:
          _target_: xlm.eval.CompositePostHocEvaluator
          evaluators:
            math500_prediction:
              _target_: xlm.eval.Math500Eval
            gsm8k_prediction:
              _target_: xlm.eval.GSM8KEval
            denovo_prediction:
              _target_: xlm.tasks.molgen.DeNovoEval
              use_bracket_safe: true
    """
    def __init__(self, evaluators: Dict[str, Any]):
        self.evaluators = evaluators  # dataloader_name -> evaluator

    def eval(self, predictions, tokenizer=None, dataloader_name=None):
        for pattern, evaluator in self.evaluators.items():
            if pattern in (dataloader_name or ""):
                return evaluator.eval(predictions, tokenizer=tokenizer)
        return predictions, {}
```

**Harness change**: one line — pass `dataloader_name` through to `eval()`:

```python
# harness.py, compute_post_hoc_metrics
predictions, aggregated_metrics = self.post_hoc_evaluator.eval(
    predictions, tokenizer=self.tokenizer, dataloader_name=dataloader_name
)
```

The existing `DeNovoEval` / `FragmentEval` classes already accept `**kwargs`,
so adding `dataloader_name` to `eval()` is backward-compatible.

#### B. Separate results output (instead of overwriting predictions JSONL)

The current `update_logged_predictions` (harness.py:1404-1413) writes metrics
back into the same predictions JSONL. For LLM eval, we want a separate results
file with the question, generated answer, extracted answer, correctness, and
aggregated metrics.

Approach: `compute_post_hoc_metrics` writes a separate results JSON to
`{predictions_dir}/{split}/{dataloader_name}/results_{epoch}_{step}.json`
containing both per-sample results and aggregated metrics. The original
predictions JSONL is left untouched. This is ~15 lines of additive code in
`compute_post_hoc_metrics`.

#### C. Task-specific post-hoc evaluator classes

Each eval task is a class following the `DeNovoEval.eval()` pattern:

```python
class Math500Eval:
    """Post-hoc evaluator for Math500.

    Reads predictions with 'text' (generated) and 'truth' (gold answer),
    extracts and verifies answers using math_verify.
    """
    def eval(self, predictions, tokenizer=None, **kwargs):
        for pred in predictions:
            parsed = parse(pred["text"])
            gold = parse(pred["truth"])
            pred["correct"] = verify(gold, parsed)

        acc = sum(1 for p in predictions if p["correct"]) / len(predictions)
        return predictions, {"math_equal_at_1": acc}
```

#### D. Dataset preprocessing functions (prompt construction)

Just as `safe_bracket_on_the_fly_processor_combined` constructs molgen inputs,
we write task-specific preprocessing functions for LLM eval datasets:

```python
def math500_preprocess_fn(example, tokenizer, *, num_fewshot=4, fewshot_examples=None):
    """Construct prompt with fewshot context and tokenize.

    Returns dict with 'input_ids' (prompt tokens) for Dream's
    diffusion_generate, which pads with mask tokens and denoises.
    """
    prompt = ""
    for ex in fewshot_examples[:num_fewshot]:
        prompt += f"Problem: {ex['problem']}\nAnswer: {ex['solution']}\n\n"
    prompt += f"Problem: {example['problem']}\nAnswer:"

    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    return {"input_ids": input_ids, "answer": example["answer"]}
```

The `answer` field is carried through to the predictions JSONL via
`LogPredictions.additional_fields_from_batch`, so the post-hoc evaluator
can access it.

### 5.3 Multiple Predictions per Prompt

This is already a solved pattern in xlm-core. The fragment evaluation pipeline
demonstrates it:

1. **Preprocessing**: `replicate_examples` (in `xlm.datamodule`) repeats each
   example `num_samples` times contiguously in the dataset:
   ```python
   def replicate_examples(examples, tokenizer, num_samples):
       for key in examples:
           replicated = []
           for ex in examples[key]:
               for _ in range(num_samples):
                   replicated.append(deepcopy(ex))
           examples_replicated[key] = replicated
       return examples_replicated
   ```

2. **Evaluation**: `FragmentEval` (in `xlm.tasks.molgen`) groups predictions
   by their shared key before computing metrics:
   ```python
   @staticmethod
   def _split_into_groups(predictions):
       group_map = {}
       for pred in predictions:
           key = (pred.get("fragment_smiles"), pred.get("truth"))
           group_map.setdefault(key, []).append(pred)
       return list(group_map.values())
   ```
   Then per-group metrics (validity, uniqueness, diversity, quality, distance)
   are computed and averaged across groups.

For LLM eval, the same pattern applies: replicate each prompt `k` times in the
dataset, and have the post-hoc evaluator group by prompt (or by gold answer)
before computing pass@k or majority-vote accuracy.

### 5.4 Prompt-Conditioned Generation with Dream

Dream's `diffusion_generate` already supports prompt conditioning. It pads
`input_ids` with `mask_token_id` to `max_length`, then only denoises the
masked positions:

```python
# generation_dream.py:_sample
x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)
# ...
for i in range(steps):
    mask_index = (x == mask_token_id)
    # only generate at masked positions
```

So the preprocessing function provides the prompt as `input_ids`, and
`diffusion_generate` appends mask tokens and generates the continuation.
No changes to the model or predictor.

### 5.5 Summary: Effort Estimate

| Change | Effort | Touches existing code? |
|---|---|---|
| `CompositePostHocEvaluator` class | ~30 lines, new class | No |
| Pass `dataloader_name` to `eval()` | 1 line in `compute_post_hoc_metrics` | Yes, minimal |
| Separate results file output | ~15 lines in `compute_post_hoc_metrics` | Yes, additive |
| `Math500Eval` class | ~40 lines, new class | No |
| `math500_preprocess_fn` | ~30 lines, new function | No |
| Hydra configs (dataset + evaluator) | YAML files | No |

Total: ~120 lines of new code + YAML configs. The harness changes are two
small additions (pass `dataloader_name`, write results file).

### 5.6 Comparison: Native Proposal vs Standalone Eval Module (Section 3)

| Aspect | Standalone eval module (Section 3) | Native post_hoc_evaluator (this section) |
|---|---|---|
| New code | ~500 lines (4 new files) | ~120 lines (new classes + functions) |
| New abstractions | `EvalTask`, `ResultProcessor`, `evaluate()`, `GenerativeModel`, `PredictorAdapter` | `CompositePostHocEvaluator`, task-specific `Eval` classes |
| Training-time eval | Separate, must be wired in | Built-in (existing hooks) |
| Standalone eval | Dedicated entry point | `trainer.test()` or `trainer.predict()` |
| Multiple datasets | Managed by `evaluate()` | Multiple prediction dataloaders + composite evaluator |
| Multiple predictions/prompt | Must implement | Already solved (`replicate_examples` + grouping) |
| Config structure | New `configs/eval/` tree | Extends existing experiment configs |
| Model interface | New `GenerativeModel` protocol + adapter | Uses existing `Predictor` protocol directly |
