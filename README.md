# Synthetic Data Flywheel

> Built autonomously using **[NEO - Your Autonomous AI Engineering Agent](https://heyneo.com)**

[![VS Code Extension](https://img.shields.io/visual-studio-marketplace/v/NeoResearchInc.heyneo?label=VS%20Code&logo=visualstudiocode&logoColor=white&color=0078d4)](https://marketplace.visualstudio.com/items?itemName=NeoResearchInc.heyneo) [![Cursor Extension](https://img.shields.io/badge/Cursor-Extension-black?logo=cursor&logoColor=white)](https://marketplace.cursorapi.com/items/?itemName=NeoResearchInc.heyneo)

---

A closed-loop pipeline for generating, filtering, labeling, and exporting synthetic instruction-tuning data - with an LLM-as-judge filter, statistical judge calibration, and an A2A-protocol agent surface for multi-agent orchestration.

You bring seed prompts (or an existing dataset). The flywheel generates candidate pairs, scores them with a local or remote judge, calibrates the judge against human labels, exports clean training data, and optionally hands the filtered set to Unsloth for fine-tuning on a free Colab GPU. Failure cases from one cycle become seeds for the next.

Runs on CPU for everything except the optional training step.

---

## Install

Python >= 3.11.

```
git clone https://github.com/dakshjain-1616/synthetic-data-flywheel
cd synthetic-data-flywheel
pip install -e .
```

Installs the `flywheel` CLI plus the library under `synthetic_data_flywheel`.

---

## Quickstart

The quickest end-to-end run uses a small existing JSONL of pairs and a local Ollama judge.

1. Start Ollama and pull a judge model (Gemma 4 is the reference):

   ```
   ollama pull gemma4
   ```

2. Initialize directories and ingest a dataset:

   ```
   $ flywheel init
   Synthetic Data Flywheel Initialized
   Data Directory: ./data
   Checkpoint Directory: ./data/checkpoints
   Report Directory: ./reports
   Directories created successfully

   $ flywheel ingest -i demo.jsonl -n demo --tag demo1
   Ingested 8 pairs -> data/user/demo.jsonl
   ```

3. Validate, judge with Ollama, auto-label, compare, calibrate, visualize:

   ```
   flywheel validate -d demo --checks schema,length,dedup,pii --write-clean data/user/demo.clean.jsonl
   flywheel judge    -d demo --backend ollama --model gemma4:latest --tag v1
   flywheel label    -d demo --mode auto-from-judge --judgments data/judgments/demo.v1.jsonl
   flywheel calibrate -d demo --tag v1
   flywheel visualize -d demo
   ```

4. Export a train/val split of the pairs that passed the judge:

   ```
   flywheel dataset export demo \
     --to data/exports/demo.jsonl \
     --judgments data/judgments/demo.v1.jsonl \
     --filter "scores['overall'] >= 7" \
     --split train=0.8,val=0.2
   ```

All of the above run on CPU. Only the optional fine-tuning notebook (`notebooks/training_cycle_*.ipynb`) requires a GPU - see Limitations below.

---

## CLI reference

`flywheel --help` lists the command groups. Every command has `--help` with full flag docs.

```
$ flywheel --help
Usage: flywheel [OPTIONS] COMMAND [ARGS]...

  Synthetic Data Flywheel - Autonomous data generation pipeline.

Commands:
  calibrate  Measure judge 'passed' against human labels (precision/recall/F1).
  compare    Compare two+ judgment runs (Cohen's kappa, agreement, ...).
  dataset    Dataset management: ls | info | export.
  ingest     Ingest a user dataset into the flywheel's JSONL format.
  init       Initialize flywheel configuration.
  judge      Judge a dataset with an LLM-as-judge backend.
  label      Label a dataset: interactive/bulk/auto-from-judge.
  pipeline   Run declarative YAML pipelines.
  report     Generate HTML report from checkpoints.
  run        Run the synthetic data flywheel.
  status     Show current flywheel status.
  validate   Validate a dataset and write a ValidationReport.
  visualize  Render a suite of PNG charts + index.html for a dataset.
```

### init

Creates `./data`, `./data/checkpoints`, `./reports`.

### ingest

Load a dataset of `(instruction, [input,] output)` triples into the flywheel format.

```
flywheel ingest -i path/to/file.jsonl -n my_dataset
flywheel ingest -i data.csv           -n my_dataset -f csv
flywheel ingest -i hf://tatsu-lab/alpaca -n alpaca --limit 500 --hf-split train
flywheel ingest -i data.jsonl -n aliased --map "instruction=prompt,output=completion"
flywheel ingest -i data.jsonl -n x --dry-run
```

Writes `data/user/<name>.jsonl` plus `data/user/<name>.meta.json`.

### validate

Runs deterministic checks over a dataset: `schema`, `length`, `dedup`, `pii`, `lang`, `profanity`. Produces a JSON report and optionally a cleaned copy.

```
$ flywheel validate -d demo --checks schema,length,dedup,pii --write-clean data/user/demo.clean.jsonl
      Validation: demo
  Total pairs       8
  pii               1
  severity:warning  1
Report: data/validation/demo.report.json
Clean dataset written (8 pairs): data/user/demo.clean.jsonl
```

`--fail-on error|warning|never` lets you gate CI on issues.

### judge

LLM-as-judge scoring with a pluggable backend (`ollama`, `openrouter`, `anthropic`). Reads the rubric from `rubrics/default.yaml` if present, else the built-in default.

```
$ flywheel judge -d demo --backend ollama --model gemma4:latest --tag v1 --max-pairs 3
Judging 3 pairs with ollama:gemma4:latest
  Judged                3
  Passed                0 (0.0%)
  Avg overall (scored)  5.00
  Output                data/judgments/demo.v1.jsonl
  Cache                 hits=0 misses=3 writes=3
```

Flags: `--concurrency`, `--max-pairs`, `--sample 0.2`, `--no-cache`, `--rubric path/to/rubric.yaml`, `--tag v2`. Judgments are cached on disk keyed by `(backend, model, pair.id, rubric.name@version)`. Judge timeouts are governed by `JUDGE_TIMEOUT` (default 600s - large local models can take over two minutes on first call).

### label

Three modes:

```
flywheel label -d demo --mode interactive
flywheel label -d demo --mode bulk --where "output != ''" --set-status approved --tag ok
flywheel label -d demo --mode auto-from-judge --judgments data/judgments/demo.v1.jsonl --reject-below 3.5
```

Labels are stored append-only as `data/labels/<dataset>.jsonl` (latest-wins per pair).

### compare

Two or more judgment runs, same dataset - reports pass-agreement, Cohen's kappa, and Pearson correlation on the overall score.

```
$ flywheel compare -d demo --tags judge_a,judge_b
        Judge comparison: judge_a vs judge_b
  Common pairs          8
  judge_a passed / mean 6 / 7.44
  judge_b passed / mean 6 / 7.19
  Pass agreement        100.0%
  Cohen's kappa (p/f)   1.000  (near-perfect)
  Score Pearson r       0.965
  Output                reports/demo/compare.json
```

### calibrate

Treats human labels (`status == approved`) as ground truth and measures the judge's precision / recall / F1 / accuracy.

```
$ flywheel calibrate -d demo --tag judge_a --approved-is approved
  Evaluated pairs  8
  Precision        1.000
  Recall           0.750
  F1               0.857
  Accuracy         0.750
  TP/FP/TN/FN      6/0/0/2
```

### visualize

Renders a suite of PNG charts and an `index.html` for a dataset: label distribution, score distributions, pass/fail, lengths, categories, judge agreement matrix, validation breakdown.

```
$ flywheel visualize -d demo
  categories      reports/demo/categories.png
  lengths         reports/demo/lengths.png
  validation      reports/demo/validation.png
  pass_fail       reports/demo/pass_fail.png
  scores          reports/demo/scores.png
  criteria        reports/demo/criteria.png
  labels          reports/demo/labels.png
  judge_agreement reports/demo/judge_agreement.png
  index.html      reports/demo/index.html
```

### dataset ls | info | export

```
$ flywheel dataset ls
  name   pairs  source  tags
  demo   8      jsonl   demo1

$ flywheel dataset info demo
  pairs       data/user/demo.jsonl               present
  meta        data/user/demo.meta.json           present
  validation  data/validation/demo.report.json   present
  labels      data/labels/demo.jsonl             present
  judgments   data/judgments                     5 set(s)

$ flywheel dataset export demo \
    --to data/exports/demo.jsonl \
    --format jsonl \
    --judgments data/judgments/demo.judge_a.jsonl \
    --filter "scores['overall'] >= 7" \
    --split train=0.8,val=0.2
Wrote 4 pairs -> data/exports/demo.train.jsonl
Wrote 2 pairs -> data/exports/demo.val.jsonl
```

The `--filter` expression uses a safe evaluator - only arithmetic, comparisons, and subscript access into the context dict. Keys: `id, instruction, input, output, category, difficulty, metadata, scores, passed, judge_model, label`. Attribute access (`pair.output`) and function calls are rejected.

### pipeline run

Run ingest -> validate -> judge -> label -> export in a single YAML-described flow:

```yaml
# pipeline_demo.yaml
dataset: demo
steps:
  - validate:
      checks: [schema, length, dedup]
  - export:
      to: data/user/demo_pipeline.jsonl
      format: jsonl
```

```
$ flywheel pipeline run pipeline_demo.yaml
[1/2] flywheel validate -d demo --checks schema,length,dedup
[2/2] flywheel dataset export demo --to data/user/demo_pipeline.jsonl --format jsonl
   Pipeline: demo
  1  validate  ok  0
  2  export    ok  0
```

Steps dispatch through the same Click commands as manual runs - behavior is identical.

### run, status, report

`flywheel run` is the autonomous seeds-to-checkpoint loop. Generation goes through OpenRouter (`OPENROUTER_API_KEY` must be set); the judge stage uses a sync Ollama client (hardcoded in `engine.create_judge`). If Ollama is not running, generation still succeeds and pairs are saved in the checkpoint — they'll just all get `passed=false, reasoning="Judgment failed"`. `status` summarizes checkpoint state; `report` produces an HTML report across cycles.

Real captured run (1 cycle, 2 seeds, `meta-llama/llama-3.2-3b-instruct` as generator; no Ollama running, so judge errors gracefully):

```
$ export OPENROUTER_API_KEY=sk-or-...
$ export OPENROUTER_MODEL=meta-llama/llama-3.2-3b-instruct
$ flywheel run -s "benefits of green tea,history of python language" --max-cycles 1
╭───── Configuration ─────╮
│ Synthetic Data Flywheel │
│ Seeds: 2                │
│ Max Cycles: 1           │
╰─────────────────────────╯
Starting Flywheel with max_cycles=1
============================================================
Starting Cycle 1
============================================================
Using 2 seeds
Generating synthetic data...
Generated 2 pairs
Judging quality...
Passed: 0, Failed: 2            # Ollama not running → judge fallback
Cycle 1 complete. Pass rate: 0.00%
Flywheel complete. Ran 1 cycles.
       Flywheel Summary
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
┃ Metric             ┃ Value ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━┩
│ Total Cycles       │ 1     │
│ Total Passed Pairs │ 0     │
│ Avg Pass Rate      │ 0.00% │
└────────────────────┴───────┘
```

The OpenRouter-generated pair is saved verbatim inside `data/checkpoints/checkpoint_001.json`, e.g.:

```json
{
  "instruction": "benefits of green tea",
  "output": "Here is an example of an instruction-following training data in JSON format:\n\n{\n  \"instruction\": \"What are some of the benefits of drinking green tea?\",\n  \"output\": \"Green tea has numerous benefits, including: - High antioxidant content - Anti-inflammatory properties - May help with weight loss ...\",\n  \"category\": \"instruction\"\n}",
  "source_seed": "benefits of green tea"
}
```

If you have Ollama running locally you get non-zero pass rates end-to-end. If you don't, run the pipeline in two steps (`flywheel run` to generate, then `flywheel judge --backend openrouter` on the exported pairs) to get OpenRouter-judged quality scores without Ollama.

```
$ flywheel status
$ flywheel report
Report generated: reports/flywheel_report_20260423_171226.html
```

---

## Python API

Everything the CLI does is available as a library.

```python
from synthetic_data_flywheel import (
    SyntheticPair, JudgmentResult, QualityScores, CycleState, FlywheelConfig,
)
from synthetic_data_flywheel.ingest import load_dataset_jsonl, DatasetIngestor
from synthetic_data_flywheel.validator import Validator
from synthetic_data_flywheel.rubrics import default_rubric, load_rubric, render_prompt
from synthetic_data_flywheel.judge import AsyncQualityJudge
from synthetic_data_flywheel.judge_backends import get_backend
from synthetic_data_flywheel.judge_cache import JudgmentCache
from synthetic_data_flywheel.labeler import LabelStore, auto_from_judge, bulk_apply, SafeEval
from synthetic_data_flywheel.stats import cohens_kappa, pearson, prf
from synthetic_data_flywheel.evaluator import create_evaluator
from synthetic_data_flywheel.dataset_manager import create_dataset_manager
from synthetic_data_flywheel.engine import create_engine
from synthetic_data_flywheel.viz import load_inputs, render_all
```

Minimal end-to-end call:

```python
import asyncio
from pathlib import Path
from synthetic_data_flywheel.ingest import load_dataset_jsonl
from synthetic_data_flywheel.rubrics import default_rubric
from synthetic_data_flywheel.judge import AsyncQualityJudge
from synthetic_data_flywheel.judge_backends import get_backend
from synthetic_data_flywheel.judge_cache import JudgmentCache

pairs = load_dataset_jsonl("data/user/demo.jsonl")
backend = get_backend("ollama", model="gemma4:latest")
judge = AsyncQualityJudge(
    backend=backend,
    rubric=default_rubric(),
    cache=JudgmentCache(root=Path(".cache/judge")),
    backend_name="ollama",
)
judgments = asyncio.run(judge.judge_batch(pairs, concurrency=2))
print(sum(j.passed for j in judgments), "/", len(judgments), "passed")
```

Statistics you can call directly:

```python
>>> cohens_kappa([True, False, True, False], [True, True, True, False])
0.5
>>> pearson([1,2,3,4], [1,3,2,5])
0.8315...
>>> prf([True,True,False,False], [True,False,True,False])
{'precision': 0.5, 'recall': 0.5, 'f1': 0.5, 'accuracy': 0.5, 'tp': 1, 'fp': 1, 'tn': 1, 'fn': 1}
```

---

## A2A agent

`synthetic_data_flywheel.a2a_agent` exposes a FastAPI application implementing the A2A protocol surface (`/a2a/capabilities`, `/a2a/tasks/send`, `/a2a/tasks/get`, `/a2a/tasks/cancel`). This lets the flywheel be orchestrated as a node in a multi-agent ML pipeline.

Run the server:

```
python -m synthetic_data_flywheel.a2a_agent
# or: uvicorn synthetic_data_flywheel.a2a_agent:app --host 0.0.0.0 --port 8080
```

Example request:

```python
from fastapi.testclient import TestClient
from synthetic_data_flywheel.a2a_agent import app

client = TestClient(app)
print(client.get("/a2a/capabilities").json())
# {'agent_name': 'synthetic_data_flywheel', 'version': '0.1.0',
#  'capabilities': [{'name': 'generate_synthetic_data', ...},
#                   {'name': 'get_status', ...},
#                   {'name': 'generate_report', ...}]}

r = client.post("/a2a/tasks/send", json={
    "capability": "get_status",
    "inputs": [],
    "parameters": {},
})
print(r.json())
# {'task_id': '...', 'status': {'state': 'completed'},
#  'result': {'type': 'status_result',
#             'content': {'checkpoints_found': 1, 'checkpoint_dir': 'data/checkpoints'}}, ...}
```

---

## Architecture

```
                        seeds / existing dataset
                                  |
                                  v
                 +----------------+-----------------+
                 |                                  |
                 v                                  v
          generator (OpenRouter)            ingest (jsonl/csv/hf)
                 |                                  |
                 +---------------+------------------+
                                 v
                           SyntheticPair[]
                                 |
                                 v
                     validator  (schema/dedup/pii/...)
                                 |
                                 v
                 judge (Ollama | OpenRouter | Anthropic)
                           + JudgmentCache
                                 |
                                 v
                       JudgmentResult[]  -----> compare / calibrate
                                 |
                                 v
                     labeler (interactive / bulk / auto)
                                 |
                                 v
                        LabelStore (JSONL)
                                 |
                                 v
                     export + split (train/val/test)
                                 |
                                 v
                   trainer notebook (Unsloth, Colab GPU)
                                 |
                                 v
                         evaluator -> failure seeds
                                 |
                                 +--> fed back into next cycle

              -------  viz / report -> PNGs + HTML  -------
              -------  A2A agent    -> FastAPI    --------
```

Module map (`src/synthetic_data_flywheel/`):

- `models.py` - Pydantic types: `SyntheticPair`, `JudgmentResult`, `QualityScores`, `CycleState`, `FlywheelConfig`, `Label`, `ValidationReport`, `DatasetMeta`.
- `config.py` - `Settings` via pydantic-settings; reads `.env`.
- `ingest.py` - `DatasetIngestor`, `load_dataset_jsonl`, `normalize_row`. Supports jsonl, json, csv, and `hf://<repo-id>`.
- `validator.py` - `Validator` plus `check_schema/length/dedup/pii/lang/profanity`.
- `rubrics.py` - `Rubric`, `Criterion`, `load_rubric`, `render_prompt`, `default_rubric`.
- `judge.py` - `AsyncQualityJudge` (used by CLI) + legacy sync `QualityJudge` / `OllamaClient`.
- `judge_backends/` - `ollama.py`, `openrouter.py`, `anthropic.py`, plus `registry.get_backend(name)`.
- `judge_cache.py` - `JudgmentCache` (file-backed, sharded by pair id).
- `labeler.py` - `SafeEval`, `LabelStore`, `bulk_apply`, `auto_from_judge`, `interactive_loop`.
- `stats.py` - `cohens_kappa`, `pearson`, `prf`.
- `evaluator.py` - `Evaluator.evaluate_judgments`.
- `generator.py` - `OpenRouterClient` with prompt templates (`QA`, `INSTRUCTION`, `REASONING`, `CREATIVE`).
- `engine.py` - `FlywheelEngine` (cycle loop, checkpointing, failure-seed feedback).
- `trainer.py` - `Trainer.prepare_training_artifacts` (writes a Colab-ready Unsloth notebook).
- `dataset_manager.py` - `DatasetManager` (local save + HF push when a token is set).
- `report_generator.py` - HTML report across cycles.
- `viz.py` - Matplotlib charts for a single dataset.
- `pipeline.py` - Declarative YAML pipeline runner (dispatches through the same CLI).
- `a2a_agent.py` - FastAPI A2A server.
- `cli.py` - Click entry points.

---

## How it works

A single dataset moves through a series of additive stages, each producing artifacts keyed by the dataset name:

| Stage      | Reads                              | Writes                                   |
| ---------- | ---------------------------------- | ---------------------------------------- |
| ingest     | raw file / HF repo                 | `data/user/<name>.jsonl`, `.meta.json`   |
| validate   | pairs                              | `data/validation/<name>.report.json`     |
| judge      | pairs + rubric                     | `data/judgments/<name>.<tag>.jsonl`      |
| label      | pairs + optional judgments         | `data/labels/<name>.jsonl`               |
| compare    | 2+ judgment tags                   | `reports/<name>/compare.json`            |
| calibrate  | judgments + labels                 | `reports/<name>/calibrate.<tag>.json`    |
| visualize  | all of the above                   | `reports/<name>/*.png` + `index.html`    |
| export     | pairs + optional judgments/labels  | arbitrary path(s), split by ratio        |

Each stage is idempotent and re-runnable. The judge cache makes repeated judge passes free. The label store is append-only so labeling sessions can be interrupted and resumed.

The autonomous loop (`flywheel run`) repeats:

1. Generate candidate pairs from seeds via OpenRouter.
2. Judge them via Ollama.
3. Filter to the passing set.
4. Save a training artifact + a Colab-ready Unsloth notebook.
5. Extract the failure instructions and feed them as additional seeds for cycle N+1.
6. Checkpoint to `data/checkpoints/checkpoint_NNN.json`.

The cycle stops when pass rate drops below `min_pass_rate` (default 0.5) or `max_cycles` is reached.

---

## Configuration

All settings can be set via env vars or `.env` (see `.env.example`). The most common:

```
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_MODEL=qwen/qwen3-8b:free
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma4:latest
DEFAULT_JUDGE_BACKEND=ollama        # ollama | openrouter | anthropic
JUDGE_CONCURRENCY=4
JUDGE_TIMEOUT=600                    # seconds; bump for cold 8B+ local models
QUALITY_MIN_SCORE=7.0
MAX_CYCLES=10
PII_POLICY=warn                      # strict | warn | off
A2A_HOST=0.0.0.0
A2A_PORT=8080
```

---

## Tests

```
pytest
```

100 tests cover models, rubrics, validator, labeler, judge, judge backends, judge cache, ingest, viz, CLI, pipeline, dataset_manager, sharpness, and a full integration cycle.

---

## Limitations / what is gated behind external resources

- Fine-tuning (the "trainer" step) requires Unsloth + a GPU. `Trainer.prepare_training_artifacts` writes a Colab-ready notebook under `notebooks/training_cycle_NNN.ipynb`; you open it in Colab on a free T4, run all cells, and download the LoRA adapter. Running locally on CPU is not supported by Unsloth.
- `flywheel run` (the autonomous generation loop) requires `OPENROUTER_API_KEY` because generation goes through OpenRouter. The in-loop judge, however, is hardcoded to Ollama (`engine.create_judge` constructs a sync `QualityJudge` over `OllamaClient`) — if Ollama isn't available, generation still works and pairs are persisted, but every judgment falls back to `passed=false`. The standalone `flywheel judge --backend openrouter/anthropic` works fully without Ollama. Everything downstream - validate, label, compare, calibrate, visualize, export, pipeline, A2A - runs purely on local resources or on tiny CPU-only compute.
- Large local judges are slow to cold-start. Gemma 4 (9 GB) takes about 130 seconds the first time it is loaded into VRAM/RAM. The default `JUDGE_TIMEOUT` is 600s to cover this; bump further if your hardware is slower.
- Ingest from HuggingFace (`hf://<repo-id>`) requires the `datasets` package (already a dep) plus `HUGGINGFACE_TOKEN` for gated datasets.
- Anthropic judge backend requires `ANTHROPIC_API_KEY`.

---

## License

MIT. See `LICENSE`.
