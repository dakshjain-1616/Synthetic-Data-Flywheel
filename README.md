# 🔄 Synthetic Data Flywheel

**Made Autonomously Using [NEO - Your Autonomous AI Engineering Agent](https://heyneo.com)**

[![VS Code Extension](https://img.shields.io/badge/VS%20Code-NEO%20Extension-blue?logo=visual-studio-code)](https://marketplace.visualstudio.com/items?itemName=NeoResearchInc.heyneo)
[![Cursor Extension](https://img.shields.io/badge/Cursor-NEO%20Extension-blue?logo=cursor)](https://marketplace.cursorapi.com/items/?itemName=NeoResearchInc.heyneo)

A closed-loop autonomous pipeline that generates synthetic training pairs from a base model, filters them using a local LLM judge, trains a new model, evaluates it, and feeds failure cases back as seeds for the next cycle — continuously improving without human labeling.

## 🏗️ Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Seeds     │────▶│  Generator  │────▶│   Judge     │
│  (Prompts)  │     │(OpenRouter) │     │  (Ollama)   │
└─────────────┘     └─────────────┘     └──────┬──────┘
       ▲                                     │
       │                                     ▼
       │                              ┌─────────────┐
       │                              │   Filter    │
       │                              └──────┬──────┘
       │                                     │
       │                                     ▼
       │                              ┌─────────────┐
       │                              │   Dataset   │
       │                              │   Manager   │
       │                              └──────┬──────┘
       │                                     │
       │                                     ▼
       │                              ┌─────────────┐
       │                              │   Trainer   │
       │                              │  (Unsloth)  │
       │                              └──────┬──────┘
       │                                     │
       │                                     ▼
       │                              ┌─────────────┐
       │                              │  Evaluator  │
       │                              └──────┬──────┘
       │                                     │
       └─────────────────────────────────────┘
              (Failure Seeds Feedback Loop)
```

### Flywheel Cycle Visualization

<svg width="100%" height="400" viewBox="0 0 800 400" xmlns="http://www.w3.org/2000/svg">
  <text x="400" y="25" font-size="20" font-weight="bold" text-anchor="middle" fill="#333">Synthetic Data Flywheel Cycle</text>
  
  <!-- Center point -->
  <circle cx="400" cy="200" r="140" fill="none" stroke="#ddd" stroke-width="2" stroke-dasharray="5,5"/>
  
  <!-- Step 1: Seeds (Top Left) -->
  <g>
    <circle cx="280" cy="100" r="35" fill="#4CAF50" stroke="#333" stroke-width="2"/>
    <text x="280" y="100" font-size="12" font-weight="bold" text-anchor="middle" fill="white">1. Seeds</text>
    <text x="280" y="115" font-size="10" text-anchor="middle" fill="white">(Prompts)</text>
  </g>
  
  <!-- Step 2: Generate (Top Right) -->
  <g>
    <circle cx="520" cy="100" r="35" fill="#2196F3" stroke="#333" stroke-width="2"/>
    <text x="520" y="100" font-size="12" font-weight="bold" text-anchor="middle" fill="white">2. Generate</text>
    <text x="520" y="115" font-size="10" text-anchor="middle" fill="white">(OpenRouter)</text>
  </g>
  
  <!-- Step 3: Judge (Right) -->
  <g>
    <circle cx="580" cy="200" r="35" fill="#FF9800" stroke="#333" stroke-width="2"/>
    <text x="580" y="200" font-size="12" font-weight="bold" text-anchor="middle" fill="white">3. Judge</text>
    <text x="580" y="215" font-size="10" text-anchor="middle" fill="white">(Ollama)</text>
  </g>
  
  <!-- Step 4: Filter (Bottom Right) -->
  <g>
    <circle cx="520" cy="300" r="35" fill="#9C27B0" stroke="#333" stroke-width="2"/>
    <text x="520" y="300" font-size="12" font-weight="bold" text-anchor="middle" fill="white">4. Filter</text>
    <text x="520" y="315" font-size="10" text-anchor="middle" fill="white">(Quality)</text>
  </g>
  
  <!-- Step 5: Train (Bottom Left) -->
  <g>
    <circle cx="280" cy="300" r="35" fill="#F44336" stroke="#333" stroke-width="2"/>
    <text x="280" y="300" font-size="12" font-weight="bold" text-anchor="middle" fill="white">5. Train</text>
    <text x="280" y="315" font-size="10" text-anchor="middle" fill="white">(Unsloth)</text>
  </g>
  
  <!-- Step 6: Evaluate (Left) -->
  <g>
    <circle cx="220" cy="200" r="35" fill="#00BCD4" stroke="#333" stroke-width="2"/>
    <text x="220" y="200" font-size="12" font-weight="bold" text-anchor="middle" fill="white">6. Evaluate</text>
    <text x="220" y="215" font-size="10" text-anchor="middle" fill="white">(Metrics)</text>
  </g>
  
  <!-- Arrows between steps (clockwise) -->
  <defs>
    <marker id="arrowhead" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
      <polygon points="0 0, 10 3, 0 6" fill="#333"/>
    </marker>
  </defs>
  
  <!-- 1 to 2 -->
  <line x1="310" y1="115" x2="490" y2="115" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)"/>
  
  <!-- 2 to 3 -->
  <path d="M 550 130 Q 580 160 580 170" stroke="#333" stroke-width="2" fill="none" marker-end="url(#arrowhead)"/>
  
  <!-- 3 to 4 -->
  <path d="M 560 230 Q 550 260 530 270" stroke="#333" stroke-width="2" fill="none" marker-end="url(#arrowhead)"/>
  
  <!-- 4 to 5 -->
  <line x1="490" y1="285" x2="310" y2="285" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)"/>
  
  <!-- 5 to 6 -->
  <path d="M 250 270 Q 220 240 220 230" stroke="#333" stroke-width="2" fill="none" marker-end="url(#arrowhead)"/>
  
  <!-- 6 to 1 -->
  <path d="M 240 170 Q 260 130 270 115" stroke="#333" stroke-width="2" fill="none" marker-end="url(#arrowhead)"/>
  
  <!-- Feedback loop (failures back to seeds) -->
  <path d="M 220 165 Q 150 150 150 100 Q 150 70 280 70" stroke="#E91E63" stroke-width="3" fill="none" stroke-dasharray="5,5" marker-end="url(#arrowhead-pink)"/>
  <text x="130" y="120" font-size="11" fill="#E91E63" font-weight="bold">Failure Seeds</text>
  <text x="130" y="135" font-size="10" fill="#E91E63">Feedback Loop</text>
  
  <!-- Marker for pink arrow -->
  <defs>
    <marker id="arrowhead-pink" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
      <polygon points="0 0, 10 3, 0 6" fill="#E91E63"/>
    </marker>
  </defs>
</svg>

## 📦 Installation

### Prerequisites

- Python 3.10+
- [OpenRouter API key](https://openrouter.ai/keys) for generation
- [Ollama](https://ollama.ai/) installed locally for judging

### Install from Source

```bash
git clone https://github.com/yourusername/synthetic-data-flywheel.git
cd synthetic-data-flywheel
pip install -e ".[dev]"
```

### Environment Setup

Copy the example environment file and fill in your API keys:

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
# OpenRouter (for generation)
OPENROUTER_API_KEY=your_openrouter_key_here
OPENROUTER_MODEL=qwen/qwen3-8b

# Ollama (for judging)
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=gemma4

# HuggingFace (optional, for dataset upload)
HF_TOKEN=your_hf_token_here
```

## 🚀 Quick Start

### 1. Initialize the Flywheel

```bash
flywheel init
```

### 2. Run the Flywheel

```bash
flywheel run --seeds "Explain quantum computing,Write a poem about AI,What is machine learning?"
```

### 3. Check Status

```bash
flywheel status
```

### 4. Generate Report

```bash
flywheel report
```

## 📖 Usage

### CLI Commands

```bash
# Run with custom settings
flywheel run \
  --seeds "seed1,seed2,seed3" \
  --max-cycles 5 \
  --checkpoint-dir ./my-checkpoints

# Resume from checkpoint
flywheel run --seeds "seed1" --resume

# Generate HTML report
flywheel report --output ./reports

# Show current status
flywheel status --checkpoint-dir ./checkpoints
```

### Data Platform Commands

Beyond the flywheel loop, `flywheel` ships CLI subcommands for a full
training-data workflow over *your own* data. The example below has been
verified end-to-end against a local Ollama (`gemma4:latest`).

```bash
# 0) One-time
flywheel init

# 1) Ingest any dataset (JSONL / JSON / CSV / HF) into the tool's format.
#    Exact duplicates collapse automatically (deterministic pair IDs).
flywheel ingest --input my.jsonl --name toy --format jsonl \
  --map instruction=prompt,output=completion

# 2) Validate: schema, length, dedup, PII, language, profanity.
#    --write-clean emits a filtered JSONL dropping errors + duplicates.
flywheel validate --dataset toy --checks schema,length,pii,dedup \
  --write-clean data/user/toy.clean.jsonl

# 3) LLM-as-judge with pluggable backends + external YAML rubric.
#    Backends: ollama | openrouter | anthropic.
flywheel judge --dataset toy --rubric rubrics/default.yaml \
  --backend ollama --model gemma4:latest --concurrency 2 --tag v1

# 4) Label pairs — bulk expression, auto from judgments, or interactive TUI.
flywheel label --dataset toy --mode auto-from-judge \
  --judgments data/judgments/toy.v1.jsonl
flywheel label --dataset toy --mode bulk \
  --where "scores['overall'] >= 9" --set-status approved --tag high_quality \
  --judgments data/judgments/toy.v1.jsonl
flywheel label --dataset toy --mode interactive --resume

# 5) Inspect + export (with filter expression + train/val split).
flywheel dataset ls
flywheel dataset info toy
flywheel dataset export toy --to data/exports/train.jsonl \
  --filter "label['status'] == 'approved'" --split train=0.8,val=0.2

# 6) Visualize — render PNG charts + an index.html for the dataset.
flywheel visualize --dataset toy --output reports/toy

# 7) Run two judges, compare, and calibrate against labels.
flywheel judge    --dataset toy --backend ollama      --tag ollama
flywheel judge    --dataset toy --backend openrouter  --tag openrouter
flywheel compare  --dataset toy --tags ollama,openrouter
flywheel calibrate --dataset toy --tag ollama         # vs label.status=='approved'

# 8) Reproducible pipeline — one YAML, one command.
flywheel pipeline run pipeline.yaml
```

### Judge sharpness

- **Judgment cache.** Every call is keyed on
  `sha256(pair_id + rubric.name + rubric.version + backend + model)` and
  written to `data/judgments/.cache/`. Re-running a judge is instant and
  free. Disable per call with `--no-cache`.
- **`flywheel compare -d <ds> --tags a,b`.** Computes Cohen's κ on
  pass/fail, pass-agreement %, and Pearson correlation on overall scores
  between two judgment runs. Writes `reports/<ds>/compare.json`.
- **`flywheel calibrate -d <ds> --tag <t>`.** Treats `label.status ==
  'approved'` as ground truth and reports precision / recall / F1 / accuracy
  for the judge's `passed` decision. Writes `reports/<ds>/calibrate.<t>.json`.

### Pipeline YAML

`flywheel pipeline run <config.yaml>` runs steps in order, reusing the
same CLI logic so behaviour matches manual runs 1:1. Supported step
names: `ingest | validate | judge | label | export | visualize | compare
| calibrate`. `judgments: auto` in `label` / `export` resolves to the
previous `judge` step's tag, so you rarely need to spell paths out.

```yaml
dataset: toy
steps:
  - ingest:
      input: toy.jsonl
      format: jsonl
      map: {instruction: prompt, output: completion}
  - validate:
      checks: [schema, dedup, pii, length]
      fail_on: never
      write_clean: true
  - judge:
      backend: ollama
      model: gemma4:latest
      tag: v1
      max_pairs: 10
      concurrency: 2
  - label:
      mode: auto-from-judge
      judgments: auto
  - export:
      to: data/exports/toy.jsonl
      filter: "label['status'] == 'approved'"
      split: {train: 0.8, val: 0.2}
  - visualize: {}
```

Steps stop on the first non-zero exit by default; pass `--keep-going` to
continue.

### Visualizations

`flywheel visualize -d <dataset>` reads the dataset's pairs, judgments,
labels and validation report and renders a chart suite under
`reports/<dataset>/` (override with `--output`):

| file | chart | what it shows |
| --- | --- | --- |
| `categories.png` | horizontal bar | pair counts per `category` |
| `lengths.png` | side-by-side histograms | char lengths of instruction + output |
| `validation.png` | horizontal bar | issue counts per validation check |
| `pass_fail.png` | bar | judge pass vs fail totals |
| `scores.png` | histogram | overall-score distribution (0-10) |
| `criteria.png` | bar | mean coherence / accuracy / helpfulness / overall |
| `labels.png` | bar | label distribution (approved / needs_edit / rejected / skipped), latest-wins per pair |
| `judge_agreement.png` | 2×2 heatmap | pass/fail agreement between two judgment tags — rendered only when ≥2 `--tag` runs exist for the dataset |
| `index.html` | gallery | all charts on one page with a summary row |

**Rubrics** are YAML files under `rubrics/` — see `rubrics/default.yaml` and
`rubrics/factuality.yaml`. `--rubric` is optional; if omitted the tool uses
`rubrics/default.yaml` when present.

**Judge backends** (`src/synthetic_data_flywheel/judge_backends/`) all
implement a common async `JudgeBackend` protocol:

| backend | required env | default model |
| --- | --- | --- |
| `ollama` | local server on `OLLAMA_BASE_URL` (default `http://localhost:11434`) | `ollama_model` setting |
| `openrouter` | `OPENROUTER_API_KEY` | `openrouter_model` setting |
| `anthropic` | `ANTHROPIC_API_KEY` | `claude-haiku-4-5-20251001` |

**Bulk-label / export filter expressions** are evaluated in a restricted
sandbox over `{instruction, output, category, difficulty, metadata, scores,
passed, judge_model, label}`. Only literals, comparisons, boolean ops and
subscripts are allowed — attribute access, function calls and dunders are
rejected.

**Timeouts**: local 7–8B judge models (e.g. Gemma) can take 30–90 s per
call, so the default `judge_timeout` is 180 s. Bump it via env
(`JUDGE_TIMEOUT=300`) if you see `ReadTimeout` in the judge summary's
"Failed (errors)" row.

On-disk layout (added by these commands):

```
data/
  user/        <ds>.jsonl    <ds>.meta.json
  validation/  <ds>.report.json
  labels/      <ds>.jsonl                     # append-only, latest-wins
  judgments/   <ds>.<tag>.jsonl
rubrics/       default.yaml  factuality.yaml
```

The original `flywheel run | status | report | init` flywheel-loop commands
are unchanged.

### Python API

```python
import asyncio
from synthetic_data_flywheel import create_engine

# Create engine with seeds
engine = create_engine(
    seeds=["Explain Python", "What is AI?"],
    max_cycles=3,
)

# Run the flywheel
asyncio.run(engine.run_full_loop())

# Get summary
summary = engine.get_summary()
print(f"Generated {summary['total_passed_pairs']} pairs")
```

### A2A Agent (Multi-Agent Orchestration)

Start the A2A agent server:

```bash
python -m synthetic_data_flywheel.a2a_agent
```

Or use the FastAPI app directly:

```python
from synthetic_data_flywheel.a2a_agent import create_a2a_app
import uvicorn

app = create_a2a_app()
uvicorn.run(app, host="0.0.0.0", port=8000)
```

#### A2A Protocol Endpoints

- `GET /a2a/capabilities` - Get agent capabilities
- `POST /a2a/tasks/send` - Send a task to the agent
- `POST /a2a/tasks/get` - Get task status and result
- `POST /a2a/tasks/cancel` - Cancel a running task

## 🧪 Testing

Run the test suite:

```bash
pytest tests/ -v
```

Run with coverage:

```bash
pytest tests/ --cov=synthetic_data_flywheel --cov-report=html
```

## 📁 Project Structure

```
synthetic-data-flywheel/
├── src/synthetic_data_flywheel/
│   ├── __init__.py          # Package exports
│   ├── config.py            # Configuration management
│   ├── models.py            # Pydantic data models
│   ├── generator.py         # OpenRouter client
│   ├── judge.py             # Ollama client
│   ├── dataset_manager.py   # HuggingFace datasets
│   ├── trainer.py           # Unsloth notebook generation
│   ├── evaluator.py         # Metrics computation
│   ├── engine.py            # Main flywheel loop
│   ├── cli.py               # Command-line interface
│   ├── a2a_agent.py         # A2A protocol server
│   └── report_generator.py  # HTML report generation
├── tests/                   # Unit tests
├── data/                    # Data directory
├── docs/                    # Documentation
├── notebooks/               # Jupyter notebooks
├── reports/                 # Generated reports
├── templates/               # Jinja2 templates
├── pyproject.toml          # Project configuration
├── .env.example            # Environment template
└── README.md               # This file
```

## ⚙️ Configuration

All configuration is managed through environment variables or `.env` file:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENROUTER_API_KEY` | OpenRouter API key | Required |
| `OPENROUTER_MODEL` | Model for generation | `qwen/qwen3-8b` |
| `OLLAMA_HOST` | Ollama server URL | `http://localhost:11434` |
| `OLLAMA_MODEL` | Model for judging | `gemma4` |
| `QUALITY_MIN_SCORE` | Minimum quality threshold | `7.0` |
| `MAX_CYCLES` | Maximum flywheel cycles | `5` |
| `HF_TOKEN` | HuggingFace token | Optional |

## 🔧 Advanced Usage

### Custom Prompt Templates

```python
from synthetic_data_flywheel.generator import PromptTemplate

# Use built-in templates
template = PromptTemplate.get("REASONING")

# Or create custom
custom_template = """Based on: {seed}

Generate a step-by-step reasoning problem with solution."""
```

### Custom Quality Rubric

```python
from synthetic_data_flywheel.judge import QualityJudge

judge = QualityJudge()
# Modify QUALITY_RUBRIC_TEMPLATE in judge.py for custom criteria
```

### Dataset Versioning

```python
from synthetic_data_flywheel.dataset_manager import create_dataset_manager

dm = create_dataset_manager()

# Save to HuggingFace Hub
dm.save_to_huggingface(pairs, repo_id="username/dataset-name")

# Load from HuggingFace Hub
pairs = dm.load_from_huggingface("username/dataset-name")
```

## 📊 Metrics

The flywheel tracks:

- **Pass Rate**: Percentage of pairs passing quality judgment
- **Quality Scores**: Coherence, accuracy, helpfulness (0-10)
- **Cycle Duration**: Time per flywheel cycle
- **Total Generated**: Cumulative synthetic pairs

View metrics in generated HTML reports or via `flywheel status`.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open a Pull Request

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- [OpenRouter](https://openrouter.ai/) for LLM API access
- [Ollama](https://ollama.ai/) for local LLM inference
- [Unsloth](https://github.com/unslothai/unsloth) for efficient fine-tuning
- [HuggingFace](https://huggingface.co/) for dataset hosting
