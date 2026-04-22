# Synthetic Data Flywheel - Production Plan

## Goal
Build a complete, production-ready closed-loop autonomous pipeline that generates synthetic training pairs, filters them using a local LLM judge, trains a new model, evaluates it, and feeds failure cases back as seeds for continuous improvement.

## Research Summary

### Model IDs Verified (April 2026)
- **Qwen3 8B (Teacher/Generator)**: OpenRouter: `qwen/qwen3-8b`, HuggingFace: `Qwen/Qwen3-8B`
- **Qwen3 4B (Student)**: OpenRouter: `qwen/qwen3-4b`, HuggingFace: `Qwen/Qwen3-4B`
- **Gemma 4 E2B (Judge)**: HuggingFace: `google/gemma-4-E2B`, Ollama: `gemma4:e2b`

### APIs & Protocols
- **OpenRouter**: Uses OpenAI-compatible API format with base_url `https://openrouter.ai/api/v1`
- **Ollama**: Local HTTP API at `http://localhost:11434`, Python client available
- **A2A Protocol**: Agent2Agent Protocol Specification (Linux Foundation) - JSON-RPC style agent communication

### Training Framework
- **Unsloth**: For efficient Qwen3 4B fine-tuning, generates Colab-ready notebooks

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SYNTHETIC DATA FLYWHEEL                     │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐ │
│  │  Generator  │───→│    Judge    │───→│   Training Data     │ │
│  │ (Qwen3 8B)  │    │(Gemma4 E2B) │    │   (HF Dataset)      │ │
│  │  OpenRouter │    │   Ollama    │    │                     │ │
│  └─────────────┘    └─────────────┘    └─────────────────────┘ │
│         ↑                                              │        │
│         │           ┌─────────────┐                    ↓        │
│         └───────────│  Feedback   │←─────────────────┘        │
│                     │   Loop      │                             │
│                     │ (Failures→  │                             │
│                     │   Seeds)     │                             │
│                     └──────┬───────┘                             │
│                            │                                    │
│  ┌─────────────┐    ┌─────┴──────┐    ┌─────────────────────┐   │
│  │  Evaluator  │←───│   Trainer  │←───│   Fine-tuned Model  │   │
│  │ (Held-out)  │    │ (Unsloth)  │    │   (Qwen3 4B)        │   │
│  └──────┬──────┘    └────────────┘    └─────────────────────┘   │
│         │                                                        │
│         ↓                                                        │
│  ┌─────────────┐    ┌─────────────────────────────────────────┐   │
│  │   Metrics   │───→│   Dashboard (HTML Report per cycle)    │   │
│  │   Report    │    │                                        │   │
│  └─────────────┘    └─────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Subtasks

### Phase 1: Project Foundation
1. **Create project structure** - Set up directories, pyproject.toml with all dependencies, .env.example
2. **Create core data models** - Pydantic models for SyntheticPair, JudgmentResult, CycleState, FlywheelConfig
3. **Implement configuration management** - Load from .env, validate configs, defaults

### Phase 2: Core Pipeline Components
4. **Implement Generator module** - OpenRouter client for Qwen3 8B, prompt templates for QA/instruction generation
5. **Implement Judge module** - Ollama client for Gemma4 E2B, quality scoring rubric, filtering logic
6. **Implement Dataset Manager** - HuggingFace datasets integration, save/load, versioning
7. **Implement Trainer module** - Unsloth notebook template generation, Colab-ready output
8. **Implement Evaluator module** - Held-out test evaluation, metric computation (accuracy, perplexity)

### Phase 3: Orchestration & Feedback
9. **Implement Flywheel engine** - Main loop orchestration, cycle management, checkpointing
10. **Implement feedback extraction** - Identify failure cases, convert to seeds for next cycle
11. **Implement state checkpointing** - JSON state files, resume capability

### Phase 4: Interfaces & Reporting
12. **Implement CLI** - Click-based interface with rich terminal UI, `flywheel run` command
13. **Implement A2A Agent** - HTTP server exposing A2A protocol endpoints for multi-agent orchestration
14. **Implement Report Generator** - Jinja2-based HTML dashboard with per-cycle metrics visualization

### Phase 5: Testing & Documentation
15. **Write unit tests** - 15+ tests covering all modules with pytest, mocks for external APIs
16. **Create README** - Installation, usage, architecture docs, examples
17. **Final integration test** - End-to-end dry run with mocked services

## Deliverables

| File Path | Description |
|-----------|-------------|
| `/app/synthetic_data_flywheel_2356/pyproject.toml` | Project dependencies and metadata |
| `/app/synthetic_data_flywheel_2356/.env.example` | Environment variable template |
| `/app/synthetic_data_flywheel_2356/src/synthetic_data_flywheel/__init__.py` | Package init with version |
| `/app/synthetic_data_flywheel_2356/src/synthetic_data_flywheel/models.py` | Pydantic data models |
| `/app/synthetic_data_flywheel_2356/src/synthetic_data_flywheel/config.py` | Configuration management |
| `/app/synthetic_data_flywheel_2356/src/synthetic_data_flywheel/generator.py` | Synthetic pair generation via OpenRouter |
| `/app/synthetic_data_flywheel_2356/src/synthetic_data_flywheel/judge.py` | Quality filtering via Ollama |
| `/app/synthetic_data_flywheel_2356/src/synthetic_data_flywheel/dataset.py` | HF datasets management |
| `/app/synthetic_data_flywheel_2356/src/synthetic_data_flywheel/trainer.py` | Unsloth notebook generation |
| `/app/synthetic_data_flywheel_2356/src/synthetic_data_flywheel/evaluator.py` | Model evaluation |
| `/app/synthetic_data_flywheel_2356/src/synthetic_data_flywheel/flywheel.py` | Main orchestration loop |
| `/app/synthetic_data_flywheel_2356/src/synthetic_data_flywheel/feedback.py` | Failure case extraction |
| `/app/synthetic_data_flywheel_2356/src/synthetic_data_flywheel/checkpoint.py` | State persistence |
| `/app/synthetic_data_flywheel_2356/src/synthetic_data_flywheel/a2a_agent.py` | A2A protocol HTTP server |
| `/app/synthetic_data_flywheel_2356/src/synthetic_data_flywheel/report.py` | HTML metrics dashboard |
| `/app/synthetic_data_flywheel_2356/src/synthetic_data_flywheel/cli.py` | Click CLI with Rich UI |
| `/app/synthetic_data_flywheel_2356/src/synthetic_data_flywheel/prompts.py` | Prompt templates |
| `/app/synthetic_data_flywheel_2356/notebooks/` | Generated Colab notebooks directory |
| `/app/synthetic_data_flywheel_2356/tests/test_*.py` | 15+ pytest unit tests |
| `/app/synthetic_data_flywheel_2356/README.md` | Documentation |

## Evaluation Criteria
- [ ] All 9 core modules implemented with type hints and docstrings
- [ ] CLI working: `flywheel run --task sentiment --cycles 3 --output data/`
- [ ] A2A agent HTTP server responding to protocol requests
- [ ] 15+ unit tests passing with pytest
- [ ] HTML report generation with cycle metrics
- [ ] Checkpoint/resume functionality working
- [ ] Proper error handling and logging throughout
- [ ] Rich terminal UI for progress indication

## Notes
- OpenRouter requires API key in OPENROUTER_API_KEY
- Ollama must be running locally for judge (port 11434)
- All models are April 2026 releases (Qwen3, Gemma 4)
- Unsloth notebook template will be generated, not executed locally
