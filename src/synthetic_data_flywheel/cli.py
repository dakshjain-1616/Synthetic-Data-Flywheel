"""CLI module - Click-based interface with rich terminal UI."""

import asyncio
import json
import random
from pathlib import Path
from typing import Iterable, List, Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from synthetic_data_flywheel.config import get_settings
from synthetic_data_flywheel.engine import create_engine
from synthetic_data_flywheel.models import SyntheticPair
from synthetic_data_flywheel.report_generator import create_report_generator


console = Console()


@click.group()
@click.version_option(version="0.1.0")
def main():
    """Synthetic Data Flywheel - Autonomous data generation pipeline."""
    pass


@main.command()
@click.option(
    "--seeds",
    "-s",
    required=True,
    help="Comma-separated list of seed prompts",
)
@click.option(
    "--max-cycles",
    "-c",
    type=int,
    default=None,
    help="Maximum number of flywheel cycles",
)
@click.option(
    "--checkpoint-dir",
    "-d",
    type=click.Path(),
    default=None,
    help="Directory for checkpoints",
)
@click.option(
    "--resume",
    "-r",
    is_flag=True,
    help="Resume from latest checkpoint",
)
def run(seeds: str, max_cycles: Optional[int], checkpoint_dir: Optional[str], resume: bool):
    """Run the synthetic data flywheel."""
    settings = get_settings()
    
    seed_list = [s.strip() for s in seeds.split(",") if s.strip()]
    
    console.print(Panel.fit(
        f"[bold green]Synthetic Data Flywheel[/bold green]\n"
        f"Seeds: {len(seed_list)}\n"
        f"Max Cycles: {max_cycles or settings.max_cycles}",
        title="Configuration",
    ))
    
    engine = create_engine(
        seeds=seed_list,
        checkpoint_dir=checkpoint_dir,
        max_cycles=max_cycles,
    )
    
    if resume:
        if engine.load_checkpoint():
            console.print("[yellow]Resumed from checkpoint[/yellow]")
        else:
            console.print("[red]No checkpoint found, starting fresh[/red]")
    
    try:
        asyncio.run(engine.run_full_loop())
        
        # Print summary
        summary = engine.get_summary()
        
        table = Table(title="Flywheel Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Total Cycles", str(summary["total_cycles"]))
        table.add_row("Total Passed Pairs", str(summary["total_passed_pairs"]))
        table.add_row("Avg Pass Rate", f"{summary['avg_pass_rate']:.2%}")
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.ClickException(str(e))


@main.command()
@click.option(
    "--checkpoint-dir",
    "-d",
    type=click.Path(),
    default=None,
    help="Directory containing checkpoints",
)
def status(checkpoint_dir: Optional[str]):
    """Show current flywheel status."""
    settings = get_settings()
    cp_dir = Path(checkpoint_dir or settings.checkpoint_dir)
    
    if not cp_dir.exists():
        console.print("[yellow]No checkpoints found[/yellow]")
        return
    
    checkpoints = sorted(cp_dir.glob("checkpoint_*.json"))
    
    if not checkpoints:
        console.print("[yellow]No checkpoints found[/yellow]")
        return
    
    console.print(f"[green]Found {len(checkpoints)} checkpoint(s)[/green]")
    
    # Load latest
    engine = create_engine(seeds=[])
    engine.load_checkpoint(str(checkpoints[-1]))
    
    summary = engine.get_summary()
    
    table = Table(title="Current Status")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    table.add_row("Current Cycle", str(summary["total_cycles"]))
    table.add_row("Total Passed Pairs", str(summary["total_passed_pairs"]))
    table.add_row("Avg Pass Rate", f"{summary['avg_pass_rate']:.2%}")
    
    console.print(table)
    
    if summary["cycles"]:
        cycle_table = Table(title="Cycle History")
        cycle_table.add_column("Cycle", style="cyan")
        cycle_table.add_column("Pass Rate", style="green")
        cycle_table.add_column("Avg Quality", style="blue")
        cycle_table.add_column("Duration (s)", style="magenta")
        
        for c in summary["cycles"]:
            cycle_table.add_row(
                str(c["cycle_id"]),
                f"{c['pass_rate']:.2%}",
                f"{c['avg_quality']:.2f}",
                str(int(c.get("duration", 0))),
            )
        
        console.print(cycle_table)


@main.command()
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Output directory for reports",
)
@click.option(
    "--checkpoint-dir",
    "-d",
    type=click.Path(),
    default=None,
    help="Directory containing checkpoints",
)
def report(output: Optional[str], checkpoint_dir: Optional[str]):
    """Generate HTML report from checkpoints."""
    settings = get_settings()
    
    cp_dir = Path(checkpoint_dir or settings.checkpoint_dir)
    report_dir = Path(output or settings.report_output_dir)
    
    if not cp_dir.exists():
        console.print("[red]No checkpoints found[/red]")
        return
    
    checkpoints = sorted(cp_dir.glob("checkpoint_*.json"))
    
    if not checkpoints:
        console.print("[red]No checkpoints found[/red]")
        return
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Generating report...", total=None)
        
        # Load all cycles
        engine = create_engine(seeds=[])
        engine.load_checkpoint(str(checkpoints[-1]))
        
        # Generate report
        generator = create_report_generator(output_dir=str(report_dir))
        report_path = generator.generate_report(engine.cycles)
        
        progress.update(task, completed=True)
    
    console.print(f"[green]Report generated: {report_path}[/green]")


@main.command()
def init():
    """Initialize flywheel configuration."""
    settings = get_settings()
    
    console.print(Panel.fit(
        "[bold green]Synthetic Data Flywheel Initialized[/bold green]\n\n"
        f"Data Directory: {settings.data_dir}\n"
        f"Checkpoint Directory: {settings.checkpoint_dir}\n"
        f"Report Directory: {settings.report_output_dir}",
        title="Initialization",
    ))
    
    # Create directories
    Path(settings.data_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.report_output_dir).mkdir(parents=True, exist_ok=True)
    
    console.print("[green]Directories created successfully[/green]")


# ===========================================================================
# Data-platform commands: ingest | validate | judge | label | dataset
# ===========================================================================


def _load_pairs_for_dataset(name: str) -> List[SyntheticPair]:
    from synthetic_data_flywheel.ingest import load_dataset_jsonl
    s = get_settings()
    p = Path(s.user_data_dir) / f"{name}.jsonl"
    if not p.exists():
        raise click.ClickException(f"Dataset not found: {p}")
    return load_dataset_jsonl(p)


def _load_judgments_jsonl(path: Path) -> dict:
    from synthetic_data_flywheel.models import JudgmentResult
    out = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            j = JudgmentResult.from_dict(json.loads(line))
            out[str(j.pair_id)] = j
    return out


def _parse_mapping(expr: Optional[str]) -> dict:
    if not expr:
        return {}
    out = {}
    for kv in expr.split(","):
        kv = kv.strip()
        if not kv or "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        out[k.strip()] = v.strip()
    return out


@main.command()
@click.option("--input", "-i", "input_", required=True, help="Path or hf://<id>")
@click.option("--name", "-n", required=True, help="Dataset name (stored under data/user/<name>.jsonl)")
@click.option("--format", "-f", "fmt", type=click.Choice(["auto", "jsonl", "json", "csv", "hf"]),
              default="auto")
@click.option("--map", "mapping", default=None,
              help="Field remap, e.g. 'instruction=prompt,output=completion'")
@click.option("--tag", default=None, help="Tag stored in per-pair metadata")
@click.option("--limit", type=int, default=None)
@click.option("--hf-split", default="train")
@click.option("--dry-run", is_flag=True)
def ingest(input_, name, fmt, mapping, tag, limit, hf_split, dry_run):
    """Ingest a user dataset into the flywheel's JSONL format."""
    from synthetic_data_flywheel.ingest import DatasetIngestor
    ing = DatasetIngestor()
    if dry_run:
        from itertools import islice
        rows = list(islice(ing._iter_rows(input_, fmt, hf_split), 3))
        console.print(f"[cyan]Preview[/cyan] ({len(rows)} rows):")
        for r in rows:
            console.print(r)
        return
    path, meta = ing.ingest(input_, name=name, fmt=fmt, mapping=_parse_mapping(mapping),
                             tag=tag, limit=limit, hf_split=hf_split)
    console.print(f"[green]Ingested {meta.row_count} pairs[/green] -> {path}")


@main.command()
@click.option("--dataset", "-d", required=True)
@click.option("--checks", default="schema,length,dedup,pii",
              help="Comma-separated: schema,length,dedup,pii,lang,profanity")
@click.option("--min-instruction-len", type=int, default=3)
@click.option("--max-output-len", type=int, default=8000)
@click.option("--lang", default=None)
@click.option("--pii", "pii_policy", type=click.Choice(["strict", "warn", "off"]), default=None)
@click.option("--write-clean", type=click.Path(), default=None)
@click.option("--output", "-o", type=click.Path(), default=None)
@click.option("--fail-on", type=click.Choice(["error", "warning", "never"]), default="never")
def validate(dataset, checks, min_instruction_len, max_output_len, lang, pii_policy,
             write_clean, output, fail_on):
    """Validate a dataset and write a ValidationReport."""
    from synthetic_data_flywheel.validator import Validator
    s = get_settings()
    pairs = _load_pairs_for_dataset(dataset)
    opts = {
        "min_instruction_len": min_instruction_len,
        "max_output_len": max_output_len,
        "lang": lang,
        "pii_policy": pii_policy or s.pii_policy,
    }
    validator = Validator(options=opts)
    report = validator.validate(
        pairs,
        checks=[c.strip() for c in checks.split(",") if c.strip()],
        dataset=dataset,
    )

    report_path = Path(output) if output else Path(s.validation_dir) / f"{dataset}.report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report.to_dict(), indent=2))

    # Summary
    table = Table(title=f"Validation: {dataset}")
    table.add_column("Metric", style="cyan"); table.add_column("Value", style="magenta")
    table.add_row("Total pairs", str(report.total_pairs))
    for k, count in sorted(report.counts.items()):
        table.add_row(k, str(count))
    console.print(table)
    console.print(f"[green]Report:[/green] {report_path}")

    if write_clean:
        clean = validator.filter_clean(pairs, report)
        out = Path(write_clean)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for p in clean:
                f.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")
        console.print(f"[green]Clean dataset written ({len(clean)} pairs):[/green] {out}")

    errors = report.counts.get("severity:error", 0)
    warnings = report.counts.get("severity:warning", 0)
    if fail_on == "error" and errors:
        raise click.ClickException(f"{errors} error(s) found")
    if fail_on == "warning" and (errors or warnings):
        raise click.ClickException(f"{errors + warnings} issue(s) found")


@main.command()
@click.option("--dataset", "-d", required=True)
@click.option("--rubric", "rubric_path", type=click.Path(), default=None,
              help="Path to rubric YAML/JSON (default: rubrics/default.yaml if present)")
@click.option("--backend", type=click.Choice(["ollama", "openrouter", "anthropic"]), default=None)
@click.option("--model", default=None)
@click.option("--concurrency", type=int, default=None)
@click.option("--max-pairs", type=int, default=None)
@click.option("--sample", type=float, default=None,
              help="Fraction [0,1] to sample")
@click.option("--output", "-o", type=click.Path(), default=None)
@click.option("--tag", default=None)
@click.option("--no-cache", "no_cache", is_flag=True,
              help="Skip the judgment cache; always re-call the backend.")
def judge(dataset, rubric_path, backend, model, concurrency, max_pairs, sample, output, tag,
          no_cache):
    """Judge a dataset with an LLM-as-judge backend."""
    from synthetic_data_flywheel.judge import AsyncQualityJudge
    from synthetic_data_flywheel.judge_backends import get_backend
    from synthetic_data_flywheel.judge_cache import JudgmentCache
    from synthetic_data_flywheel.rubrics import default_rubric, load_rubric

    s = get_settings()
    pairs = _load_pairs_for_dataset(dataset)

    if sample is not None and 0 < sample < 1:
        k = max(1, int(len(pairs) * sample))
        random.seed(42)
        pairs = random.sample(pairs, k)
    if max_pairs is not None:
        pairs = pairs[:max_pairs]

    if rubric_path:
        rubric = load_rubric(rubric_path)
    else:
        default_yaml = Path(s.rubrics_dir) / "default.yaml"
        rubric = load_rubric(default_yaml) if default_yaml.exists() else default_rubric()

    backend_name = backend or s.default_judge_backend
    be = get_backend(backend_name, model=model) if model else get_backend(backend_name)

    cache = JudgmentCache(root=Path(s.judgments_dir) / ".cache", enabled=not no_cache)
    judger = AsyncQualityJudge(backend=be, rubric=rubric, tag=tag, cache=cache,
                                backend_name=backend_name)
    concurrency = concurrency or s.judge_concurrency

    async def _run():
        try:
            return await judger.judge_batch(pairs, concurrency=concurrency)
        finally:
            await judger.close()

    console.print(f"[cyan]Judging {len(pairs)} pairs with {backend_name}:{be.model}[/cyan]")
    judgments = asyncio.run(_run())

    out_path = Path(output) if output else Path(s.judgments_dir) / f"{dataset}.{tag or backend_name}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for j in judgments:
            f.write(json.dumps(j.to_dict(), ensure_ascii=False) + "\n")

    passed = sum(1 for j in judgments if j.passed)
    failures = [j for j in judgments if j.judgment_reasoning.startswith("Judgment failed:")]
    scored = [j for j in judgments if j not in failures]
    avg = (sum(j.scores.overall for j in scored) / len(scored)) if scored else 0.0
    table = Table(title=f"Judge: {dataset}")
    table.add_column("Metric", style="cyan"); table.add_column("Value", style="magenta")
    table.add_row("Judged", str(len(judgments)))
    table.add_row("Passed", f"{passed} ({passed / max(1, len(judgments)):.1%})")
    table.add_row("Failed (errors)", str(len(failures)))
    table.add_row("Avg overall (scored)", f"{avg:.2f}")
    table.add_row("Output", str(out_path))
    cstats = cache.stats()
    table.add_row("Cache", f"hits={cstats['hits']} misses={cstats['misses']} writes={cstats['writes']}")
    console.print(table)
    if failures:
        # Show a sample error so timeouts/auth issues surface immediately.
        console.print(f"[yellow]Sample failure:[/yellow] {failures[0].judgment_reasoning}")


@main.command()
@click.option("--dataset", "-d", required=True)
@click.option("--mode", type=click.Choice(["interactive", "bulk", "auto-from-judge"]), required=True)
@click.option("--where", "expr", default=None, help="Bulk filter expression")
@click.option("--set-status", "status", type=click.Choice(["approved", "rejected", "needs_edit", "skipped"]),
              default=None)
@click.option("--tag", default=None)
@click.option("--note", default=None)
@click.option("--judgments", "judgments_path", type=click.Path(), default=None)
@click.option("--reject-below", type=float, default=3.5)
@click.option("--output", "-o", type=click.Path(), default=None)
@click.option("--resume", is_flag=True)
def label(dataset, mode, expr, status, tag, note, judgments_path, reject_below, output, resume):
    """Label a dataset: interactive/bulk/auto-from-judge."""
    from synthetic_data_flywheel.labeler import (
        LabelStore, auto_from_judge, bulk_apply, interactive_loop,
    )

    s = get_settings()
    pairs = _load_pairs_for_dataset(dataset)
    store_path = Path(output) if output else Path(s.labels_dir) / f"{dataset}.jsonl"
    store = LabelStore(store_path)
    existing = store.load()

    judgments = {}
    if judgments_path:
        judgments = _load_judgments_jsonl(Path(judgments_path))

    if mode == "auto-from-judge":
        if not judgments:
            raise click.ClickException("--judgments is required for auto-from-judge")
        labels = auto_from_judge(judgments.values(), reject_below=reject_below)
    elif mode == "bulk":
        if not expr or not status:
            raise click.ClickException("--where and --set-status are required for bulk mode")
        labels = bulk_apply(pairs, judgments, expr=expr, status=status,
                            tag=tag, note=note, existing=existing)
    else:  # interactive
        # Only skip already-labeled when the user opts in via --resume.
        skip = existing if resume else {}
        labels = interactive_loop(pairs, judgments=judgments, already_labeled=skip)

    if resume:
        # Treat --resume as "idempotent re-run": drop labels whose current
        # status+tag already match the latest entry in the store. Applies to
        # bulk and auto-from-judge; interactive already skips via `skip` above.
        def _same(pid: str, new_status: str, new_tag):
            cur = existing.get(pid)
            return cur is not None and cur.status == new_status and cur.tag == new_tag
        labels = [l for l in labels if not _same(str(l.pair_id), l.status, l.tag)]

    n = store.extend(labels)
    console.print(f"[green]Wrote {n} labels[/green] -> {store.path}")


@main.group()
def dataset():
    """Dataset management: ls | info | export."""


@dataset.command("ls")
def dataset_ls():
    s = get_settings()
    root = Path(s.user_data_dir)
    if not root.exists():
        console.print("[yellow]No datasets[/yellow]")
        return
    table = Table(title="Datasets")
    for col in ("name", "pairs", "source", "tags"):
        table.add_column(col)
    for meta_path in sorted(root.glob("*.meta.json")):
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        table.add_row(meta.get("name", meta_path.stem),
                      str(meta.get("row_count", "?")),
                      meta.get("source", "?"),
                      ",".join(meta.get("tags") or []))
    console.print(table)


@dataset.command("info")
@click.argument("name")
def dataset_info(name):
    s = get_settings()
    paths = {
        "pairs": Path(s.user_data_dir) / f"{name}.jsonl",
        "meta": Path(s.user_data_dir) / f"{name}.meta.json",
        "validation": Path(s.validation_dir) / f"{name}.report.json",
        "labels": Path(s.labels_dir) / f"{name}.jsonl",
    }
    table = Table(title=f"Dataset: {name}")
    table.add_column("Artifact", style="cyan"); table.add_column("Path")
    table.add_column("Status", style="magenta")
    for k, p in paths.items():
        table.add_row(k, str(p), "present" if p.exists() else "missing")
    # Judgments: glob
    j_dir = Path(s.judgments_dir)
    judgments = sorted(j_dir.glob(f"{name}.*.jsonl")) if j_dir.exists() else []
    table.add_row("judgments", str(j_dir), f"{len(judgments)} set(s)")
    console.print(table)


@dataset.command("export")
@click.argument("name")
@click.option("--to", "to_path", required=True, type=click.Path())
@click.option("--format", "fmt", type=click.Choice(["jsonl", "json", "csv"]), default="jsonl")
@click.option("--filter", "expr", default=None)
@click.option("--judgments", "judgments_path", type=click.Path(), default=None)
@click.option("--split", default=None, help="e.g. train=0.9,val=0.1")
@click.option("--seed", type=int, default=42)
def dataset_export(name, to_path, fmt, expr, judgments_path, split, seed):
    """Export a dataset (optionally with filter + train/val split)."""
    from synthetic_data_flywheel.labeler import LabelStore, SafeEval, _pair_context
    import csv as _csv

    s = get_settings()
    pairs = _load_pairs_for_dataset(name)

    judgments = {}
    if judgments_path:
        judgments = _load_judgments_jsonl(Path(judgments_path))

    labels_path = Path(s.labels_dir) / f"{name}.jsonl"
    labels = LabelStore(labels_path).load() if labels_path.exists() else {}

    if expr:
        predicate = SafeEval(expr)
        filtered = []
        for p in pairs:
            ctx = _pair_context(p, judgments.get(str(p.id)), labels.get(str(p.id)))
            try:
                if predicate(ctx):
                    filtered.append(p)
            except Exception:
                continue
        pairs = filtered

    splits = _parse_splits(split)
    outputs = _apply_splits(pairs, splits, seed)

    to_path_p = Path(to_path)
    to_path_p.parent.mkdir(parents=True, exist_ok=True)

    for split_name, items in outputs.items():
        target = _split_target(to_path_p, split_name, fmt) if split_name else to_path_p
        _write_pairs(items, target, fmt)
        console.print(f"[green]Wrote {len(items)} pairs[/green] -> {target}")


def _parse_splits(expr):
    if not expr:
        return {}
    out = {}
    total = 0.0
    for kv in expr.split(","):
        k, v = kv.split("=")
        out[k.strip()] = float(v)
        total += float(v)
    if abs(total - 1.0) > 0.01:
        raise click.ClickException("Split ratios must sum to 1.0")
    return out


def _apply_splits(pairs, splits, seed):
    if not splits:
        return {"": pairs}
    rng = random.Random(seed)
    shuffled = pairs.copy()
    rng.shuffle(shuffled)
    out = {}
    n = len(shuffled)
    idx = 0
    keys = list(splits.keys())
    for i, k in enumerate(keys):
        if i == len(keys) - 1:
            out[k] = shuffled[idx:]
        else:
            count = int(n * splits[k])
            out[k] = shuffled[idx:idx + count]
            idx += count
    return out


def _split_target(base: Path, split_name: str, fmt: str) -> Path:
    stem, ext = base.stem, base.suffix or f".{fmt}"
    return base.with_name(f"{stem}.{split_name}{ext}")


def _write_pairs(pairs, target: Path, fmt: str):
    import csv as _csv
    if fmt == "jsonl":
        with target.open("w", encoding="utf-8") as f:
            for p in pairs:
                f.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")
    elif fmt == "json":
        target.write_text(json.dumps([p.to_dict() for p in pairs], indent=2))
    elif fmt == "csv":
        with target.open("w", encoding="utf-8", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["id", "instruction", "input", "output", "category"])
            for p in pairs:
                w.writerow([str(p.id), p.instruction, p.input or "", p.output, p.category or ""])


@main.command()
@click.option("--dataset", "-d", required=True)
@click.option("--tags", required=True, help="Comma-separated judgment tags to compare")
@click.option("--output", "-o", type=click.Path(), default=None,
              help="Output JSON (default: reports/<dataset>/compare.json)")
def compare(dataset, tags, output):
    """Compare two+ judgment runs (Cohen's κ, pass agreement, score correlation)."""
    from synthetic_data_flywheel.stats import cohens_kappa, pearson

    s = get_settings()
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    if len(tag_list) < 2:
        raise click.ClickException("--tags needs at least two comma-separated values")

    judgments = {}
    for tag in tag_list:
        p = Path(s.judgments_dir) / f"{dataset}.{tag}.jsonl"
        if not p.exists():
            raise click.ClickException(f"No judgments found: {p}")
        judgments[tag] = _load_judgments_jsonl(p)

    # Compare the first two (extend later if more)
    a, b = tag_list[0], tag_list[1]
    a_map = judgments[a]
    b_map = judgments[b]
    common = sorted(set(a_map) & set(b_map))
    if not common:
        raise click.ClickException(f"No overlapping pair_ids between '{a}' and '{b}'.")

    a_pass = [a_map[k].passed for k in common]
    b_pass = [b_map[k].passed for k in common]
    a_score = [a_map[k].scores.overall for k in common]
    b_score = [b_map[k].scores.overall for k in common]
    agreement = sum(1 for x, y in zip(a_pass, b_pass) if x == y) / len(common)
    kappa = cohens_kappa(a_pass, b_pass)
    corr = pearson(a_score, b_score)

    result = {
        "dataset": dataset,
        "tags": [a, b],
        "n_common": len(common),
        "pass_agreement": agreement,
        "cohens_kappa": kappa,
        "score_pearson": corr,
        "a_passed": sum(a_pass),
        "b_passed": sum(b_pass),
        "a_mean": sum(a_score) / len(a_score),
        "b_mean": sum(b_score) / len(b_score),
    }

    out_path = Path(output) if output else Path(s.report_output_dir) / dataset / "compare.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))

    table = Table(title=f"Judge comparison: {a} vs {b}")
    table.add_column("Metric", style="cyan"); table.add_column("Value", style="magenta")
    table.add_row("Common pairs", str(result["n_common"]))
    table.add_row(f"{a} passed / mean", f"{result['a_passed']} / {result['a_mean']:.2f}")
    table.add_row(f"{b} passed / mean", f"{result['b_passed']} / {result['b_mean']:.2f}")
    table.add_row("Pass agreement", f"{agreement:.1%}")
    table.add_row("Cohen's κ (pass/fail)", f"{kappa:.3f}  {_kappa_label(kappa)}")
    table.add_row("Score Pearson r", f"{corr:.3f}")
    table.add_row("Output", str(out_path))
    console.print(table)


def _kappa_label(k: float) -> str:
    if k < 0: return "(below chance)"
    if k < 0.20: return "(poor)"
    if k < 0.40: return "(fair)"
    if k < 0.60: return "(moderate)"
    if k < 0.80: return "(substantial)"
    return "(near-perfect)"


@main.command()
@click.option("--dataset", "-d", required=True)
@click.option("--tag", required=True, help="Judgment tag to calibrate against labels")
@click.option("--approved-is", default="approved",
              help="Label status counted as positive ground truth (default: approved)")
@click.option("--output", "-o", type=click.Path(), default=None)
def calibrate(dataset, tag, approved_is, output):
    """Measure judge 'passed' against human labels (precision/recall/F1)."""
    from synthetic_data_flywheel.labeler import LabelStore
    from synthetic_data_flywheel.stats import prf

    s = get_settings()
    jp = Path(s.judgments_dir) / f"{dataset}.{tag}.jsonl"
    if not jp.exists():
        raise click.ClickException(f"No judgments: {jp}")
    judgments = _load_judgments_jsonl(jp)

    lp = Path(s.labels_dir) / f"{dataset}.jsonl"
    if not lp.exists():
        raise click.ClickException(f"No labels: {lp}. Label some pairs first.")
    labels = LabelStore(lp).load()

    common = sorted(set(judgments) & set(labels))
    if not common:
        raise click.ClickException("No pairs have both a judgment and a label.")

    preds = [judgments[k].passed for k in common]
    truth = [labels[k].status == approved_is for k in common]
    m = prf(preds, truth)
    m["dataset"] = dataset
    m["tag"] = tag
    m["approved_is"] = approved_is
    m["n_evaluated"] = len(common)

    out_path = Path(output) if output else Path(s.report_output_dir) / dataset / f"calibrate.{tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(m, indent=2))

    table = Table(title=f"Calibration: judge '{tag}' vs label.status=='{approved_is}'")
    table.add_column("Metric", style="cyan"); table.add_column("Value", style="magenta")
    table.add_row("Evaluated pairs", str(m["n_evaluated"]))
    table.add_row("Precision", f"{m['precision']:.3f}")
    table.add_row("Recall",    f"{m['recall']:.3f}")
    table.add_row("F1",        f"{m['f1']:.3f}")
    table.add_row("Accuracy",  f"{m['accuracy']:.3f}")
    table.add_row("TP/FP/TN/FN", f"{m['tp']}/{m['fp']}/{m['tn']}/{m['fn']}")
    table.add_row("Output", str(out_path))
    console.print(table)


@main.command()
@click.option("--dataset", "-d", required=True)
@click.option("--output", "-o", type=click.Path(), default=None,
              help="Output directory (default: reports/<dataset>/)")
def visualize(dataset, output):
    """Render a suite of PNG charts + index.html for a dataset."""
    from synthetic_data_flywheel.viz import load_inputs, render_all

    s = get_settings()
    out_dir = Path(output) if output else Path(s.report_output_dir) / dataset
    inputs = load_inputs(
        dataset,
        user_data_dir=s.user_data_dir,
        judgments_dir=s.judgments_dir,
        labels_dir=s.labels_dir,
        validation_dir=s.validation_dir,
    )
    if not inputs.pairs and not inputs.judgments_by_tag:
        raise click.ClickException(
            f"No data found for dataset '{dataset}'. Ingest it first.")
    images, index = render_all(inputs, out_dir)
    table = Table(title=f"Visualizations: {dataset}")
    table.add_column("Chart", style="cyan"); table.add_column("Path")
    for img in images:
        table.add_row(img.stem, str(img))
    table.add_row("index.html", str(index))
    console.print(table)


@main.group()
def pipeline():
    """Run declarative YAML pipelines."""


@pipeline.command("run")
@click.argument("config", type=click.Path(exists=True, dir_okay=False))
@click.option("--keep-going", is_flag=True, help="Continue past failing steps.")
def pipeline_run(config, keep_going):
    """Run every step declared in a pipeline YAML file."""
    from synthetic_data_flywheel.pipeline import load_pipeline, run_pipeline

    cfg = load_pipeline(config)
    result = run_pipeline(cfg, stop_on_error=not keep_going,
                           echo=lambda m: console.print(m))

    table = Table(title=f"Pipeline: {result.dataset}")
    table.add_column("#", style="dim"); table.add_column("Step", style="cyan")
    table.add_column("Status"); table.add_column("Exit")
    for i, s in enumerate(result.steps, 1):
        ok = "[green]ok[/green]" if s.ok else "[red]fail[/red]"
        table.add_row(str(i), s.name, ok, str(s.exit_code))
    console.print(table)
    if not result.ok:
        raise click.ClickException("Pipeline had failing step(s).")


if __name__ == "__main__":
    main()
