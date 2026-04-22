"""CLI - Click-based interface with rich terminal UI."""

import asyncio
from pathlib import Path
from typing import List, Optional

import click
import structlog
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from synthetic_data_flywheel.config import get_settings
from synthetic_data_flywheel.engine import FlywheelEngine
from synthetic_data_flywheel.generator import PromptTemplate
from synthetic_data_flywheel.report_generator import ReportGenerator

logger = structlog.get_logger()
console = Console()


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Synthetic Data Flywheel - Autonomous synthetic data generation."""
    pass


@cli.command()
@click.option(
    "--seeds",
    "-s",
    multiple=True,
    help="Seed prompts for generation",
)
@click.option(
    "--seeds-file",
    "-f",
    type=click.Path(exists=True),
    help="File containing seed prompts (one per line)",
)
@click.option(
    "--max-cycles",
    "-c",
    type=int,
    default=None,
    help="Maximum number of cycles to run",
)
@click.option(
    "--template",
    "-t",
    type=click.Choice(PromptTemplate.list_templates()),
    default="instruction",
    help="Prompt template type",
)
@click.option(
    "--min-pass-rate",
    "-p",
    type=float,
    default=0.5,
    help="Minimum pass rate to continue",
)
@click.option(
    "--max-concurrent",
    "-m",
    type=int,
    default=5,
    help="Maximum concurrent generations",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Run in dry-run mode (no actual API calls)",
)
def run(
    seeds: tuple,
    seeds_file: Optional[str],
    max_cycles: Optional[int],
    template: str,
    min_pass_rate: float,
    max_concurrent: int,
    dry_run: bool,
):
    """Run the synthetic data flywheel."""
    console.print(Panel.fit(
        "[bold blue]🔄 Synthetic Data Flywheel[/bold blue]\n"
        "Autonomous synthetic data generation pipeline",
        title="Welcome",
        border_style="blue",
    ))
    
    # Load seeds
    seed_list = list(seeds)
    if seeds_file:
        with open(seeds_file, "r") as f:
            seed_list.extend(line.strip() for line in f if line.strip())
    
    if not seed_list:
        # Default seeds
        seed_list = [
            "Explain the concept of machine learning",
            "Write a Python function to calculate factorial",
            "Describe the water cycle",
            "What are the benefits of renewable energy?",
            "How does photosynthesis work?",
        ]
        console.print("[yellow]Using default seeds (no seeds provided)[/yellow]")
    
    console.print(f"[green]Loaded {len(seed_list)} seeds[/green]")
    
    if dry_run:
        console.print("[yellow]DRY RUN MODE - No API calls will be made[/yellow]")
        return
    
    # Initialize engine
    settings = get_settings()
    engine = FlywheelEngine()
    
    # Run flywheel
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Running flywheel...", total=None)
        
        try:
            cycles = asyncio.run(engine.run_full_loop(
                initial_seeds=seed_list,
                max_cycles=max_cycles or settings.max_cycles,
                min_pass_rate=min_pass_rate,
                template_type=template,
            ))
            
            progress.update(task, completed=True)
            
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise click.ClickException(str(e))
    
    # Display results
    console.print("\n[bold green]✓ Flywheel complete![/bold green]")
    
    table = Table(title="Cycle Summary")
    table.add_column("Cycle", style="cyan")
    table.add_column("Generated", style="magenta")
    table.add_column("Passed", style="green")
    table.add_column("Pass Rate", style="yellow")
    table.add_column("Avg Quality", style="blue")
    
    for cycle in cycles:
        table.add_row(
            str(cycle.cycle_id),
            str(len(cycle.generated_pairs)),
            str(len(cycle.passed_pairs)),
            f"{cycle.pass_rate:.1%}",
            f"{cycle.avg_quality_score:.2f}",
        )
    
    console.print(table)
    
    # Show report location
    summary = engine.get_summary()
    console.print(f"\n[bold]Total Generated:[/bold] {summary['total_generated']}")
    console.print(f"[bold]Total Passed:[/bold] {summary['total_passed']}")
    console.print(f"[bold]Overall Pass Rate:[/bold] {summary['overall_pass_rate']:.1%}")


@cli.command()
def status():
    """Show flywheel status and checkpoint information."""
    settings = get_settings()
    engine = FlywheelEngine()
    
    checkpoints = engine.list_checkpoints()
    
    console.print(Panel.fit(
        "[bold blue]Flywheel Status[/bold blue]",
        border_style="blue",
    ))
    
    if not checkpoints:
        console.print("[yellow]No checkpoints found[/yellow]")
        return
    
    console.print(f"[green]Found {len(checkpoints)} checkpoint(s)[/green]")
    
    table = Table(title="Checkpoints")
    table.add_column("Cycle ID", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Generated", style="magenta")
    table.add_column("Passed", style="green")
    table.add_column("Pass Rate", style="yellow")
    
    for cycle_id in checkpoints:
        cycle_state = engine.load_checkpoint(cycle_id)
        if cycle_state:
            table.add_row(
                str(cycle_state.cycle_id),
                cycle_state.status,
                str(len(cycle_state.generated_pairs)),
                str(len(cycle_state.passed_pairs)),
                f"{cycle_state.pass_rate:.1%}",
            )
    
    console.print(table)


@cli.command()
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output path for report",
)
def report(output: Optional[str]):
    """Generate HTML report from checkpoints."""
    settings = get_settings()
    engine = FlywheelEngine()
    report_gen = ReportGenerator()
    
    checkpoints = engine.list_checkpoints()
    
    if not checkpoints:
        console.print("[yellow]No checkpoints found to report on[/yellow]")
        return
    
    console.print(f"[cyan]Generating report from {len(checkpoints)} checkpoint(s)...[/cyan]")
    
    cycles = []
    for cycle_id in checkpoints:
        cycle_state = engine.load_checkpoint(cycle_id)
        if cycle_state:
            cycles.append(cycle_state)
    
    if output:
        output_path = Path(output)
    else:
        output_path = None
    
    report_path = report_gen.generate_flywheel_report(cycles, output_path)
    
    console.print(f"[green]✓ Report generated:[/green] {report_path}")


@cli.command()
@click.argument("seeds", nargs=-1)
def generate(seeds: tuple):
    """Quick generation from seeds (single cycle)."""
    if not seeds:
        raise click.UsageError("Please provide at least one seed")
    
    console.print(f"[cyan]Generating from {len(seeds)} seed(s)...[/cyan]")
    
    engine = FlywheelEngine()
    
    try:
        cycle = asyncio.run(engine.run_cycle(
            seeds=list(seeds),
            cycle_id=1,
        ))
        
        console.print(f"[green]✓ Generated {len(cycle.generated_pairs)} pairs[/green]")
        console.print(f"[green]✓ Passed {len(cycle.passed_pairs)} pairs[/green]")
        console.print(f"[green]✓ Pass rate: {cycle.pass_rate:.1%}[/green]")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.ClickException(str(e))


@cli.command()
def config():
    """Show current configuration."""
    settings = get_settings()
    
    table = Table(title="Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("OpenRouter Model", settings.openrouter_model)
    table.add_row("Ollama Host", settings.ollama_base_url)
    table.add_row("Ollama Model", settings.ollama_model)
    table.add_row("Quality Threshold", str(settings.quality_min_score))
    table.add_row("Max Cycles", str(settings.max_cycles))
    table.add_row("Data Directory", str(settings.data_dir))
    table.add_row("Checkpoint Directory", str(settings.checkpoint_dir))
    
    console.print(table)


def main():
    """Entry point for CLI."""
    cli()


if __name__ == "__main__":
    main()
