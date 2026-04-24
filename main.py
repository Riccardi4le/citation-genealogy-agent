"""Citation Genealogy Agent — CLI entry point."""

from __future__ import annotations
import sys
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

load_dotenv()

from agent.graph import build_graph
from agent.state import AgentState

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def run(
    claim: str = typer.Argument(..., help="The claim to verify (in quotes)"),
    source: str = typer.Option(
        ..., "--source", "-s",
        help="Citation to trace: paper title, 'Author Year', or DOI"
    ),
    max_depth: int = typer.Option(
        5, "--max-depth", "-d",
        help="Maximum citation hops to trace (default: 5)"
    ),
    output: Path = typer.Option(
        None, "--output", "-o",
        help="Save report to this .md file"
    ),
):
    """
    Trace a citation chain backward to the primary source and detect distortions.

    Example:
      python main.py "chewing gum improves focus by 35%" --source "Rossi et al. 2023"
    """
    console.print(Panel(
        Text.from_markup(
            f"[bold blue]Citation Genealogy Agent[/bold blue]\n"
            f"Claim: [italic]{claim}[/italic]\n"
            f"Source: [cyan]{source}[/cyan] | Max depth: {max_depth}"
        ),
        expand=False,
    ))

    graph = build_graph()

    initial: AgentState = {
        "original_claim": claim,
        "source_ref": source,
        "max_depth": max_depth,
        "tree": {},
        "queue": [],
        "visited": [],
        "errors": [],
        "current_node_id": None,
        "final_report": None,
        "status": "running",
    }

    try:
        with console.status("[bold green]Tracing citation tree…", spinner="dots"):
            final = graph.invoke(initial)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(0)
    except Exception as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise

    report = final.get("final_report") or "No report generated."

    if output:
        output.write_text(report, encoding="utf-8")
        console.print(f"\n[green]Report saved to[/green] {output}")
    else:
        console.print()
        console.print(Markdown(report))

    if final.get("errors"):
        console.print("\n[yellow]Warnings:[/yellow]")
        for err in final["errors"]:
            console.print(f"  · {err}")


if __name__ == "__main__":
    app()
