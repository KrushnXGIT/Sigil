"""Sigil CLI — `sigil` console script.

Phase 0 commands:
    sigil version                — print version
    sigil config path            — print where user config lives
    sigil config show [--path X] — show resolved config (after defaults overlay)
    sigil config validate [-p X] — load + validate without showing

More commands land as their phases ship (`sigil daemon`, `sigil train`, ...).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.syntax import Syntax

from sigil import __version__
from sigil.config import ConfigError, load_config, user_config_file
from sigil.logging import setup_logging

console = Console()


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
)
@click.option("--json-logs/--pretty-logs", default=None, help="Force log output style.")
def cli(log_level: str, json_logs: bool | None) -> None:
    """Sigil — hand-gesture control for Windows."""
    setup_logging(level=log_level, json=json_logs)  # type: ignore[arg-type]


@cli.command()
def version() -> None:
    """Print Sigil's version and exit."""
    click.echo(__version__)


@cli.group()
def config() -> None:
    """Inspect and validate configuration."""


# Subcommands from other modules are attached here. As more layers ship,
# they get one `cli.add_command(...)` line each.
from sigil.cli.dataset import dataset as _dataset_group  # noqa: E402
from sigil.cli.perception import perception as _perception_group  # noqa: E402

cli.add_command(_perception_group)
cli.add_command(_dataset_group)
from sigil.cli.daemon import daemon as _daemon_group  # noqa: E402

cli.add_command(_daemon_group)


@config.command("path")
def config_path() -> None:
    """Print the path where user gestures.yaml lives (creating directories isn't needed)."""
    click.echo(str(user_config_file()))


@config.command("show")
@click.option(
    "-p",
    "--path",
    "config_path",
    type=click.Path(path_type=Path),
    help="Load config from this file instead of the user default.",
)
def config_show(config_path: Path | None) -> None:
    """Show the resolved configuration (defaults + user overrides, validated)."""
    try:
        cfg = load_config(path=config_path)
    except ConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(1)

    payload = cfg.model_dump(mode="json")
    rendered = json.dumps(payload, indent=2)
    if sys.stdout.isatty():
        console.print(Syntax(rendered, "json", background_color="default"))
    else:
        click.echo(rendered)


@config.command("validate")
@click.option("-p", "--path", "config_path", type=click.Path(exists=True, path_type=Path))
def config_validate(config_path: Path | None) -> None:
    """Load + validate config without printing it. Exit code 0 on success."""
    try:
        load_config(path=config_path)
    except ConfigError as exc:
        console.print(f"[red]Invalid config:[/red]\n{exc}")
        sys.exit(1)
    console.print("[green]Config OK[/green]")


if __name__ == "__main__":
    cli()
