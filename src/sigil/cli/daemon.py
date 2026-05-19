"""``sigil daemon`` Click subcommands.

V0 adds:
    --ed / --enable-dynamic flag for ``sigil daemon run`` that
    enables Tier 2 (swipe) and Tier 3 (pointer) dynamic detectors.
    Default OFF — V0 demos a clean static-gesture system; dynamic
    detectors are opt-in for testing.

Patch 3 added:
    --overlay flag launches the Tkinter overlay alongside the daemon.
    The daemon runs in a background thread, the overlay on the main
    thread, and they communicate via a bounded thread-safe queue.

The default (no flags) keeps Patch 2's behaviour exactly: console-only,
single-threaded, identical UX to ``sigil daemon run`` that worked in
Patch 2.
"""

from __future__ import annotations

import queue
import signal
import sys
import threading
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
def daemon() -> None:
    """Sigil runtime daemon: orchestrates perception → interpreter → executor."""


@daemon.command("run")
@click.option(
    "--model",
    "model_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to .onnx model. Default: most recent under " "models/static-classifier/.",
)
@click.option(
    "--no-auto-activate",
    is_flag=True,
    help="Don't auto-activate on startup or on gesture.",
)
@click.option(
    "--confidence-threshold",
    type=float,
    default=None,
    help="Minimum top-1 probability to surface a GestureEvent.",
)
@click.option(
    "--detection-threshold",
    type=float,
    default=None,
    help="Minimum MediaPipe detection confidence per hand.",
)
@click.option(
    "--overlay/--no-overlay",
    default=False,
    help="Show the Sigi mascot overlay window. Requires tkinter "
    "(bundled with standard Python on Windows).",
)
@click.option(
    "-ed",
    "--enable-dynamic",
    "enable_dynamic",
    is_flag=True,
    default=False,
    help="Enable Tier 2 swipe detection and Tier 3 pointer mode. "
    "Default OFF — V0 ships static gestures only by default.",
)
def run(
    model_path: Path | None,
    no_auto_activate: bool,
    confidence_threshold: float | None,
    detection_threshold: float | None,
    overlay: bool,
    enable_dynamic: bool,
) -> None:
    """Start the Sigil daemon. Ctrl+C to stop."""
    from sigil.daemon.runtime import SigilDaemon

    if model_path is None:
        model_path = _find_default_model()
        console.print(f"[cyan]Auto-discovered model:[/cyan] {model_path}")

    # Set up overlay queue + callback first so the daemon can be
    # constructed with the right on_event hook.
    overlay_queue: queue.Queue | None = None
    on_event = None
    if overlay:
        try:
            import tkinter  # noqa: F401 — fail fast if missing
        except ImportError:
            console.print(
                "[red]--overlay needs tkinter[/red] (bundled with Python "
                "on Windows; install python3-tk on Linux).",
            )
            sys.exit(2)
        # Bounded queue: if the overlay falls behind, we drop old
        # events. The overlay only ever reads the latest anyway.
        overlay_queue = queue.Queue(maxsize=64)

        def _on_event(event) -> None:
            try:
                overlay_queue.put_nowait(event)
            except queue.Full:
                # Drop one then put — bounded growth, latest wins.
                try:
                    overlay_queue.get_nowait()
                    overlay_queue.put_nowait(event)
                except queue.Empty:
                    pass

        on_event = _on_event

    try:
        sigil = SigilDaemon(
            model_path,
            auto_activate_on_gesture=not no_auto_activate,
            confidence_threshold=confidence_threshold,
            detection_threshold=detection_threshold,
            on_event=on_event,
            enable_swipes=enable_dynamic,
            enable_pointer=enable_dynamic,
        )
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Failed to start daemon:[/red] {exc}")
        sys.exit(2)

    _print_startup_banner(
        sigil,
        model_path,
        no_auto_activate,
        overlay,
        enable_dynamic,
    )

    if overlay:
        _run_with_overlay(sigil, overlay_queue)
    else:
        _run_console_only(sigil)


@daemon.command("list-verbs")
def list_verbs() -> None:
    """Print the registered action verbs."""
    from sigil.executor.registry import DEFAULT_REGISTRY

    table = Table(title="Registered verbs")
    table.add_column("Action", style="cyan")
    table.add_column("Destructive?", justify="center")
    table.add_column("Description")
    for name in sorted(DEFAULT_REGISTRY):
        verb = DEFAULT_REGISTRY[name]
        destructive = "yes" if getattr(verb, "is_destructive", False) else ""
        table.add_row(name, destructive, verb.description)
    console.print(table)


# --- run topologies ---------------------------------------------------


def _run_console_only(sigil) -> None:
    """Patch-2 behaviour: daemon on the main thread, Ctrl+C to stop."""

    def _on_sigint(_signum, _frame) -> None:
        console.print("\n[yellow]Stopping (Ctrl+C)…[/yellow]")
        sigil.stop()

    signal.signal(signal.SIGINT, _on_sigint)
    stats = sigil.run()
    _print_session_summary(stats)


def _run_with_overlay(sigil, overlay_queue) -> None:
    """Daemon in background thread + Tkinter on the main thread.

    Tkinter must own the main thread, so we invert: daemon goes into a
    background thread, Tkinter's mainloop blocks the main thread, and
    when the overlay closes (Escape, close button) we signal the
    daemon to stop and wait for it.
    """
    from sigil.ui.overlay import SigilOverlay

    daemon_thread = threading.Thread(
        target=_daemon_thread_main,
        args=(sigil,),
        daemon=True,
        name="sigil-daemon",
    )
    daemon_thread.start()

    overlay = SigilOverlay(overlay_queue)
    try:
        final_state = overlay.run()  # blocks until close
    finally:
        sigil.stop()
        daemon_thread.join(timeout=5.0)
    _print_session_summary(sigil.stats)


def _daemon_thread_main(sigil) -> None:
    """Run the daemon. If it crashes, print the FULL traceback so a
    silent ImportError doesn't get masked by the overlay window
    sitting there doing nothing."""
    import traceback

    try:
        sigil.run()
    except Exception as exc:  # noqa: BLE001
        console.print()
        console.print(
            "[bold red]Daemon thread crashed — the overlay window is now "
            "running against a dead daemon.[/bold red]",
        )
        console.print(f"[red]{type(exc).__name__}: {exc}[/red]")
        console.print("[red]" + "─" * 60 + "[/red]")
        console.print(traceback.format_exc())
        console.print("[red]" + "─" * 60 + "[/red]")
        sigil._should_stop = True


# --- presentation helpers ---------------------------------------------


def _find_default_model() -> Path:
    base = Path("models/static-classifier")
    if not base.is_dir():
        raise click.ClickException(
            f"No {base} directory found. Pass --model explicitly.",
        )
    candidates = sorted(
        base.glob("*/best.onnx"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise click.ClickException(
            f"No best.onnx found under {base}. Run export_onnx first.",
        )
    return candidates[0]


def _print_startup_banner(
    sigil,
    model_path: Path,
    no_auto_activate: bool,
    overlay: bool,
    enable_dynamic: bool,
) -> None:
    console.print()
    console.print("[bold green]Sigil daemon starting[/bold green]")
    console.print(f"  Model:           {model_path}")
    console.print(
        f"  Classes:         {sigil.classifier.num_classes} "
        f"({', '.join(sigil.classifier.label_map.values())})",
    )
    console.print(
        f"  Conf threshold:  {sigil.classifier.confidence_threshold:.2f}",
    )
    console.print(
        f"  Det threshold:   {sigil.classifier.detection_threshold:.2f}",
    )
    console.print(
        f"  Auto-activate:   "
        f"{'NO (wake word required)' if no_auto_activate else 'YES (Patch 2 default)'}",
    )
    console.print(f"  Overlay:         {'YES (Sigi)' if overlay else 'no'}")
    console.print(
        f"  Dynamic (swipe + pointer): " f"{'YES' if enable_dynamic else 'NO (--ed to enable)'}",
    )
    console.print()
    if overlay:
        console.print(
            "[dim]Sigi appears bottom-right. Drag to reposition. "
            "Escape or close-button to stop.[/dim]",
        )
    else:
        console.print(
            "[dim]Ctrl+C to stop. Logs go to the structlog stream above.[/dim]",
        )
    console.print()


def _print_session_summary(stats) -> None:
    console.print()
    console.print("[bold]=== Session summary ===[/bold]")
    console.print(f"Elapsed:              {stats.elapsed_seconds:.1f} s")
    console.print(f"Frames processed:     {stats.frames_processed}")
    console.print(f"Avg FPS:              {stats.average_fps:.1f}")
    console.print(f"Events classified:    {stats.events_classified}")
    console.print(f"Auto-activations:     {stats.auto_activations}")
    console.print(f"Dispatches attempted: {stats.dispatches_attempted}")
    console.print(
        f"Dispatches succeeded: [green]{stats.dispatches_succeeded}[/green]",
    )
    if stats.dispatches_failed:
        console.print(
            f"Dispatches failed:    [red]{stats.dispatches_failed}[/red]",
        )
    if getattr(stats, "pointer_active_frames", 0):
        console.print(
            f"Pointer active frames: {stats.pointer_active_frames}",
        )

    if stats.per_action:
        table = Table(title="Per-action breakdown")
        table.add_column("Action")
        table.add_column("Successes", justify="right")
        table.add_column("Failures", justify="right")
        for action in sorted(stats.per_action):
            info = stats.per_action[action]
            table.add_row(
                action,
                str(info.get("successes", 0)),
                str(info.get("failures", 0)),
            )
        console.print(table)


__all__ = ["daemon"]
