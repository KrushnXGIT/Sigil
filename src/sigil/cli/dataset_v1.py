"""``sigil dataset-v1 ...`` — CLI for V1 (dynamic) dataset pipeline.

Commands:
    sigil dataset-v1 instructions  — print download instructions for Jester
    sigil dataset-v1 validate      — sanity-check a downloaded Jester copy
    sigil dataset-v1 build         — preprocess Jester videos → parquet caches
    sigil dataset-v1 summarise     — class + count summary of a parquet

Per ADR-0012, this pipeline is the gating step for V1 training. The
``build`` command is the long one: running MediaPipe over thousands
of videos takes hours. Provide ``--max-per-class 50`` for a 5-minute
smoke-test run before committing to the full ~8-hour preprocessing.

The output parquets feed the V1 training script (V1 patch 2).
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from sigil.logging import get_logger

log = get_logger(__name__)
console = Console()


DEFAULT_JESTER_ROOT = Path("datasets/raw/jester")
DEFAULT_OUTPUT_DIR = Path("datasets/processed/v1")


@click.group(name="dataset-v1")
def dataset_v1() -> None:
    """V1 dynamic-gesture dataset pipeline (20BN-Jester + MediaPipe)."""


# ---------------------------------------------------------------------------
# instructions
# ---------------------------------------------------------------------------


@dataset_v1.command("instructions")
@click.option(
    "-d",
    "--dest",
    default=DEFAULT_JESTER_ROOT,
    show_default=True,
    type=click.Path(path_type=Path),
    help="Where Jester data should end up.",
)
def instructions_cmd(dest: Path) -> None:
    """Print download + extraction instructions for 20BN-Jester."""
    from sigil.intelligence.dataset.jester import print_download_instructions
    console.print(print_download_instructions(dest))


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


@dataset_v1.command("validate")
@click.option(
    "-r",
    "--root",
    default=DEFAULT_JESTER_ROOT,
    show_default=True,
    type=click.Path(path_type=Path),
    help="Jester root directory (contains annotations/ and 20bn-jester-v1/).",
)
def validate_cmd(root: Path) -> None:
    """Verify a downloaded Jester copy has the expected layout."""
    from sigil.intelligence.dataset.jester import (
        JesterError,
        validate_jester_layout,
    )
    try:
        validate_jester_layout(root)
    except JesterError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(2)
    console.print(f"[green]Jester layout OK at {root}[/green]")


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


@dataset_v1.command("build")
@click.option(
    "-r",
    "--root",
    default=DEFAULT_JESTER_ROOT,
    show_default=True,
    type=click.Path(path_type=Path),
    help="Jester root directory.",
)
@click.option(
    "-o",
    "--output-dir",
    default=DEFAULT_OUTPUT_DIR,
    show_default=True,
    type=click.Path(path_type=Path),
    help="Where train/val parquet files will be written.",
)
@click.option(
    "--max-per-class",
    type=int,
    default=None,
    help="Cap records per Sigil class per split. Use 50 for a "
         "~5-minute pipeline-validation run. Omit for full preprocessing.",
)
@click.option(
    "--splits",
    default="train,val",
    show_default=True,
    help="Comma-separated splits to process.",
)
def build_cmd(
    root: Path,
    output_dir: Path,
    max_per_class: int | None,
    splits: str,
) -> None:
    """Preprocess Jester videos into landmark-sequence parquet caches.

    This is the long step: ~8 hours CPU for the full V1 set
    (~25,000 videos × 36 frames = 900,000 MediaPipe inferences).
    Use --max-per-class 50 for a smoke test first.
    """
    try:
        from sigil.intelligence.dataset.jester import (
            JESTER_TO_SIGIL_V1,
            JesterError,
            SPLIT_CSV_NAMES,
            csv_path_for_split,
            filter_to_v1_classes,
            parse_jester_csv,
            validate_jester_layout,
            videos_dir_for_split,
        )
        from sigil.intelligence.dataset.temporal_preprocessing import (
            preprocess_records,
        )
        from sigil.intelligence.dataset.temporal_storage import (
            write_temporal_records,
        )
        from sigil.perception.landmarks import HandLandmarkerWrapper
    except ImportError as exc:
        console.print(f"[red]Missing extras:[/red] {exc}")
        sys.exit(2)

    # Validate layout first — fail fast.
    try:
        validate_jester_layout(root)
    except JesterError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(2)

    output_dir.mkdir(parents=True, exist_ok=True)

    requested = [s.strip() for s in splits.split(",") if s.strip()]
    chosen = [s for s in requested if s in SPLIT_CSV_NAMES]
    if not chosen:
        console.print(
            f"[red]No valid splits in {splits!r}. "
            f"Available: {list(SPLIT_CSV_NAMES)}[/red]",
        )
        sys.exit(2)

    # Use num_hands=1 — V1 gestures are single-handed swipes. The runtime
    # uses the same single-hand selection (highest detection_confidence
    # primary_hand), so train and serve match.
    landmarker = HandLandmarkerWrapper(num_hands=1)

    with landmarker:
        for split_name in chosen:
            csv_path = csv_path_for_split(root, split_name)
            videos = videos_dir_for_split(root, split_name)
            console.print(f"\n[bold cyan]=== Split: {split_name} ===[/bold cyan]")
            console.print(f"Parsing {csv_path}…")

            records, parse_stats = parse_jester_csv(
                csv_path,
                videos_dir=videos,
                sigil_label_for=JESTER_TO_SIGIL_V1,
                split=split_name,
            )
            console.print(
                f"  Parsed {parse_stats.total_records} V1 records "
                f"(skipped {parse_stats.skipped_unknown_label} non-V1 labels, "
                f"{parse_stats.skipped_missing_video} missing videos)",
            )
            for cls, n in sorted(parse_stats.per_sigil_class.items()):
                console.print(f"    {cls:<22} {n}")

            if max_per_class is not None:
                records = filter_to_v1_classes(
                    records, max_per_class=max_per_class,
                )
                console.print(
                    f"  Capped at {max_per_class}/class → {len(records)} records",
                )

            if not records:
                console.print(
                    f"[yellow]Split {split_name!r}: no records — skipping.[/yellow]",
                )
                continue

            console.print(
                f"  Preprocessing {len(records)} videos through MediaPipe…",
            )
            console.print(
                f"  [dim](progress logged every 100 videos to structlog)[/dim]",
            )

            try:
                sequences, labels, video_ids, stats = preprocess_records(
                    records,
                    landmarker=landmarker,
                )
            except KeyboardInterrupt:
                console.print(
                    "\n[yellow]Preprocessing interrupted (Ctrl+C). "
                    "No parquet written for this split.[/yellow]",
                )
                sys.exit(130)

            if stats.successful == 0:
                console.print(
                    f"[red]Split {split_name!r}: 0 successful sequences. "
                    f"Check MediaPipe setup.[/red]",
                )
                continue

            out_path = output_dir / f"{split_name}.parquet"
            write_result = write_temporal_records(
                sequences=sequences,
                labels=labels,
                video_ids=video_ids,
                split=split_name,
                output_path=out_path,
            )
            console.print(
                f"  [green]{split_name}:[/green] {write_result.rows} sequences → "
                f"{write_result.path}  ({write_result.size_mb:.1f} MB, "
                f"success rate {stats.success_rate:.1%}, "
                f"{stats.elapsed_seconds / 60:.1f} min)",
            )

    console.print("\n[bold green]Build complete.[/bold green]")
    console.print(
        f"[dim]Next: train the V1 model (V1 patch 2 — coming separately).[/dim]",
    )


# ---------------------------------------------------------------------------
# summarise
# ---------------------------------------------------------------------------


@dataset_v1.command("summarise")
@click.argument("parquet_path", type=click.Path(exists=True, path_type=Path))
def summarise_cmd(parquet_path: Path) -> None:
    """Show class + sequence-count summary for a V1 parquet."""
    try:
        from sigil.intelligence.dataset.temporal_storage import (
            read_temporal_arrays,
        )
    except ImportError as exc:
        console.print(f"[red]Missing extras:[/red] {exc}")
        sys.exit(2)

    sequences, labels, video_ids = read_temporal_arrays(parquet_path)
    if len(labels) == 0:
        console.print(f"[yellow]{parquet_path} is empty.[/yellow]")
        return

    table = Table(title=f"V1 landmark sequences: {parquet_path}")
    table.add_column("Class")
    table.add_column("Count", justify="right")
    table.add_column("% of total", justify="right")

    counts = Counter(labels.tolist())
    total = sum(counts.values())
    for cls in sorted(counts):
        c = counts[cls]
        table.add_row(cls, str(c), f"{c / total:.1%}")

    console.print(table)
    console.print(
        f"\n[bold]Total:[/bold] {total} sequences, shape "
        f"{sequences.shape} ({sequences.nbytes / (1024 * 1024):.1f} MB in memory)",
    )


__all__ = ["dataset_v1"]
