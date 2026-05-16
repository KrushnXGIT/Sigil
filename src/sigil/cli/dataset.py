"""`sigil dataset ...` — CLI for the annotation-based dataset pipeline.

Per ADR-0003, Phase 2a builds training data from HaGRIDv2's annotation
archive (JSON with pre-computed 2D landmarks) — not from downloaded images.

Commands:
    sigil dataset download         — fetch + extract annotations.zip
    sigil dataset build            — parse annotations -> train/val/test parquet
    sigil dataset summarise        — class + user distribution of a parquet
    sigil dataset sample-images    — pull a small image sample (verify step)
    sigil dataset verify-alignment — measure our-MediaPipe vs HaGRID landmarks
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

# Default Tier 1 gesture set, in Sigil naming.
DEFAULT_GESTURES: tuple[str, ...] = (
    "open_palm",  # reserved
    "thumbs_up",  # reserved
    "thumbs_down",  # reserved
    "fist",
    "peace",
    "ok",
    "no_gesture",  # explicit negative class — essential
)


def _split_gestures(value: str) -> list[str]:
    return [g.strip() for g in value.split(",") if g.strip()]


@click.group()
def dataset() -> None:
    """Dataset prep — HaGRIDv2 annotations -> landmark parquet cache."""


# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------


@dataset.command("download")
@click.option(
    "-d",
    "--dest",
    default="datasets/raw",
    show_default=True,
    type=click.Path(path_type=Path),
    help="Where annotations.zip + extracted JSON tree will live.",
)
@click.option("--force", is_flag=True, help="Re-download even if the archive exists.")
@click.option("--no-extract", is_flag=True, help="Download but don't unzip.")
def download_cmd(dest: Path, force: bool, no_extract: bool) -> None:
    """Download + extract HaGRIDv2's annotations.zip (JSON with 2D landmarks)."""
    try:
        from sigil.intelligence.dataset.hagrid import download_annotations

        download_annotations(dest, skip_existing=not force, extract=not no_extract)
        console.print(f"[green]Annotations ready under {dest}[/green]")
    except Exception as exc:
        console.print(f"[red]Download failed:[/red] {exc}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


@dataset.command("build")
@click.option(
    "-a",
    "--annotations-dir",
    default="datasets/raw",
    show_default=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory annotations.zip extracted into.",
)
@click.option(
    "-g",
    "--gestures",
    default=",".join(DEFAULT_GESTURES),
    help="Comma-separated Sigil gesture names.",
)
@click.option(
    "-o",
    "--output-dir",
    default="datasets/processed",
    show_default=True,
    type=click.Path(path_type=Path),
    help="Where train/val/test parquet files will be written.",
)
@click.option(
    "--max-per-gesture",
    type=int,
    help="Cap records per gesture per split — useful for smoke tests.",
)
@click.option(
    "--no-normalize",
    is_flag=True,
    help="Skip normalization (NOT RECOMMENDED — train/serve diverge).",
)
def build_cmd(
    annotations_dir: Path,
    gestures: str,
    output_dir: Path,
    max_per_gesture: int | None,
    no_normalize: bool,
) -> None:
    """Parse HaGRIDv2 annotations into train/val/test parquet caches."""
    try:
        from sigil.intelligence.dataset.annotations import SPLITS, parse_split
        from sigil.intelligence.dataset.hagrid import sigil_to_hagrid
        from sigil.intelligence.dataset.storage import write_landmark_records
    except ImportError as exc:
        console.print(f"[red]Missing extras:[/red] {exc}")
        sys.exit(2)

    sigil_gestures = _split_gestures(gestures)
    try:
        hagrid_to_sigil = sigil_to_hagrid(sigil_gestures)
    except Exception as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(2)

    output_dir.mkdir(parents=True, exist_ok=True)
    total_written = 0

    for split in SPLITS:
        try:
            records, stats = parse_split(
                annotations_dir,
                split,
                hagrid_to_sigil=hagrid_to_sigil,
                normalize=not no_normalize,
                max_per_gesture=max_per_gesture,
            )
        except Exception as exc:
            console.print(f"[red]Parse failed for split {split!r}:[/red] {exc}")
            sys.exit(1)

        if not records:
            console.print(f"[yellow]Split {split!r}: no records produced — skipping.[/yellow]")
            continue

        out_path = output_dir / f"{split}.parquet"
        n = write_landmark_records(records, out_path)
        total_written += n
        console.print(
            f"[green]{split}:[/green] {n} records -> {out_path}  "
            f"(emit rate {stats.emit_rate:.1%}, "
            f"skipped: no_lm={stats.skipped_no_landmarks} "
            f"bad_shape={stats.skipped_bad_shape} "
            f"degenerate={stats.skipped_degenerate})"
        )

    if total_written == 0:
        console.print("[red]No records written across any split.[/red]")
        sys.exit(1)
    console.print(f"\n[bold green]Done — {total_written} records total.[/bold green]")


# ---------------------------------------------------------------------------
# summarise
# ---------------------------------------------------------------------------


@dataset.command("summarise")
@click.argument("parquet_path", type=click.Path(exists=True, path_type=Path))
def summarise_cmd(parquet_path: Path) -> None:
    """Show gesture-class + user distribution for a landmark parquet."""
    try:
        from sigil.intelligence.dataset.storage import read_landmark_arrays
    except ImportError as exc:
        console.print(f"[red]Missing extras:[/red] {exc}")
        sys.exit(2)

    _X, y, users = read_landmark_arrays(parquet_path)
    table = Table(title=f"Landmark cache: {parquet_path}")
    table.add_column("Gesture")
    table.add_column("Count", justify="right")
    table.add_column("% of total", justify="right")
    table.add_column("Unique users", justify="right")

    counts = Counter(y.tolist())
    total = sum(counts.values())
    user_sets: dict[str, set[str]] = {}
    for gesture, user in zip(y.tolist(), users.tolist(), strict=True):
        user_sets.setdefault(gesture, set()).add(user)

    for gesture in sorted(counts):
        c = counts[gesture]
        table.add_row(
            gesture,
            str(c),
            f"{c / total:.1%}",
            str(len(user_sets[gesture])),
        )
    console.print(table)
    console.print(
        f"\n[bold]Total:[/bold] {total} samples, " f"{len(set(users.tolist()))} unique users"
    )


# ---------------------------------------------------------------------------
# sample-images
# ---------------------------------------------------------------------------


@dataset.command("sample-images")
@click.option(
    "-g",
    "--gestures",
    default=",".join(DEFAULT_GESTURES),
    help="Comma-separated Sigil gesture names.",
)
@click.option(
    "-d",
    "--dest",
    default="datasets/image_sample",
    show_default=True,
    type=click.Path(path_type=Path),
    help="Where sampled images land.",
)
@click.option(
    "-n",
    "--n-per-gesture",
    default=50,
    show_default=True,
    type=int,
    help="Images to pull per gesture.",
)
def sample_images_cmd(gestures: str, dest: Path, n_per_gesture: int) -> None:
    """Pull a SMALL HaGRID image sample via HTTP range requests (verify step)."""
    try:
        from sigil.intelligence.dataset.hagrid import sample_images

        result = sample_images(
            _split_gestures(gestures),
            dest,
            n_per_gesture=n_per_gesture,
        )
        console.print(f"[green]Sampled {result.total} images to {result.image_dir}[/green]")
        for cls, paths in result.images_by_gesture.items():
            console.print(f"  {cls}: {len(paths)}")
    except Exception as exc:
        console.print(f"[red]Sampling failed:[/red] {exc}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# verify-alignment
# ---------------------------------------------------------------------------


@dataset.command("verify-alignment")
@click.option(
    "-a",
    "--annotations-dir",
    default="datasets/raw",
    show_default=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "-i",
    "--image-dir",
    default="datasets/image_sample",
    show_default=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Sampled images (from `sample-images` or a manual download).",
)
@click.option(
    "-g",
    "--gestures",
    default=",".join(DEFAULT_GESTURES),
    help="Comma-separated Sigil gesture names.",
)
@click.option(
    "--split", default="train", show_default=True, type=click.Choice(["train", "val", "test"])
)
@click.option("--max-images", default=300, show_default=True, type=int)
def verify_alignment_cmd(
    annotations_dir: Path,
    image_dir: Path,
    gestures: str,
    split: str,
    max_images: int,
) -> None:
    """Measure how closely HaGRIDv2's annotation landmarks match our MediaPipe.

    This is the ADR-0003 safety net. A PASS (mean error within ~3% of palm
    span) means it's safe to train on HaGRIDv2's 2D annotations directly.
    """
    try:
        from sigil.intelligence.dataset.hagrid import sigil_to_hagrid
        from sigil.intelligence.dataset.verify import verify_alignment
    except ImportError as exc:
        console.print(f"[red]Missing extras:[/red] {exc}")
        sys.exit(2)

    try:
        hagrid_to_sigil = sigil_to_hagrid(_split_gestures(gestures))
        report = verify_alignment(
            annotations_dir,
            image_dir,
            hagrid_to_sigil=hagrid_to_sigil,
            split=split,
            max_images=max_images,
        )
    except Exception as exc:
        console.print(f"[red]Verification failed:[/red] {exc}")
        sys.exit(1)

    colour = "green" if report.passes else "yellow"
    console.print(f"[{colour}]{report.summary()}[/{colour}]")

    # Per-landmark breakdown — helps spot if a specific fingertip is the
    # source of disagreement (vs uniform skew).
    table = Table(title="Per-landmark mean error (palm-scale units)")
    table.add_column("Landmark")
    table.add_column("Mean error", justify="right")
    from sigil.perception.types import LANDMARK_NAMES

    worst = sorted(
        range(len(report.per_landmark_mean)),
        key=lambda i: report.per_landmark_mean[i],
        reverse=True,
    )[:5]
    for i in worst:
        table.add_row(LANDMARK_NAMES[i], f"{report.per_landmark_mean[i]:.4f}")
    console.print("Top-5 worst-aligned landmarks:")
    console.print(table)

    if not report.passes:
        console.print(
            "[yellow]Mean error exceeds the 3% threshold. Consider "
            "self-extraction (the full image download) before a training "
            "run, or investigate the worst landmarks above.[/yellow]"
        )
        sys.exit(3)


__all__ = ["dataset"]

# ---------------------------------------------------------------------------
# inspect-alignment
# ---------------------------------------------------------------------------


@dataset.command("inspect-alignment")
@click.option(
    "-a",
    "--annotations-dir",
    default="datasets/raw",
    show_default=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "-i",
    "--image-dir",
    default="datasets/image_sample",
    show_default=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Sampled images (from `sample-images` or a manual download).",
)
@click.option(
    "-g",
    "--gestures",
    default=",".join(DEFAULT_GESTURES),
    help="Comma-separated Sigil gesture names.",
)
@click.option(
    "--split", default="train", show_default=True, type=click.Choice(["train", "val", "test"])
)
@click.option("--max-images", default=300, show_default=True, type=int)
@click.option(
    "-k",
    "--top-k",
    default=8,
    show_default=True,
    type=int,
    help="Number of worst-aligned pairs to render in the report.",
)
@click.option(
    "--rank-by",
    default="fingertip_mean",
    show_default=True,
    type=click.Choice(["max", "mean", "fingertip_mean"]),
    help="Metric used to pick the top-K pairs.",
)
@click.option(
    "-o",
    "--output",
    default="alignment_inspection.html",
    show_default=True,
    type=click.Path(path_type=Path),
    help="Self-contained HTML report path.",
)
def inspect_alignment_cmd(
    annotations_dir: Path,
    image_dir: Path,
    gestures: str,
    split: str,
    max_images: int,
    top_k: int,
    rank_by: str,
    output: Path,
) -> None:
    """Diagnose whether an alignment residual is systematic or random.

    Generates a single HTML file: image overlays of the worst-aligned hand
    pairs (HaGRID in red, ours in blue, displacement arrows in amber) plus
    a per-landmark directional analysis that classifies each landmark as
    systematic / random / mixed.
    """
    try:
        from sigil.intelligence.dataset.hagrid import sigil_to_hagrid
        from sigil.intelligence.dataset.inspect import build_html_report, inspect_alignment
    except ImportError as exc:
        console.print(f"[red]Missing extras:[/red] {exc}")
        sys.exit(2)

    try:
        hagrid_to_sigil = sigil_to_hagrid(_split_gestures(gestures))
        report = inspect_alignment(
            annotations_dir,
            image_dir,
            hagrid_to_sigil=hagrid_to_sigil,
            split=split,
            max_images=max_images,
            top_k=top_k,
            rank_by=rank_by,
        )
    except Exception as exc:
        console.print(f"[red]Inspection failed:[/red] {exc}")
        sys.exit(1)

    build_html_report(report, output)

    # CLI summary so you don't have to open the HTML to see the headline.
    from sigil.intelligence.dataset.inspect import FINGERTIP_INDICES
    from sigil.perception.types import LANDMARK_NAMES

    classes = [report.directional.classify(i) for i in FINGERTIP_INDICES]
    sys_count = sum(1 for c in classes if c == "systematic")
    rand_count = sum(1 for c in classes if c == "random")
    if sys_count >= 3:
        verdict, colour = "SYSTEMATIC", "red"
    elif rand_count >= 3:
        verdict, colour = "RANDOM", "green"
    else:
        verdict, colour = "MIXED", "yellow"
    console.print(f"[{colour}]Fingertip verdict: {verdict}[/{colour}]")

    table = Table(title="Per-fingertip directional analysis")
    table.add_column("Fingertip")
    table.add_column("|mean|", justify="right")
    table.add_column("noise", justify="right")
    table.add_column("ratio", justify="right")
    table.add_column("verdict")
    for i in FINGERTIP_INDICES:
        mag = report.directional.mean_magnitude(i)
        noise = report.directional.noise_magnitude(i)
        ratio = report.directional.systematic_ratio(i)
        ratio_str = "inf" if ratio == float("inf") else f"{ratio:.2f}"
        table.add_row(
            LANDMARK_NAMES[i],
            f"{mag:.4f}",
            f"{noise:.4f}",
            ratio_str,
            report.directional.classify(i),
        )
    console.print(table)
    console.print(
        f"[cyan]Report:[/cyan] {output}  ({report.total_pairs} pairs, "
        f"top-{len(report.top_pairs)} by {rank_by})"
    )
