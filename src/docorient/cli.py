from __future__ import annotations

import argparse
import sys
from pathlib import Path

from docorient._version import __version__
from docorient.batch.processor import process_directory
from docorient.config import OrientationConfig
from docorient.detection.secondary import is_secondary_engine_available


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="docorient",
        description="Detect and correct document image orientation.",
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing document images to process.",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        dest="output_dir",
        help="Output directory for corrected images. Default: auto-generated UUID.",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=None,
        help="Number of parallel worker processes. Default: cpu_count - 2.",
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=0,
        help="Maximum number of images to process. 0 means all.",
    )
    parser.add_argument(
        "--quality", "-q",
        type=int,
        default=92,
        help="Output JPEG quality (1-100). Default: 92.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=2.0,
        help="Secondary engine confidence threshold. Default: 2.0.",
    )
    parser.add_argument(
        "--no-secondary",
        action="store_true",
        default=False,
        help="Disable secondary engine (primary engine only).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        default=False,
        help="Disable resume from previous run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Only show what would be done, without processing.",
    )
    parser.add_argument(
        "--version", "-V",
        action="version",
        version=f"docorient {__version__}",
    )
    return parser


def _print_dry_run_info(input_path: Path, config: OrientationConfig, limit: int) -> None:
    from docorient.batch.scanner import scan_directory

    temp_output = input_path.parent / "__dry_run_temp__"
    pages_by_source = scan_directory(
        input_path,
        temp_output,
        supported_extensions=config.supported_extensions,
        limit=limit,
    )

    total_files = len(pages_by_source)
    total_pages = sum(len(pages) for pages in pages_by_source.values())

    print(f"\n{'=' * 60}")
    print("  DRY RUN - No changes will be made")
    print(f"{'=' * 60}")
    print(f"  Input:            {input_path}")
    print(f"  Files:            {total_files} source documents")
    print(f"  Pages:            {total_pages} images")
    print(f"  Workers:          {config.effective_workers}")
    print(f"  Secondary engine: {'enabled' if is_secondary_engine_available() else 'disabled'}")
    print(f"  Quality:          {config.output_quality}")
    print(f"  Resume:           {'enabled' if config.resume_enabled else 'disabled'}")
    print(f"{'=' * 60}\n")


def _print_summary(summary) -> None:
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Output:               {summary.output_directory}")
    print(f"  Files processed:      {summary.total_files}")
    print(f"  Total pages:          {summary.total_pages}")
    print(f"  Already correct (0°): {summary.already_correct}")
    print(f"  Corrected:            {summary.corrected}")
    if summary.corrected_by_majority > 0:
        print(f"    (majority vote):    {summary.corrected_by_majority}")
    print(f"  Errors:               {summary.errors}")
    print(f"{'=' * 60}\n")

    corrected_pages = [
        page for page in summary.pages
        if page.orientation.angle != 0 and page.error is None
    ]
    if corrected_pages:
        print("Corrections applied:")
        for page in corrected_pages:
            print(f"  {page.image_name}: {page.orientation.angle}° ({page.orientation.method})")
        print()


def main() -> None:
    parser = _build_argument_parser()
    arguments = parser.parse_args()

    input_path = Path(arguments.input_dir)
    if not input_path.is_dir():
        print(f"Error: '{arguments.input_dir}' is not a valid directory.", file=sys.stderr)
        sys.exit(1)

    secondary_threshold = arguments.confidence if not arguments.no_secondary else float("inf")

    config = OrientationConfig(
        secondary_confidence_threshold=secondary_threshold,
        output_quality=arguments.quality,
        workers=arguments.workers,
        resume_enabled=not arguments.no_resume,
    )

    if arguments.dry_run:
        _print_dry_run_info(input_path, config, arguments.limit)
        return

    print(f"\n{'=' * 60}")
    print(f"  docorient v{__version__}")
    print(f"{'=' * 60}")
    print(f"  Input:            {input_path}")
    print(f"  Output:           {arguments.output_dir or 'auto (UUID)'}")
    print(f"  Workers:          {config.effective_workers}")
    print(f"  Secondary engine: {'disabled' if arguments.no_secondary else 'enabled'}")
    print(f"  Quality:          {config.output_quality}")
    print(f"{'=' * 60}\n")

    summary = process_directory(
        input_dir=input_path,
        output_dir=arguments.output_dir,
        config=config,
        limit=arguments.limit,
    )

    _print_summary(summary)


if __name__ == "__main__":
    main()
