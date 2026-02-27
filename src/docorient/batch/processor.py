from __future__ import annotations

import multiprocessing
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

from docorient.batch.scanner import ScannedPage, scan_directory
from docorient.batch.worker import initialize_worker, process_batch
from docorient.config import RESUME_LOG_FILENAME, OrientationConfig
from docorient.exceptions import BatchProcessingError
from docorient.types import BatchSummary, PageResult


def _load_completed_sources(resume_log_path: Path) -> set[str]:
    if not resume_log_path.exists():
        return set()
    with open(resume_log_path) as resume_log:
        return {line.strip() for line in resume_log if line.strip()}


def _distribute_into_batches(
    items: list[tuple[str, list[ScannedPage]]],
    batch_count: int,
) -> list[list[tuple[str, list[ScannedPage]]]]:
    batches: list[list[tuple[str, list[ScannedPage]]]] = [[] for _ in range(batch_count)]
    for item_index, item in enumerate(items):
        target_batch = item_index % batch_count
        batches[target_batch].append(item)
    return batches


def _build_summary(
    input_directory: str,
    output_directory: str,
    total_files: int,
    all_page_results: dict[str, list[PageResult]],
    source_file_names: list[str],
) -> BatchSummary:
    all_pages: list[PageResult] = []
    already_correct_count = 0
    corrected_count = 0
    corrected_by_majority_count = 0
    error_count = 0

    for source_name in source_file_names:
        for page_result in all_page_results.get(source_name, []):
            all_pages.append(page_result)
            if page_result.error is not None:
                error_count += 1
            elif page_result.orientation.angle != 0:
                corrected_count += 1
                if "->majority" in page_result.orientation.method:
                    corrected_by_majority_count += 1
            else:
                already_correct_count += 1

    return BatchSummary(
        input_directory=input_directory,
        output_directory=output_directory,
        total_files=total_files,
        total_pages=len(all_pages),
        already_correct=already_correct_count,
        corrected=corrected_count,
        corrected_by_majority=corrected_by_majority_count,
        errors=error_count,
        pages=tuple(all_pages),
    )


def _resolve_output_directory(
    input_path: Path, output_dir: str | Path | None
) -> Path:
    if output_dir is None:
        return input_path.parent / str(uuid.uuid4())
    return Path(output_dir).resolve()


def _filter_pending_sources(
    pages_by_source: dict[str, list[ScannedPage]],
    source_file_names: list[str],
    resume_log_path: Path,
    resume_enabled: bool,
) -> list[tuple[str, list[ScannedPage]]]:
    already_completed = set()
    if resume_enabled:
        already_completed = _load_completed_sources(resume_log_path)

    return [
        (source_name, pages_by_source[source_name])
        for source_name in source_file_names
        if source_name not in already_completed
    ]


def _run_parallel_processing(
    pending_sources: list[tuple[str, list[ScannedPage]]],
    config: OrientationConfig,
    resume_log_path: Path,
    show_progress: bool,
) -> dict[str, list[PageResult]]:
    worker_count = min(config.effective_workers, len(pending_sources))
    batches = _distribute_into_batches(pending_sources, worker_count)

    progress_counter = multiprocessing.Value("i", 0)
    progress_lock = multiprocessing.Lock()
    config_as_dict = asdict(config)

    worker_pool = multiprocessing.Pool(
        processes=worker_count,
        initializer=initialize_worker,
        initargs=(progress_counter, progress_lock, str(resume_log_path), config_as_dict),
        maxtasksperchild=1,
    )

    async_results = [
        worker_pool.apply_async(process_batch, (batch,)) for batch in batches
    ]
    worker_pool.close()

    progress_bar = None
    if show_progress:
        progress_bar = tqdm(
            total=len(pending_sources),
            desc="Correcting",
            unit="file",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

    try:
        while not all(async_result.ready() for async_result in async_results):
            if progress_bar is not None:
                progress_bar.n = progress_counter.value
                progress_bar.refresh()
            time.sleep(0.3)
    except KeyboardInterrupt:
        worker_pool.terminate()
        worker_pool.join()
        if progress_bar is not None:
            progress_bar.close()
        sys.exit(1)

    if progress_bar is not None:
        progress_bar.n = progress_counter.value
        progress_bar.refresh()
        progress_bar.close()

    all_page_results: dict[str, list[PageResult]] = {}
    for async_result in async_results:
        try:
            batch_results = async_result.get(timeout=60)
            for source_name, page_results in batch_results:
                all_page_results[source_name] = page_results
        except Exception as pool_error:
            raise BatchProcessingError(
                f"Worker pool error: {pool_error}"
            ) from pool_error

    worker_pool.join()
    return all_page_results


def process_directory(
    input_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    config: OrientationConfig | None = None,
    limit: int = 0,
    show_progress: bool = True,
) -> BatchSummary:
    effective_config = config or OrientationConfig()
    input_path = Path(input_dir).resolve()
    output_path = _resolve_output_directory(input_path, output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pages_by_source = scan_directory(
        input_path,
        output_path,
        supported_extensions=effective_config.supported_extensions,
        limit=limit,
    )

    source_file_names = list(pages_by_source.keys())
    total_files = len(source_file_names)

    if not pages_by_source:
        return _build_summary(str(input_path), str(output_path), 0, {}, [])

    resume_log_path = output_path / RESUME_LOG_FILENAME

    pending_sources = _filter_pending_sources(
        pages_by_source, source_file_names, resume_log_path, effective_config.resume_enabled
    )

    if not pending_sources:
        return _build_summary(
            str(input_path), str(output_path), total_files, {}, source_file_names
        )

    all_page_results = _run_parallel_processing(
        pending_sources, effective_config, resume_log_path, show_progress
    )

    return _build_summary(
        str(input_path), str(output_path), total_files, all_page_results, source_file_names
    )
