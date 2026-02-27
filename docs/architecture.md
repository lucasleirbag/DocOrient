# Architecture

## Package Structure

```
src/docorient/
├── __init__.py              Public API re-exports
├── _version.py              Version string
├── _imaging.py              ImageIO class for image operations
├── config.py                OrientationConfig dataclass
├── types.py                 Result dataclasses
├── exceptions.py            Exception hierarchy
├── rotation.py              Image rotation utility
├── voting.py                Majority voting logic
├── correction.py            correct_image, correct_document_pages
├── cli.py                   CLI entry point
├── detection/
│   ├── __init__.py          Re-exports detection API
│   ├── base.py              DetectionEngine Protocol
│   ├── engine.py            DetectionPipeline orchestrator
│   ├── primary.py           PrimaryEngine (90°/270°)
│   └── secondary.py         SecondaryEngine (180°)
└── batch/
    ├── __init__.py          Re-exports process_directory
    ├── scanner.py           Directory scanning and grouping
    ├── worker.py            WorkerContext and multiprocessing logic
    └── processor.py         Batch orchestrator with resume and progress
```

## Detection Pipeline

```
detect_orientation()
        │
        ▼
DetectionPipeline.run()
Iterates through list[DetectionEngine]
        │
        ▼
PrimaryEngine.detect()
Analyzes pixel density distribution to determine text alignment.
        │
        ├── angle ∈ {90, 270} ──► Return result immediately
        │
        └── angle = 0 (aligned) ──► SecondaryEngine.detect()
                                            │
                                  is_available()?
                                            │
                               Yes ─────────┴───────── No
                                ▼                       ▼
                     Runs secondary analysis    Return 0° (no change)
                     Checks for 180° inversion
                     with confidence scoring
                                │
                  confidence >= threshold?
                                │
                    Yes ────────┴──────── No
                     ▼                    ▼
              Return 180°           Return 0°
```

## Engine Architecture

The detection system is built on a `DetectionEngine` Protocol:

```
DetectionEngine (Protocol)
├── name: str
└── detect(image, config) → OrientationResult | None

PrimaryEngine implements DetectionEngine
├── Always returns OrientationResult (never None)
└── Detects 0°, 90°, 270° via energy analysis

SecondaryEngine implements DetectionEngine
├── Returns None when unavailable or low confidence
└── Detects 180° via optional OCR dependency

DetectionPipeline
├── Holds list[DetectionEngine] (default: [PrimaryEngine(), SecondaryEngine()])
└── Executes engines in sequence with short-circuit logic
```

## Batch Processing Pipeline

```
process_directory()
        │
        ▼
   _resolve_output_directory()
   Resolves explicit path or generates UUID
        │
        ▼
   scan_directory()          ← scanner.py
   Groups images by
   source document name
        │
        ▼
   _filter_pending_sources()
   Loads resume log, skips completed
        │
        ▼
   _run_parallel_processing()
   ┌─────────────────────────────────────┐
   │  multiprocessing.Pool              │
   │  WorkerContext encapsulates state   │
   │                                     │
   │  For each source in batch:          │
   │  1. Run detection per page          │
   │  2. Apply majority voting           │
   │  3. Rotate and save each page       │
   │  4. Append to resume log            │
   └─────────────────────────────────────┘
        │
        ▼
   Collect results
   Build BatchSummary
```

## Majority Voting

When a document has multiple pages, individual detections may disagree.
Majority voting resolves this:

1. Collect all `reliable` detections from the document's pages
2. Find the most common angle (`Counter.most_common`)
3. Override any unreliable detection that differs from the majority

Implemented in `voting.apply_majority_voting()`, reused by both
`correct_document_pages()` and `batch/worker._process_single_source()`.

## Multiprocessing Design

- **WorkerContext dataclass** encapsulates all shared state in a single object
- **Progress tracking** via `multiprocessing.Value` + `multiprocessing.Lock`
- **Resume log** written atomically per source file, protected by the shared lock
- **`maxtasksperchild=1`** prevents memory accumulation in long jobs
- **Round-robin batch distribution** across workers

## Configuration Immutability

`OrientationConfig` is `frozen=True, slots=True` — immutable after creation,
safe to share across threads, minimal memory footprint.
