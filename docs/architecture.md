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
│   ├── primary.py           PrimaryEngine (axis detection)
│   └── flip_classifier.py   FlipClassifierEngine (180° CNN)
├── models/
│   └── orientation_detector.onnx   MobileNetV2 ONNX model (~8.5 MB)
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
Iterates through [PrimaryEngine, FlipClassifierEngine]
        │
        ▼
PrimaryEngine.detect()
Analyzes pixel energy distribution to classify axis:
  - Horizontal text (0° or 180°)
  - Vertical text (90° or 270°)
        │
        ▼
FlipClassifierEngine via _resolve_flip()
        │
        ├── If horizontal → CNN classifies directly
        │   ├── Upright → return 0°
        │   └── Upside-down → return 180°
        │
        └── If vertical → normalize by rotating 90° CCW, then CNN classifies
            ├── Upright (was at 90°) → return 90°
            └── Upside-down (was at 270°) → return 270°
```

## Engine Architecture

```
DetectionEngine (Protocol)
├── name: str
└── detect(image, config) → OrientationResult | None

PrimaryEngine implements DetectionEngine
├── Always returns OrientationResult (never None)
├── Computes horizontal/vertical energy via pixel projection
└── Classifies text axis with ~97% accuracy

FlipClassifierEngine implements DetectionEngine
├── Returns None when confidence below threshold
├── Uses embedded MobileNetV2 ONNX model
├── Binary classification: upright vs upside-down
├── Inference via ONNX Runtime (~5ms on CPU)
├── Trained on 3,000+ diverse documents (96.8% accuracy)
└── Model input: 224×224 RGB, ImageNet-normalized

DetectionPipeline
├── Holds list[DetectionEngine] (default: [PrimaryEngine(), FlipClassifierEngine()])
├── PrimaryEngine detects axis
└── FlipClassifier resolves flip direction within the detected axis
```

## FlipClassifier Details

The FlipClassifierEngine wraps a MobileNetV2 fine-tuned for binary classification:

- **Task**: Classify if a document image is upside-down (180° rotated)
- **Architecture**: MobileNetV2 with custom binary classifier head
- **Input**: 224×224 RGB image, center-cropped from 256×256 resize
- **Normalization**: ImageNet mean/std
- **Output**: Single logit → sigmoid → P(upside_down)
- **Export format**: ONNX (opset 13)
- **Model size**: ~8.5 MB
- **Inference**: ~5ms per image on CPU via ONNX Runtime
- **Standalone accuracy**: 96.8% (validation set)
- **Pipeline accuracy**: 98% across all 4 orientations (0°, 90°, 180°, 270°)

### Training Data

The model was trained on 3,049 document images from 5 diverse public datasets:

| Dataset | Count | Type |
|---|---|---|
| FUNSD | 199 | Scanned forms |
| Invoices & Receipts | 250 | Financial documents |
| CORD-v2 | 900 | Point-of-sale receipts |
| DocLayNet v1.1 | 1,000 | Varied document layouts |
| DocumentVQA | 700 | Diverse document types |

Each image generates 2 training samples (upright + 180° flipped), with 3x augmentation
(random crop, color jitter, blur, grayscale, perspective, erasing). Total effective
training samples: ~15,500.

For vertical images (90°/270°), the pipeline first normalizes by rotating 90° CCW,
then passes the horizontally-aligned image to the classifier.

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
