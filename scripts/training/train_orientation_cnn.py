from __future__ import annotations

import gc
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

import builtins
_original_print = builtins.print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _original_print(*args, **kwargs)

SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 20
FREEZE_BACKBONE_EPOCHS = 3
VALIDATION_SPLIT = 0.15
AUGMENTATION_FACTOR = 3
GRADIENT_CLIP_NORM = 1.0
LABEL_SMOOTHING = 0.05
MAX_IMAGE_DIMENSION = 800

LABEL_UPRIGHT = 0
LABEL_UPSIDE_DOWN = 1

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_OUTPUT_DIR = PROJECT_ROOT / "models"
IMAGE_CACHE_DIR = PROJECT_ROOT / "data" / "image_cache"


def set_reproducibility(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TrainingMetrics:
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    learning_rate: float
    epoch_duration_seconds: float


def constrain_and_save(image: Image.Image, output_path: Path) -> None:
    width, height = image.size
    if max(width, height) > MAX_IMAGE_DIMENSION:
        scale_factor = MAX_IMAGE_DIMENSION / max(width, height)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    image.save(str(output_path), format="JPEG", quality=90)


def download_datasets_if_needed(cache_dir: Path) -> dict[str, int]:
    existing_images = list(cache_dir.rglob("*.jpg"))
    if len(existing_images) > 100:
        print(f"  Cache encontrado com {len(existing_images)} imagens. Reutilizando...")
        breakdown = {}
        for subdir in sorted(cache_dir.iterdir()):
            if subdir.is_dir():
                count = len(list(subdir.glob("*.jpg")))
                if count > 0:
                    breakdown[subdir.name] = count
        return breakdown

    from datasets import load_dataset

    cache_dir.mkdir(parents=True, exist_ok=True)
    breakdown: dict[str, int] = {}

    print("  FUNSD (formularios escaneados)...")
    funsd_dir = cache_dir / "funsd"
    funsd_dir.mkdir(exist_ok=True)
    count = 0
    for split_name in ("train", "test"):
        split_data = load_dataset("nielsr/funsd", split=split_name)
        for example in split_data:
            pil_image = example.get("image")
            if pil_image is not None:
                constrain_and_save(pil_image.convert("RGB"), funsd_dir / f"{count:05d}.jpg")
                count += 1
        del split_data
        gc.collect()
    breakdown["funsd"] = count
    print(f"    -> {count} imagens")

    print("  Invoices & Receipts...")
    inv_dir = cache_dir / "invoices"
    inv_dir.mkdir(exist_ok=True)
    dataset = load_dataset("mychen76/invoices-and-receipts_ocr_v1", split="train", streaming=True)
    count = 0
    for example in dataset:
        if count >= 250:
            break
        pil_image = example.get("image")
        if pil_image is not None:
            constrain_and_save(pil_image.convert("RGB"), inv_dir / f"{count:05d}.jpg")
            count += 1
    breakdown["invoices"] = count
    print(f"    -> {count} imagens")
    gc.collect()

    print("  CORD-v2 (recibos)...")
    cord_dir = cache_dir / "cordv2"
    cord_dir.mkdir(exist_ok=True)
    count = 0
    for split_name in ("train", "validation"):
        split_data = load_dataset("naver-clova-ix/cord-v2", split=split_name)
        for example in split_data:
            pil_image = example.get("image")
            if pil_image is not None:
                constrain_and_save(pil_image.convert("RGB"), cord_dir / f"{count:05d}.jpg")
                count += 1
        del split_data
        gc.collect()
    breakdown["cord_v2"] = count
    print(f"    -> {count} imagens")

    print("  DocLayNet v1.1 (documentos variados)...")
    doc_dir = cache_dir / "doclaynet"
    doc_dir.mkdir(exist_ok=True)
    dataset = load_dataset("ds4sd/DocLayNet-v1.1", split="train", streaming=True)
    count = 0
    for example in dataset:
        if count >= 1000:
            break
        pil_image = example.get("image")
        if pil_image is not None:
            constrain_and_save(pil_image.convert("RGB"), doc_dir / f"{count:05d}.jpg")
            count += 1
            if count % 200 == 0:
                print(f"    ... {count}/1000")
    breakdown["doclaynet"] = count
    print(f"    -> {count} imagens")
    gc.collect()

    print("  DocumentVQA (documentos diversos)...")
    vqa_dir = cache_dir / "documentvqa"
    vqa_dir.mkdir(exist_ok=True)
    dataset = load_dataset("HuggingFaceM4/DocumentVQA", split="train", streaming=True)
    count = 0
    seen_ids: set[int] = set()
    for example in dataset:
        if count >= 700:
            break
        qid = example.get("questionId")
        if qid in seen_ids:
            continue
        seen_ids.add(qid)
        pil_image = example.get("image")
        if pil_image is not None:
            constrain_and_save(pil_image.convert("RGB"), vqa_dir / f"{count:05d}.jpg")
            count += 1
            if count % 200 == 0:
                print(f"    ... {count}/700")
    breakdown["documentvqa"] = count
    print(f"    -> {count} imagens")
    gc.collect()

    return breakdown


def collect_cached_image_paths(cache_dir: Path) -> list[Path]:
    return sorted(cache_dir.rglob("*.jpg"))


def split_paths_train_val(
    image_paths: list[Path],
    val_ratio: float = VALIDATION_SPLIT,
) -> tuple[list[Path], list[Path]]:
    indices = list(range(len(image_paths)))
    random.shuffle(indices)

    val_count = max(1, int(len(image_paths) * val_ratio))
    val_indices = set(indices[:val_count])

    train_paths = [path for idx, path in enumerate(image_paths) if idx not in val_indices]
    val_paths = [path for idx, path in enumerate(image_paths) if idx in val_indices]

    return train_paths, val_paths


class DiskOrientationDataset(Dataset):
    def __init__(
        self,
        image_paths: list[Path],
        transform: transforms.Compose,
        augmentation_factor: int = 1,
    ) -> None:
        self.image_paths = image_paths
        self.transform = transform
        self.augmentation_factor = augmentation_factor

    def __len__(self) -> int:
        return len(self.image_paths) * 2 * self.augmentation_factor

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        effective_index = index % (len(self.image_paths) * 2)
        path_index = effective_index // 2
        is_flipped = effective_index % 2 == 1

        image = Image.open(self.image_paths[path_index]).convert("RGB")

        if is_flipped:
            image = image.transpose(Image.Transpose.ROTATE_180)

        tensor_image = self.transform(image)
        label = LABEL_UPSIDE_DOWN if is_flipped else LABEL_UPRIGHT

        return tensor_image, label


def build_train_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.6, 1.0), ratio=(0.7, 1.4)),
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1)],
            p=0.6,
        ),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 3.0))],
            p=0.3,
        ),
        transforms.RandomGrayscale(p=0.3),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
        transforms.RandomPerspective(distortion_scale=0.15, p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.2)),
    ])


def build_val_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_model() -> nn.Module:
    backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)
    in_features = backbone.classifier[1].in_features
    backbone.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 1),
    )
    return backbone


def freeze_backbone(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False


def unfreeze_backbone(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


def smooth_labels(labels: torch.Tensor, smoothing: float = LABEL_SMOOTHING) -> torch.Tensor:
    return labels * (1.0 - smoothing) + 0.5 * smoothing


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    is_training: bool,
) -> tuple[float, float]:
    if is_training:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    batch_count = 0

    context = torch.no_grad() if not is_training else torch.enable_grad()

    with context:
        for batch_images, batch_labels in dataloader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.float().to(device)
            smoothed_labels = smooth_labels(batch_labels) if is_training else batch_labels

            if is_training and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            logits = model(batch_images).squeeze(1)
            loss = criterion(logits, smoothed_labels)

            if is_training and optimizer is not None:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
                optimizer.step()

            running_loss += loss.item() * batch_images.size(0)
            predictions = (torch.sigmoid(logits) > 0.5).long()
            correct_predictions += (predictions == batch_labels.long()).sum().item()
            total_samples += batch_images.size(0)
            batch_count += 1

            if is_training and batch_count % 100 == 0:
                partial_acc = correct_predictions / total_samples
                print(f"    batch {batch_count}: acc={partial_acc:.1%}", end="\r")

    average_loss = running_loss / total_samples if total_samples > 0 else 0.0
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0

    return average_loss, accuracy


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
) -> list[TrainingMetrics]:
    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    metrics_history: list[TrainingMetrics] = []
    best_val_accuracy = 0.0
    best_model_state = None
    patience_counter = 0
    patience_limit = 7

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start_time = time.perf_counter()

        if epoch == FREEZE_BACKBONE_EPOCHS + 1:
            print(f"\n  [Epoch {epoch}] Descongelando backbone para fine-tuning completo...")
            unfreeze_backbone(model)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=LEARNING_RATE * 0.1,
                weight_decay=1e-4,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=NUM_EPOCHS - FREEZE_BACKBONE_EPOCHS
            )

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, is_training=True,
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, None, device, is_training=False,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        epoch_duration = time.perf_counter() - epoch_start_time

        epoch_metrics = TrainingMetrics(
            epoch=epoch,
            train_loss=train_loss,
            train_accuracy=train_acc,
            val_loss=val_loss,
            val_accuracy=val_acc,
            learning_rate=current_lr,
            epoch_duration_seconds=epoch_duration,
        )
        metrics_history.append(epoch_metrics)

        phase_label = "FROZEN" if epoch <= FREEZE_BACKBONE_EPOCHS else "FULL"
        improvement_marker = ""
        if val_acc > best_val_accuracy:
            improvement_marker = " *BEST*"
            patience_counter = 0
        else:
            patience_counter += 1

        print(
            f"  Epoch {epoch:>2}/{NUM_EPOCHS} [{phase_label}] "
            f"| Train: loss={train_loss:.4f} acc={train_acc:.1%} "
            f"| Val: loss={val_loss:.4f} acc={val_acc:.1%} "
            f"| LR={current_lr:.2e} | {epoch_duration:.0f}s{improvement_marker}"
        )

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_model_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}

        if device.type == "mps":
            torch.mps.empty_cache()

        if patience_counter >= patience_limit and epoch > FREEZE_BACKBONE_EPOCHS + 3:
            print(f"\n  Early stopping: sem melhoria por {patience_limit} epochs")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n  Melhor modelo restaurado (val_acc={best_val_accuracy:.1%})")

    return metrics_history


def export_to_onnx(model: nn.Module, output_path: Path) -> None:
    model = model.cpu().eval()
    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["image"],
        output_names=["logit"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "logit": {0: "batch_size"},
        },
        opset_version=13,
        dynamo=False,
    )

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Modelo ONNX salvo: {output_path}")
    print(f"  Tamanho: {file_size_mb:.1f} MB")


def validate_onnx_model(
    onnx_path: Path,
    val_paths: list[Path],
) -> tuple[float, float]:
    session = ort.InferenceSession(str(onnx_path))
    input_name = session.get_inputs()[0].name

    val_transform = build_val_transforms()

    correct = 0
    total = 0
    inference_times: list[float] = []

    for image_path in val_paths:
        image = Image.open(image_path).convert("RGB")

        for is_flipped in (False, True):
            test_image = image.transpose(Image.Transpose.ROTATE_180) if is_flipped else image
            expected_label = LABEL_UPSIDE_DOWN if is_flipped else LABEL_UPRIGHT

            tensor = val_transform(test_image).unsqueeze(0).numpy()

            start_time = time.perf_counter()
            outputs = session.run(None, {input_name: tensor})
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            inference_times.append(elapsed_ms)

            logit = outputs[0][0][0]
            predicted = LABEL_UPSIDE_DOWN if logit > 0.0 else LABEL_UPRIGHT
            if predicted == expected_label:
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0.0
    avg_latency = float(np.mean(inference_times))

    return accuracy, avg_latency


def save_training_report(
    metrics_history: list[TrainingMetrics],
    onnx_accuracy: float,
    onnx_latency: float,
    onnx_path: Path,
    output_dir: Path,
    total_base_images: int,
    dataset_breakdown: dict[str, int],
) -> None:
    report = {
        "model": "MobileNetV2",
        "task": "binary_classification_upright_vs_upside_down",
        "image_size": IMAGE_SIZE,
        "num_epochs": len(metrics_history),
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "augmentation_factor": AUGMENTATION_FACTOR,
        "label_smoothing": LABEL_SMOOTHING,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "total_base_images": total_base_images,
        "dataset_breakdown": dataset_breakdown,
        "onnx_model_path": str(onnx_path),
        "onnx_model_size_mb": onnx_path.stat().st_size / (1024 * 1024),
        "onnx_validation_accuracy": onnx_accuracy,
        "onnx_avg_inference_ms": onnx_latency,
        "training_history": [
            {
                "epoch": metrics.epoch,
                "train_loss": metrics.train_loss,
                "train_accuracy": metrics.train_accuracy,
                "val_loss": metrics.val_loss,
                "val_accuracy": metrics.val_accuracy,
                "learning_rate": metrics.learning_rate,
                "epoch_duration_seconds": metrics.epoch_duration_seconds,
            }
            for metrics in metrics_history
        ],
    }

    report_path = output_dir / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Relatorio salvo: {report_path}")


def main() -> None:
    set_reproducibility()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print(f"\n{'=' * 72}")
    print(f"  DocOrient - Treinamento CNN v2 (Dataset Expandido + MPS)")
    print(f"{'=' * 72}")
    print(f"\n  Device: {device}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Label smoothing: {LABEL_SMOOTHING}")
    print(f"  Gradient clipping: {GRADIENT_CLIP_NORM}")
    print(f"  Modo: carregamento de disco (low memory)")

    print("\n[1/6] Preparando datasets (cache em disco)...")
    IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dataset_breakdown = download_datasets_if_needed(IMAGE_CACHE_DIR)
    all_image_paths = collect_cached_image_paths(IMAGE_CACHE_DIR)

    total_base_images = len(all_image_paths)
    print(f"\n  Total de imagens base: {total_base_images}")
    for dataset_name, count in dataset_breakdown.items():
        print(f"    {dataset_name}: {count}")

    print("\n[2/6] Dividindo train/val e criando datasets de disco...")
    train_paths, val_paths = split_paths_train_val(all_image_paths, VALIDATION_SPLIT)

    print(f"  Train: {len(train_paths)} imagens ({len(train_paths) * 2} pares)")
    print(f"  Val:   {len(val_paths)} imagens ({len(val_paths) * 2} pares)")

    train_dataset = DiskOrientationDataset(
        train_paths,
        transform=build_train_transforms(),
        augmentation_factor=AUGMENTATION_FACTOR,
    )
    val_dataset = DiskOrientationDataset(
        val_paths,
        transform=build_val_transforms(),
        augmentation_factor=1,
    )

    effective_train_size = len(train_dataset)
    print(f"  Train efetivo (x{AUGMENTATION_FACTOR} augmentation): {effective_train_size} amostras")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    print("\n[3/6] Construindo MobileNetV2 para classificacao binaria...")
    model = build_model().to(device)
    freeze_backbone(model)

    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total_params = sum(param.numel() for param in model.parameters())
    print(f"  Parametros treinaveis (fase frozen): {trainable_params:,}")
    print(f"  Parametros totais: {total_params:,}")

    print(f"\n[4/6] Treinando ({NUM_EPOCHS} epochs max, early stopping ativo)...\n")
    metrics_history = train_model(model, train_loader, val_loader, device)

    print("\n[5/6] Exportando para ONNX...")
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = MODEL_OUTPUT_DIR / "orientation_detector.onnx"
    export_to_onnx(model, onnx_path)

    print("\n[6/6] Validando modelo ONNX com onnxruntime...")
    onnx_accuracy, onnx_latency = validate_onnx_model(onnx_path, val_paths)
    print(f"  Acuracia ONNX (validacao): {onnx_accuracy:.1%}")
    print(f"  Latencia media: {onnx_latency:.1f} ms/imagem")

    save_training_report(
        metrics_history, onnx_accuracy, onnx_latency, onnx_path,
        MODEL_OUTPUT_DIR, total_base_images, dataset_breakdown,
    )

    total_training_time = sum(m.epoch_duration_seconds for m in metrics_history)

    print(f"\n{'=' * 72}")
    print(f"  TREINAMENTO CONCLUIDO")
    print(f"{'=' * 72}")
    print(f"  Modelo ONNX: {onnx_path}")
    print(f"  Acuracia: {onnx_accuracy:.1%}")
    print(f"  Latencia: {onnx_latency:.1f} ms")
    print(f"  Tamanho: {onnx_path.stat().st_size / (1024 * 1024):.1f} MB")
    print(f"  Tempo total treino: {total_training_time / 60:.1f} min")
    print(f"  Imagens base: {total_base_images}")
    print(f"  Epochs efetivos: {len(metrics_history)}")
    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    main()
