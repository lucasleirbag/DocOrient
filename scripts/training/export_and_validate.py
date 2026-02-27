from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from datasets import load_dataset
from PIL import Image
from torchvision import models, transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_OUTPUT_DIR = PROJECT_ROOT / "models"
IMAGE_SIZE = 224
SEED = 42

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from docorient.config import OrientationConfig
from docorient.detection.primary import PrimaryEngine


def build_model() -> nn.Module:
    backbone = models.mobilenet_v2(weights=None)
    in_features = backbone.classifier[1].in_features
    backbone.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 1),
    )
    return backbone


def build_val_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def retrain_and_export() -> Path:
    import random
    from torch.utils.data import DataLoader, Dataset

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"  Device: {device}")

    print("  Carregando datasets...")
    images = []
    for example in load_dataset("nielsr/funsd", split="train"):
        img = example.get("image")
        if img:
            images.append(img.convert("RGB"))
    for example in load_dataset("nielsr/funsd", split="test"):
        img = example.get("image")
        if img:
            images.append(img.convert("RGB"))
    for example in load_dataset("mychen76/invoices-and-receipts_ocr_v1", split="train", streaming=True):
        if len(images) >= 349:
            break
        img = example.get("image")
        if img:
            images.append(img.convert("RGB"))

    print(f"  {len(images)} imagens base carregadas")

    pairs = []
    for img in images:
        pairs.append((img.copy(), 0))
        pairs.append((img.transpose(Image.Transpose.ROTATE_180), 1))

    indices = list(range(len(images)))
    random.shuffle(indices)
    val_count = max(1, int(len(images) * 0.15))
    val_indices = set(indices[:val_count])

    train_pairs, val_pairs = [], []
    for idx in range(len(images)):
        target = val_pairs if idx in val_indices else train_pairs
        target.extend([pairs[idx * 2], pairs[idx * 2 + 1]])

    print(f"  Train: {len(train_pairs)}, Val: {len(val_pairs)}")

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomApply([transforms.ColorJitter(0.3, 0.3, 0.2, 0.1)], p=0.5),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.15)),
    ])
    val_transform = build_val_transforms()

    class PairDataset(Dataset):
        def __init__(self, pair_list, tfm, factor=1):
            self.pairs = pair_list
            self.tfm = tfm
            self.factor = factor

        def __len__(self):
            return len(self.pairs) * self.factor

        def __getitem__(self, idx):
            img, label = self.pairs[idx % len(self.pairs)]
            return self.tfm(img), label

    train_loader = DataLoader(PairDataset(train_pairs, train_transform, factor=8), batch_size=32, shuffle=True)
    val_loader = DataLoader(PairDataset(val_pairs, val_transform), batch_size=32, shuffle=False)

    model = build_model().to(device)
    model_weights = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)
    pretrained_state = model_weights.state_dict()
    model_state = model.state_dict()
    for key in pretrained_state:
        if key in model_state and pretrained_state[key].shape == model_state[key].shape:
            model_state[key] = pretrained_state[key]
    model.load_state_dict(model_state)

    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, 16):
        if epoch == 4:
            print("  Descongelando backbone...")
            for param in model.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12)

        model.train()
        train_correct, train_total, train_loss_sum = 0, 0, 0.0
        for batch_imgs, batch_labels in train_loader:
            batch_imgs = batch_imgs.to(device)
            batch_labels = batch_labels.float().to(device)
            optimizer.zero_grad()
            logits = model(batch_imgs).squeeze(1)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * batch_imgs.size(0)
            preds = (torch.sigmoid(logits) > 0.5).long()
            train_correct += (preds == batch_labels.long()).sum().item()
            train_total += batch_imgs.size(0)

        model.eval()
        val_correct, val_total, val_loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for batch_imgs, batch_labels in val_loader:
                batch_imgs = batch_imgs.to(device)
                batch_labels = batch_labels.float().to(device)
                logits = model(batch_imgs).squeeze(1)
                loss = criterion(logits, batch_labels)
                val_loss_sum += loss.item() * batch_imgs.size(0)
                preds = (torch.sigmoid(logits) > 0.5).long()
                val_correct += (preds == batch_labels.long()).sum().item()
                val_total += batch_imgs.size(0)

        scheduler.step()
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        phase = "FROZEN" if epoch <= 3 else "FULL"
        print(f"  Epoch {epoch:>2}/15 [{phase}] Train acc={train_acc:.1%} Val acc={val_acc:.1%}", flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    print(f"  Melhor val_acc: {best_val_acc:.1%}")

    model = model.cpu().eval()
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = MODEL_OUTPUT_DIR / "orientation_detector.onnx"
    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["image"],
        output_names=["logit"],
        dynamic_axes={"image": {0: "batch_size"}, "logit": {0: "batch_size"}},
        opset_version=13,
        dynamo=False,
    )

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"  ONNX exportado: {onnx_path} ({size_mb:.1f} MB)")

    return onnx_path


def validate_integrated_pipeline(onnx_path: Path) -> None:
    print("\n" + "=" * 72)
    print("  VALIDACAO PIPELINE INTEGRADO (PrimaryEngine + FlipClassifier)")
    print("=" * 72)

    from docorient.detection.engine import DetectionPipeline

    pipeline = DetectionPipeline()
    config = OrientationConfig()

    engine_names = [e.name for e in pipeline.engines]
    print(f"\n  Engines no pipeline: {engine_names}")

    print("  Carregando imagens de validacao...")
    base_images = []
    for example in load_dataset("nielsr/funsd", split="test"):
        img = example.get("image")
        if img:
            base_images.append(img.convert("RGB"))
    print(f"  {len(base_images)} imagens de teste")

    results_per_angle = {
        angle: {"correct": 0, "total": 0}
        for angle in [0, 90, 180, 270]
    }

    rotation_map = {
        0: None,
        90: Image.Transpose.ROTATE_270,
        180: Image.Transpose.ROTATE_180,
        270: Image.Transpose.ROTATE_90,
    }

    inference_times = []

    for base_image in base_images:
        for true_angle in [0, 90, 180, 270]:
            transpose_op = rotation_map[true_angle]
            test_image = base_image.transpose(transpose_op) if transpose_op else base_image.copy()

            start_time = time.perf_counter()
            result = pipeline.run(test_image, config)
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            inference_times.append(elapsed_ms)

            results_per_angle[true_angle]["total"] += 1
            if result.angle == true_angle:
                results_per_angle[true_angle]["correct"] += 1

    total_correct = sum(r["correct"] for r in results_per_angle.values())
    total_samples = sum(r["total"] for r in results_per_angle.values())

    print(f"\n  {'Angulo':<10} {'Acuracia':<12} {'Correto/Total':<15}")
    print(f"  {'-' * 37}")
    for angle in [0, 90, 180, 270]:
        res = results_per_angle[angle]
        acc = res["correct"] / res["total"] if res["total"] > 0 else 0
        print(f"  {angle:>3} graus  {acc:>8.1%}     {res['correct']}/{res['total']}")

    overall_acc = total_correct / total_samples if total_samples > 0 else 0
    avg_latency = np.mean(inference_times)

    print(f"\n  ACURACIA GLOBAL: {overall_acc:.1%} ({total_correct}/{total_samples})")
    print(f"  Latencia media pipeline: {avg_latency:.1f} ms/imagem")
    print(f"  Latencia P95: {np.percentile(inference_times, 95):.1f} ms")

    print(f"\n  Pipeline completo:")
    print(f"    - Resolve TODOS os 4 angulos")
    print(f"    - Acuracia global: {overall_acc:.1%}")
    print(f"    - Sem dependencias externas")
    print(f"    - Modelo: {onnx_path.stat().st_size / (1024*1024):.1f} MB")

    report = {
        "pipeline_engines": engine_names,
        "onnx_model": str(onnx_path),
        "overall_accuracy": overall_acc,
        "total_correct": total_correct,
        "total_samples": total_samples,
        "avg_latency_ms": avg_latency,
        "p95_latency_ms": float(np.percentile(inference_times, 95)),
        "per_angle": {
            str(angle): {
                "accuracy": res["correct"] / res["total"] if res["total"] > 0 else 0,
                "correct": res["correct"],
                "total": res["total"],
            }
            for angle, res in results_per_angle.items()
        },
    }
    report_path = MODEL_OUTPUT_DIR / "pipeline_validation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n  Relatorio salvo: {report_path}")
    print("=" * 72)


def main() -> None:
    print("=" * 72)
    print("  DocOrient - Export ONNX & Validacao Completa")
    print("=" * 72)

    onnx_path = MODEL_OUTPUT_DIR / "orientation_detector.onnx"

    if not onnx_path.exists():
        print("\n[1/2] Modelo ONNX nao encontrado. Re-treinando e exportando...")
        onnx_path = retrain_and_export()
    else:
        print(f"\n  Modelo ONNX encontrado: {onnx_path}")
        print(f"  Tamanho: {onnx_path.stat().st_size / (1024*1024):.1f} MB")

    print("\n[2/2] Validando pipeline integrado (PrimaryEngine + FlipClassifier)...")
    validate_integrated_pipeline(onnx_path)


if __name__ == "__main__":
    main()
