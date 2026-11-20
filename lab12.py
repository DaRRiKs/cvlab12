import csv
import hashlib
import json
import os
import platform
import random
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
import torchvision
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.models import ResNet18_Weights, resnet18

DATA = ROOT / "data"
RESULTS = ROOT / "results"
SEED = 42
CLASSES = ("bird", "cat", "dog")
PER_CLASS = 100
BATCH_SIZE = 16
EPOCHS = (5, 3)
LEARNING_RATES = (1e-3, 1e-4)
MODES = ("frozen backbone", "fine-tuning layer4 + fc")
FIELDS = ("Режим", "Trainable params", "Accuracy", "Macro-F1",
          "Same-class cosine", "Different-class cosine")


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(min(4, os.cpu_count() or 1))


def prepare_data(transform):
    dataset = CIFAR10(root=str(DATA), train=True, download=True, transform=transform)
    original_targets = np.asarray(dataset.targets)
    rng = np.random.default_rng(SEED)
    indices, labels, hashes = [], [], []
    seen = set()
    for label, name in enumerate(CLASSES):
        candidates = np.flatnonzero(original_targets == dataset.classes.index(name))
        chosen = 0
        for index in rng.permutation(candidates):
            digest = hashlib.sha256(dataset.data[index].tobytes()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            indices.append(int(index))
            labels.append(label)
            hashes.append(digest)
            chosen += 1
            if chosen == PER_CLASS:
                break
        if chosen != PER_CLASS:
            raise ValueError(f"Not enough unique images for {name}")
    train_pos, val_pos = train_test_split(
        np.arange(len(indices)), test_size=0.2, random_state=SEED, stratify=labels
    )
    assert set(train_pos).isdisjoint(val_pos)
    assert set(np.asarray(hashes)[train_pos]).isdisjoint(np.asarray(hashes)[val_pos])
    assert set(np.asarray(labels)[train_pos]) == set(range(len(CLASSES)))
    assert set(np.asarray(labels)[val_pos]) == set(range(len(CLASSES)))
    train_positions = set(train_pos.tolist())
    rows = []
    remapped_targets = original_targets.copy()
    for pos, (index, label, digest) in enumerate(zip(indices, labels, hashes)):
        relative = f"images/{CLASSES[label]}/{index:05d}.png"
        path = DATA / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            Image.fromarray(dataset.data[index]).save(path)
        with Image.open(path) as image:
            actual_hash = hashlib.sha256(np.asarray(image.convert("RGB")).tobytes()).hexdigest()
        assert actual_hash == digest, f"Image changed: {relative}"
        remapped_targets[index] = label
        rows.append({"source_index": index, "path": relative, "class": CLASSES[label],
                     "label": label, "split": "train" if pos in train_positions else "val",
                     "pixel_sha256": digest})
    dataset.targets = remapped_targets.tolist()
    with (DATA / "split.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts = {name: {split: sum(r["class"] == name and r["split"] == split for r in rows)
                     for split in ("train", "val")} for name in CLASSES}
    train_set = Subset(dataset, [indices[i] for i in train_pos])
    val_set = Subset(dataset, [indices[i] for i in val_pos])
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                              generator=torch.Generator().manual_seed(SEED))
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    val_paths = np.asarray([rows[i]["path"] for i in val_pos])
    print(f"Data: train={len(train_set)}, val={len(val_set)}, counts={counts}", flush=True)
    return train_loader, val_loader, val_paths, counts


def configure_training(model, fine_tune):
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.fc.parameters():
        parameter.requires_grad = True
    if fine_tune:
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=LEARNING_RATES[int(fine_tune)])
    return optimizer, sum(p.numel() for p in parameters)


def frozen_state(model, fine_tune):
    prefixes = ("fc.", "layer4.") if fine_tune else ("fc.",)
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            if not name.startswith(prefixes)}


def train_model(model, loader, optimizer, epochs, fine_tune, device, history):
    before = frozen_state(model, fine_tune)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.eval()
        model.fc.train()
        if fine_tune:
            model.layer4.train()
        total_loss = 0.0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
        epoch_loss = total_loss / len(loader.dataset)
        history.append({"mode": MODES[int(fine_tune)], "epoch": epoch + 1, "loss": epoch_loss})
        print(f"{MODES[int(fine_tune)]}: epoch {epoch + 1}/{epochs}, loss={epoch_loss:.6f}", flush=True)
    after = frozen_state(model, fine_tune)
    assert all(torch.equal(before[name], after[name]) for name in before)


def extract_embeddings(model, images):
    features = model.maxpool(model.relu(model.bn1(model.conv1(images))))
    features = model.layer1(features)
    features = model.layer2(features)
    features = model.layer3(features)
    features = model.layer4(features)
    features = model.avgpool(features)
    return torch.flatten(features, 1)


def evaluate(model, loader, device):
    model.eval()
    embeddings, targets, predictions = [], [], []
    with torch.inference_mode():
        for images, labels in loader:
            features = extract_embeddings(model, images.to(device))
            logits = model.fc(features)
            embeddings.append(features.cpu().numpy())
            targets.append(labels.numpy())
            predictions.append(logits.argmax(dim=1).cpu().numpy())
    embeddings = np.concatenate(embeddings)
    targets = np.concatenate(targets)
    predictions = np.concatenate(predictions)
    assert embeddings.shape == (len(loader.dataset), 512)
    assert np.isfinite(embeddings).all()
    print(f"Validation embeddings shape: {embeddings.shape}", flush=True)
    return embeddings, targets, predictions


def analyze_embeddings(embeddings, labels):
    original = embeddings.astype(np.float64)
    mean = original.mean(axis=0, keepdims=True)
    centered = original - mean
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    energy = singular_values ** 2
    if energy.sum() <= 0:
        raise ValueError("PCA requires nonzero variance")
    ratios = energy / energy.sum()
    coordinates = centered @ vt[:2].T
    normalized = original / np.maximum(np.linalg.norm(original, axis=1, keepdims=True), 1e-12)
    similarities = normalized @ normalized.T
    i, j = np.triu_indices(len(labels), k=1)
    same = labels[i] == labels[j]
    assert same.any() and (~same).any()
    return {"mean": mean, "components": vt[:2], "singular_values": singular_values,
            "explained_variance_ratio": ratios, "coordinates": coordinates,
            "same_cosine": float(similarities[i[same], j[same]].mean()),
            "different_cosine": float(similarities[i[~same], j[~same]].mean()),
            "same_pairs": int(same.sum()), "different_pairs": int((~same).sum())}


def metric_row(mode, parameters, labels, predictions, analysis):
    values = [mode, str(parameters), f"{accuracy_score(labels, predictions):.6f}",
              f"{f1_score(labels, predictions, average='macro', zero_division=0):.6f}",
              f"{analysis['same_cosine']:.6f}", f"{analysis['different_cosine']:.6f}"]
    return dict(zip(FIELDS, values))


def save_metrics(rows):
    with (RESULTS / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def metrics_table(rows):
    lines = ["| " + " | ".join(FIELDS) + " |", "| " + " | ".join(["---"] * len(FIELDS)) + " |"]
    lines.extend("| " + " | ".join(row[field] for field in FIELDS) + " |" for row in rows)
    return "\n".join(lines)


def plot_pca(analyses, labels):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    colors = ("#0072B2", "#D55E00", "#009E73")
    for axis, analysis, title in zip(axes, analyses, MODES):
        points = analysis["coordinates"]
        ratios = analysis["explained_variance_ratio"]
        for label, (name, color) in enumerate(zip(CLASSES, colors)):
            mask = labels == label
            axis.scatter(points[mask, 0], points[mask, 1], color=color, label=name,
                         s=40, alpha=0.8, edgecolors="white", linewidths=0.5)
        axis.set_title(title)
        axis.set_xlabel(f"PC1 ({ratios[0]:.2%})")
        axis.set_ylabel(f"PC2 ({ratios[1]:.2%})")
        axis.legend(title="Class")
        axis.grid(alpha=0.2)
    fig.savefig(RESULTS / "pca_comparison.png", dpi=180)
    plt.close(fig)


def conclusions(rows, analyses):
    if len(analyses) != 2:
        return ("Обучение с ImageNet-весами не завершено. Accuracy, macro-F1, cosine similarity "
                "и PCA-график отсутствуют. Изменение метрик и разделимости признаков не установлено.")
    first, second = rows
    lines = []
    for field in FIELDS[2:]:
        a, b = float(first[field]), float(second[field])
        lines.append(f"{field}: {a:.6f} → {b:.6f}; изменение {b - a:+.6f}.")
    gaps = [float(row[FIELDS[4]]) - float(row[FIELDS[5]]) for row in rows]
    lines.append(f"Разрыв same-class − different-class: {gaps[0]:.6f} → {gaps[1]:.6f}.")
    if all(float(second[f]) > float(first[f]) for f in FIELDS[2:4]) and gaps[1] > gaps[0]:
        lines.append("На этом validation set метрики классификации и разрыв cosine согласованно "
                     "указывают на улучшение разделимости после fine-tuning.")
    elif all(float(second[f]) < float(first[f]) for f in FIELDS[2:4]) and gaps[1] < gaps[0]:
        lines.append("На этом validation set метрики классификации и разрыв cosine снизились; "
                     "fine-tuning не улучшил разделимость по этим показателям.")
    else:
        lines.append("Показатели изменились неоднозначно либо частично не изменились. "
                     "Утверждать однозначное улучшение пространства признаков нельзя.")
    for mode, analysis in zip(MODES, analyses):
        ratio = analysis["explained_variance_ratio"]
        lines.append(f"PCA {mode}: PC1 сохраняет {ratio[0]:.2%}, PC2 — {ratio[1]:.2%}; "
                     f"вместе {ratio[:2].sum():.2%} энергии центрированных признаков.")
    lines.append("На графиках показаны проекции одних и тех же изображений с неизменными цветами классов. "
                 "Две компоненты сохраняют только указанную долю информации. PCA обучена отдельно "
                 "в каждом режиме: оси могут поворачиваться и менять знак. Положение облаков между "
                 "панелями само по себе не доказывает улучшение. Вывод ограничен одной небольшой "
                 "выборкой и одним seed; статистическая значимость не проверялась.")
    return "\n\n".join(lines)


def write_report(run, rows, analyses):
    counts = run.get("counts", {})
    count_table = "\n".join(f"| {name} | {counts.get(name, {}).get('train', '')} | "
                             f"{counts.get(name, {}).get('val', '')} |" for name in CLASSES)
    figure = ("![PCA двух режимов](results/pca_comparison.png)" if run["status"] == "complete"
              else "График `results/pca_comparison.png` не создан в этом запуске: сравнение режимов не завершено.")
    status = ("Полный эксперимент выполнен." if run["status"] == "complete"
              else "Полный эксперимент НЕ выполнен. Пустые ячейки означают отсутствие измерений.")
    details = run.get("error", "Ошибок выполнения нет.")
    report = f"""# Лабораторная работа №12. Transfer Learning, PCA и SVD признаков

## Цель и статус

Исследовать 512-мерные признаки pretrained ResNet18 и сравнить обучение только classifier с fine-tuning последнего stage по accuracy, macro-F1 и геометрии embeddings.

{status} Последний этап: `{run['stage']}`. `{details}`

## Теоретические задания

Для x = (1, 2, 2), y = (2, 1, 2):

- xᵀy = 1·2 + 2·1 + 2·2 = 8.
- ‖x‖ = √(1² + 2² + 2²) = √9 = 3; ‖y‖ = √(2² + 1² + 2²) = √9 = 3.
- cos(x, y) = 8/(3·3) = 8/9 ≈ 0.8889.

Для точек (1, 1), (2, 2), (3, 3), N = 3:

```text
μ = ((1+2+3)/3, (1+2+3)/3) = (2, 2)
X_c = [[-1, -1], [0, 0], [1, 1]]
X_cᵀX_c = [[2, 2], [2, 2]]
C = (1/3)X_cᵀX_c = [[2/3, 2/3], [2/3, 2/3]]
det(C − λI) = (2/3 − λ)² − (2/3)² = λ(λ − 4/3) = 0
λ₁ = 4/3; λ₂ = 0
v₁ = (1, 1)/√2; v₂ = (1, −1)/√2
X_c v₁ = [(-1−1)/√2, 0, (1+1)/√2] = [−√2, 0, √2]
```

PC1 направлена вдоль y = x, PC2 перпендикулярна ей. Знак каждого собственного вектора можно изменить. Для этой центрированной матрицы σ₁ = √(Nλ₁) = 2, σ₂ = 0.

Отдельно для заданных сингулярных значений (10, 3, 1):

- Полная энергия ‖X‖²_F = 10² + 3² + 1² = 110.
- Rank-1 сохраняет 10²/110 = 100/110 ≈ 90.91%; остаточная энергия 10.
- Rank-2 сохраняет (10² + 3²)/110 = 109/110 ≈ 99.09%; остаточная энергия 1.

## Данные

Использован [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), авторы Alex Krizhevsky, Vinod Nair и Geoffrey Hinton. [Официальный архив Python](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz), MD5 `c58f30108f718f92721af3b95e74349a`. Это реальные цветные изображения 32×32 с исходными метками. Источник доступен без регистрации. На странице не указана отдельная лицензия на изображения; авторы просят цитировать Alex Krizhevsky, *Learning Multiple Layers of Features from Tiny Images*, 2009. Изображения используются в учебной работе.

Из официальной обучающей части (50 000 изображений) выбраны bird, cat, dog. Для каждого класса исходные индексы перемешиваются `np.random.default_rng(42)`, берутся первые 100 уникальных по SHA-256 RGB-пикселей изображений. Точные дубликаты исключаются до разделения; похожие фотографии специально не группируются. Официальная test-часть не используется. Затем применяется `train_test_split(test_size=0.2, stratify=labels, random_state=42)`.

| Класс | Train | Val |
| --- | --- | --- |
{count_table}

Фактические индексы, классы, пути, split и SHA-256 записаны в `data/split.csv`; выбранные PNG находятся в `data/images/`. Программа проверяет отсутствие пересечений по индексам и содержимому, а также наличие всех классов в обеих частях. Один validation set используется в обоих режимах, порядок в его DataLoader фиксирован. Исходные batch-файлы CIFAR-10 хранятся в `data/cifar-10-batches-py/`; torchvision проверяет их MD5, при отсутствии загружает архив, при повторном запуске использует сохранённые данные. Для этой поставки найдена и скопирована ранее скачанная локальная копия CIFAR-10, поскольку сеть среды блокирует загрузки; происхождение записано в `data/source.json`. Первоначально рассматривался TensorFlow Flowers, но его архив скачать не удалось.

## Параметры и обучение

- Seed: {SEED}; устройство проверки: `{run['device']}`; PyTorch CPU threads: {run['threads']}; DataLoader workers: 0.
- Python {run['python']}; torch {run['versions']['torch']}; torchvision {run['versions']['torchvision']}; NumPy {run['versions']['numpy']}; scikit-learn {run['versions']['sklearn']}; matplotlib {run['versions']['matplotlib']}.
- Запланировано {EPOCHS[0]} эпох frozen и {EPOCHS[1]} эпох fine-tuning; завершено эпох: {run['epochs_completed']}. Batch size: {BATCH_SIZE}. Loss: CrossEntropyLoss.
- Optimizer: Adam, betas=(0.9, 0.999), eps=1e-8, weight_decay=0; LR frozen={LEARNING_RATES[0]}, fine-tuning={LEARNING_RATES[1]}. Оптимизатор создаётся заново для режима B и получает только параметры с `requires_grad=True`.
- Веса: `ResNet18_Weights.IMAGENET1K_V1`. [Предобработка torchvision](https://docs.pytorch.org/vision/0.24/models/generated/torchvision.models.resnet18.html): `weights.transforms()` для train и val — bilinear resize короткой стороны до 256, center crop 224×224, перевод в [0,1], нормализация mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225). Случайные аугментации не применяются. Увеличение CIFAR-10 до 224×224 не восстанавливает отсутствующие детали.

В A обучается новый `Linear(512, 3)`. Перед каждой эпохой вызывается `model.eval()`, затем `model.fc.train()`: BatchNorm backbone не меняет статистики. В B используется та же модель после A; размораживаются все параметры `layer4` и `fc`, вызывается `model.layer4.train()`. Остальная сеть остаётся в eval. Проверка `frozen_state` сравнивает замороженные веса и буферы, включая BatchNorm, до и после обучения каждого режима. Для оценки и извлечения признаков используются `eval()` и `torch.inference_mode()`.

После каждого режима до перехода к следующему сохраняются checkpoint, исходные embeddings, метки, предсказания, пути, PCA/SVD и строка метрик. Используется последняя запланированная эпоха без подбора по validation. CSV, таблица отчёта и консоль формируются из одних строк с шестью знаками после запятой. Accuracy и macro-F1 приведены в долях от 0 до 1.

## PCA, SVD и cosine

```text
X ∈ R^(N×512), μ = (1/N)Σ_i X_i
X_c = X − μ
C = (1/N) · X_cᵀX_c
X_c = UΣVᵀ
C = V(ΣᵀΣ/N)Vᵀ
Z = X_c V[:, :2]
explained_energy(k) = Σ_(i≤k) σ_i² / Σ_i σ_i²
```

Столбцы V — собственные векторы C и правые сингулярные векторы X_c; λ_i = σ_i²/N. В коде используется сокращённая SVD; недостающие направления полного пространства соответствуют нулевым собственным значениям. Первые два столбца V задают направления PCA, Z — координаты точек. Деление на N−1 изменило бы собственные значения, но не направления и доли объяснённой дисперсии. PCA строится отдельно для каждого режима и только для визуализации validation; классификатор от этой PCA не зависит.

Cosine считается по исходным 512-мерным embeddings: E_i = X_i/max(‖X_i‖, 1e-12), S = EEᵀ. Используются только пары i < j. Для 60 validation-изображений по 20 на класс это 3·20·19/2 = 570 пар одного класса и C(60,2)−570 = 1200 пар разных классов. Центрирование и PCA к cosine не применяются.

{figure}

## Результаты и вывод

{metrics_table(rows)}

{conclusions(rows, analyses)}

## Контрольные вопросы

1. Для ненулевых x, y и положительных a, b: cos(ax, by) = ab·xᵀy/(|a|‖x‖·|b|‖y‖) = cos(x, y). Положительный масштаб сокращается. Если только один множитель отрицателен, знак cosine меняется; если оба отрицательны, сохраняется. Для нулевого вектора математический cosine не определён, а epsilon в программе лишь защищает вычисления.
2. PCA выбирает собственные векторы ковариационной матрицы по убыванию λ. Из SVD центрированных данных X_c = UΣVᵀ следует C = V(ΣᵀΣ/N)Vᵀ: направления PCA — правые сингулярные векторы, а доли дисперсии пропорциональны σ².
3. В `extract_embeddings()` после `model.avgpool(features)` выполняется `torch.flatten(features, 1)`; результат имеет 512 признаков. В `evaluate()` только затем вызывается `model.fc(features)`. Softmax даёт лишь три зависимые вероятности с суммой 1. Линейный classifier 512→3 и softmax теряют информацию о направлениях и расстояниях исходного пространства; по вероятностям нельзя восстановить 512-мерную геометрию.

## Запуск

```powershell
python -m pip install -r requirements.txt
python lab12.py
```

Нужен Python с доступными колёсами указанных пакетов; библиотеки проверены на Python 3.14.5. CUDA выбирается автоматически, при её отсутствии используется CPU. Первый запуск загружает отсутствующие данные и ImageNet-веса (около 45 МБ); веса кешируются в `data/weights/`. Все пути определяются относительно `lab12.py`. При успешном запуске отчёт и таблица обновляются автоматически; сведения о выполнении записываются в `results/run.json`.

Для первого запуска требуется доступ к официальному источнику ImageNet-весов `https://download.pytorch.org/models/resnet18-f37072fd.pth`. После загрузки веса сохраняются в `data/weights/checkpoints/`. Фактический статус текущего запуска, число завершённых эпох и возможная ошибка приведены в `results/run.json`. Случайные веса не используются вместо предобученных весов для получения экспериментальных результатов.
"""
    (ROOT / "report.md").write_text(report, encoding="utf-8")


def main():
    DATA.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    socket.setdefaulttimeout(30)
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.hub.set_dir(str(DATA / "weights"))
    run = {"status": "incomplete", "stage": "data preparation", "device": str(device),
           "threads": torch.get_num_threads(), "python": platform.python_version(),
           "versions": {"torch": torch.__version__, "torchvision": torchvision.__version__,
                        "numpy": np.__version__, "sklearn": sklearn.__version__,
                        "matplotlib": matplotlib.__version__},
           "seed": SEED, "epochs_planned": EPOCHS, "epochs_completed": [0, 0],
           "batch_size": BATCH_SIZE, "learning_rates": LEARNING_RATES,
           "optimizer": "Adam", "history": [], "executable": sys.executable}
    rows = [dict(zip(FIELDS, [mode, "", "", "", "", ""])) for mode in MODES]
    analyses = []
    started = time.perf_counter()
    print(f"Device: {device}; seed={SEED}; threads={torch.get_num_threads()}", flush=True)
    try:
        weights = ResNet18_Weights.IMAGENET1K_V1
        train_loader, val_loader, paths, counts = prepare_data(weights.transforms())
        run["counts"] = counts
        run["stage"] = "loading ImageNet weights"
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
        model.to(device)
        previous_labels = None
        for index, fine_tune in enumerate((False, True)):
            run["stage"] = MODES[index]
            optimizer, parameters = configure_training(model, fine_tune)
            print(f"Trainable parameters: {parameters}", flush=True)
            train_model(model, train_loader, optimizer, EPOCHS[index], fine_tune, device, run["history"])
            run["epochs_completed"][index] = EPOCHS[index]
            embeddings, labels, predictions = evaluate(model, val_loader, device)
            if previous_labels is not None:
                assert np.array_equal(labels, previous_labels)
            previous_labels = labels.copy()
            analysis = analyze_embeddings(embeddings, labels)
            analyses.append(analysis)
            rows[index] = metric_row(MODES[index], parameters, labels, predictions, analysis)
            stem = "fine_tuned" if fine_tune else "frozen"
            torch.save(model.state_dict(), RESULTS / f"{stem}_model.pt")
            np.savez_compressed(RESULTS / f"{stem}_embeddings.npz", embeddings=embeddings,
                                labels=labels, predictions=predictions, paths=paths, **analysis)
            save_metrics(rows)
            print(metrics_table(rows), flush=True)
        run["stage"] = "PCA comparison"
        plot_pca(analyses, previous_labels)
        run["status"] = "complete"
        run["stage"] = "finished"
    except Exception as error:
        run["error"] = f"{type(error).__name__}: {error}"
        print(f"Run not completed at {run['stage']}: {run['error']}", flush=True)
    finally:
        run["epochs_completed"] = [sum(item["mode"] == mode for item in run["history"]) for mode in MODES]
        run["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        save_metrics(rows)
        (RESULTS / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
        write_report(run, rows, analyses)
        print(metrics_table(rows), flush=True)
        print(f"Status: {run['status']}; elapsed={run['elapsed_seconds']:.3f}s", flush=True)
    return 0 if run["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
