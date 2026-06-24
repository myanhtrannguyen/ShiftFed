"""Shared utilities for domain-decomposed Federated Learning experiments."""

from __future__ import annotations

import csv
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset


DOMAIN_BY_RANK = {1: "mnist", 2: "svhn", 3: "usps"}
ALL_DOMAINS = ("mnist", "svhn", "usps")


@dataclass
class ExperimentConfig:
    rounds: int = 10
    local_steps: int = 100
    batch_size: int = 64
    lr: float = 0.01
    model: str = "lenet5"
    data_dir: str = "./data"
    log_dir: str = "./outputs/fl_mpi"
    seed: int = 42
    download: bool = False
    synthetic: bool = False
    train_subset: int = 0
    test_subset: int = 2000
    async_mode: bool = False
    load_balance: bool = False
    device: str = "cpu"


class LeNet5(nn.Module):
    """LeNet-5 style CNN for 1x28x28 digit classification."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SyntheticDigits(Dataset):
    """Deterministic lightweight digit-like dataset used for smoke tests."""

    def __init__(self, domain: str, train: bool, size: int = 2048, seed: int = 42) -> None:
        self.domain = domain
        self.train = train
        self.size = size
        domain_offset = {"mnist": 0, "svhn": 1000, "usps": 2000}[domain]
        generator = torch.Generator().manual_seed(seed + domain_offset + (0 if train else 777))
        self.labels = torch.randint(0, 10, (size,), generator=generator)
        base = torch.rand(size, 1, 28, 28, generator=generator) * 0.25
        rows = torch.arange(28).view(1, 1, 28, 1)
        cols = torch.arange(28).view(1, 1, 1, 28)
        label_pattern = ((rows + cols + self.labels.view(-1, 1, 1, 1)) % 10 == 0).float()
        domain_bias = {"mnist": 0.15, "svhn": 0.35, "usps": 0.05}[domain]
        self.images = torch.clamp(base + label_pattern * 0.7 + domain_bias, 0.0, 1.0)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        return self.images[index], int(self.labels[index])


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(model_name: str) -> nn.Module:
    name = model_name.lower()
    if name in {"lenet5", "lenet", "cnn"}:
        return LeNet5()
    if name == "mlp":
        return MLP()
    raise ValueError(f"Unsupported model '{model_name}'. Choose 'lenet5' or 'mlp'.")


def _torchvision_dataset(domain: str, train: bool, data_dir: str, download: bool) -> Dataset:
    from torchvision import datasets, transforms

    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
        ]
    )
    root = Path(data_dir)
    if domain == "mnist":
        return datasets.MNIST(root=str(root), train=train, download=download, transform=transform)
    if domain == "svhn":
        split = "train" if train else "test"
        target_transform = lambda label: 0 if label == 10 else label
        return datasets.SVHN(
            root=str(root),
            split=split,
            download=download,
            transform=transform,
            target_transform=target_transform,
        )
    if domain == "usps":
        return datasets.USPS(root=str(root), train=train, download=download, transform=transform)
    raise ValueError(f"Unknown domain '{domain}'.")


def get_digit_dataset(config: ExperimentConfig, domain: str, train: bool) -> Dataset:
    if config.synthetic:
        size = 2048 if train else 512
        return SyntheticDigits(domain=domain, train=train, size=size, seed=config.seed)
    try:
        return _torchvision_dataset(domain, train, config.data_dir, config.download)
    except Exception as exc:
        print(
            f"[WARN] Falling back to synthetic {domain.upper()} data because dataset loading failed: {exc}"
        )
        size = 2048 if train else 512
        return SyntheticDigits(domain=domain, train=train, size=size, seed=config.seed)


def maybe_subset(dataset: Dataset, max_items: int) -> Dataset:
    if max_items and max_items > 0 and len(dataset) > max_items:
        return Subset(dataset, range(max_items))
    return dataset


def make_loader(
    config: ExperimentConfig,
    domain: str,
    train: bool,
    shuffle: bool | None = None,
) -> DataLoader:
    dataset = get_digit_dataset(config, domain, train)
    subset_size = config.train_subset if train else config.test_subset
    dataset = maybe_subset(dataset, subset_size)
    if shuffle is None:
        shuffle = train
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=False,
    )


def state_dict_to_cpu(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in state_dict.items()}


def fedavg(
    weighted_states: Sequence[Tuple[Mapping[str, torch.Tensor], float]]
) -> Dict[str, torch.Tensor]:
    if not weighted_states:
        raise ValueError("Cannot aggregate an empty list of client states.")
    total_weight = float(sum(weight for _, weight in weighted_states))
    if total_weight <= 0:
        total_weight = float(len(weighted_states))
        weighted_states = [(state, 1.0) for state, _ in weighted_states]

    average: MutableMapping[str, torch.Tensor] = {}
    for state, weight in weighted_states:
        scale = float(weight) / total_weight
        for name, tensor in state.items():
            value = tensor.detach().cpu().float() * scale
            average[name] = value if name not in average else average[name] + value
    return dict(average)


def train_fixed_steps(
    model: nn.Module,
    loader: DataLoader,
    config: ExperimentConfig,
    device: torch.device,
    local_steps: int | None = None,
) -> Dict[str, float]:
    model.train()
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    iterator = iter(loader)
    total_loss = 0.0
    correct = 0
    total = 0
    steps_done = 0
    start = time.perf_counter()

    steps_to_run = local_steps if local_steps is not None else config.local_steps
    for _ in range(steps_to_run):
        try:
            images, labels = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            images, labels = next(iterator)

        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * labels.size(0)
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += int(labels.size(0))
        steps_done += 1

    elapsed = time.perf_counter() - start
    return {
        "loss": total_loss / max(total, 1),
        "acc": correct / max(total, 1),
        "samples": float(total),
        "steps": float(steps_done),
        "compute_time": elapsed,
    }


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loaders: Mapping[str, DataLoader],
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    model.to(device)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    metrics: Dict[str, float] = {}
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for domain, loader in loaders.items():
        domain_loss = 0.0
        domain_correct = 0
        domain_samples = 0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            domain_loss += float(criterion(logits, labels).item())
            domain_correct += int((logits.argmax(dim=1) == labels).sum().item())
            domain_samples += int(labels.size(0))
        metrics[f"{domain}_loss"] = domain_loss / max(domain_samples, 1)
        metrics[f"{domain}_acc"] = domain_correct / max(domain_samples, 1)
        total_loss += domain_loss
        total_correct += domain_correct
        total_samples += domain_samples

    metrics["global_loss"] = total_loss / max(total_samples, 1)
    metrics["global_acc"] = total_correct / max(total_samples, 1)
    return metrics


def append_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = sorted({key for row in rows for key in row.keys()})
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
