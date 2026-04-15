"""
Data loading utilities for Split Federated Learning.

Supports CIFAR-10/100 and MedMNIST with Dirichlet-based non-IID partition.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
from typing import List, Tuple, Optional, Union


def dirichlet_partition(
    dataset: Dataset,
    n_clients: int,
    alpha: float,
    seed: int = 42,
    min_samples_per_client: int = 1,
) -> List[List[int]]:
    """
    Partition dataset using Dirichlet distribution for Non-IID simulation.

    Each client receives a proportion sampled from Dir(alpha).
    Small alpha -> highly heterogeneous (clients have few classes).
    Large alpha -> nearly IID.

    Args:
        dataset: PyTorch Dataset with .targets attribute.
        n_clients: Number of clients.
        alpha: Dirichlet concentration parameter (lower = more skewed).
        seed: Random seed for reproducibility.
        min_samples_per_client: Minimum samples per client (safety).

    Returns:
        List of client_index_lists, each containing dataset indices.
    """
    np.random.seed(seed)
    labels = np.array(dataset.targets)
    n_classes = len(np.unique(labels))
    client_indices: List[List[int]] = [[] for _ in range(n_clients)]

    for c in range(n_classes):
        class_indices = np.where(labels == c)[0]
        np.random.shuffle(class_indices)

        # Dirichlet draw per class
        proportions = np.random.dirichlet([alpha] * n_clients)
        proportions = proportions / proportions.sum()  # normalize

        # Convert to sample counts
        counts = (proportions * len(class_indices)).astype(int)

        # Ensure minimum samples
        deficit = min_samples_per_client - counts
        deficit = np.maximum(deficit, 0)
        counts += deficit

        # Adjust last client to account for rounding
        counts[-1] = len(class_indices) - counts[:-1].sum()
        counts = np.maximum(counts, 0)  # safety clip

        idx = 0
        for client_id, count in enumerate(counts):
            end = min(idx + count, len(class_indices))
            if idx < end:
                client_indices[client_id].extend(class_indices[idx:end].tolist())
            idx = end

    # Shuffle each client's indices
    for idx_list in client_indices:
        np.random.shuffle(idx_list)

    return client_indices


def get_cifar10_transforms(train: bool = True, img_size: int = 32) -> transforms.Compose:
    """
    Standard CIFAR-10 transforms.

    Args:
        train: Whether to use training augmentation.
        img_size: Image resize size.

    Returns:
        transforms.Compose pipeline.
    """
    if train:
        return transforms.Compose([
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])


def get_cifar100_transforms(train: bool = True, img_size: int = 32) -> transforms.Compose:
    """CIFAR-100 transforms."""
    if train:
        return transforms.Compose([
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])


def load_cifar10(data_dir: str = "./data", download: bool = True) -> Dataset:
    return datasets.CIFAR10(data_dir, train=True, download=download, transform=get_cifar10_transforms(True))


def load_cifar100(data_dir: str = "./data", download: bool = True) -> Dataset:
    return datasets.CIFAR100(data_dir, train=True, download=download, transform=get_cifar100_transforms(True))


def load_cifar10_test(data_dir: str = "./data", download: bool = True) -> Dataset:
    return datasets.CIFAR10(data_dir, train=False, download=download, transform=get_cifar10_transforms(False))


def load_cifar100_test(data_dir: str = "./data", download: bool = True) -> Dataset:
    return datasets.CIFAR100(data_dir, train=False, download=download, transform=get_cifar100_transforms(False))


def get_dataloaders(
    dataset_name: str,
    n_clients: int,
    alpha: float,
    batch_size: int,
    data_dir: str = "./data",
    download: bool = True,
    seed: int = 42,
    shuffle: bool = True,
    num_workers: int = 0,
) -> Tuple[List[DataLoader], Dataset, Dataset]:
    """
    Create federated dataloaders with Dirichlet non-IID partition.

    Args:
        dataset_name: 'cifar10' or 'cifar100'.
        n_clients: Number of clients.
        alpha: Dirichlet alpha (0.1 = highly non-IID, 10.0 = nearly IID).
        batch_size: Batch size per client.
        data_dir: Data root directory.
        download: Whether to download if not present.
        seed: Random seed.
        shuffle: Whether to shuffle client datasets.
        num_workers: DataLoader num_workers.

    Returns:
        (client_loaders, train_dataset, test_dataset) tuple.
    """
    if dataset_name == "cifar10":
        train_dataset = load_cifar10(data_dir, download)
        test_dataset = load_cifar10_test(data_dir, download)
    elif dataset_name == "cifar100":
        train_dataset = load_cifar100(data_dir, download)
        test_dataset = load_cifar100_test(data_dir, download)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    partitions = dirichlet_partition(train_dataset, n_clients, alpha, seed=seed)

    client_loaders = []
    for indices in partitions:
        subset = Subset(train_dataset, indices)
        client_loaders.append(
            DataLoader(
                subset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                drop_last=False,
            )
        )

    return client_loaders, train_dataset, test_dataset


def create_test_loader(
    dataset_name: str,
    data_dir: str = "./data",
    batch_size: int = 128,
    download: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a centralized test loader for evaluation.

    Args:
        dataset_name: 'cifar10' or 'cifar100'.
        data_dir: Data root directory.
        batch_size: Batch size.
        download: Whether to download.
        num_workers: DataLoader num_workers.

    Returns:
        Test DataLoader.
    """
    if dataset_name == "cifar10":
        test_dataset = load_cifar10_test(data_dir, download)
    elif dataset_name == "cifar100":
        test_dataset = load_cifar100_test(data_dir, download)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )


def compute_data_stats(client_loaders: List[DataLoader]) -> dict:
    """
    Compute per-client data statistics.

    Args:
        client_loaders: List of client DataLoaders.

    Returns:
        Dict with per-client sample counts and class distributions.
    """
    stats = []
    for i, loader in enumerate(client_loaders):
        n_samples = len(loader.dataset)
        # Sample one batch to estimate class distribution
        if len(loader) > 0:
            try:
                _, labels = next(iter(loader))
                if isinstance(labels, list):
                    labels = torch.stack(labels)
                class_counts = torch.bincount(labels, minlength=10)
            except Exception:
                class_counts = None
        else:
            class_counts = None

        stats.append({
            "client_id": i,
            "n_samples": n_samples,
            "n_batches": len(loader),
            "class_counts": class_counts,
        })
    return {"clients": stats, "total_clients": len(client_loaders)}