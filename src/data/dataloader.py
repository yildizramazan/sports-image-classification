# DataLoader setup and utilities

from collections import Counter
from torch.utils.data import DataLoader, random_split
import torch


def create_dataloaders(dataset, batch_size, val_split, num_workers, seed):
    dataset_size = len(dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size

    generator = torch.Generator().manual_seed(seed)

    train_subset, val_subset = random_split(dataset, [train_size, val_size], generator=generator)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)

    return train_loader, val_loader


def compute_class_weights(dataset):
    class_counts = Counter(dataset.df["label"])
    total = sum(class_counts.values())

    weights = {cls: total / (len(class_counts) * count) for cls, count in class_counts.items()}
    weight_tensor = torch.FloatTensor([weights[cls] for cls in dataset.classes])
    return weight_tensor
