# DataLoader setup and utilities

from numpy import std
import torch
from torchvision.transforms import v2

def get_train_transforms(image_size=224):
    train_transforms=v2.Compose([
        v2.Resize(image_size+32),
        v2.RandomCrop(image_size),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(degrees=15),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.01),
        v2.ToTensor(),
        v2.Normalize(mean=[0.41773849725723267, 0.4199974238872528, 0.40476781129837036],
        std=[0.28759416937828064, 0.27821722626686096, 0.28223705291748047])
    ])
    return train_transforms

