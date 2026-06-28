# Image transformations and augmentations pipelines
from torchvision.transforms import v2
import torch


MEAN=[0.4177, 0.4200, 0.4048]
STD=[0.2876, 0.2782, 0.2822]

def get_train_transforms(image_size=224):

    train_transforms=v2.Compose([
        v2.Resize(image_size+32),
        v2.RandomCrop(image_size),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(degrees=15),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.01),
        v2.ToImage(),  
    v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.4177, 0.4200, 0.4048],
        std=[0.2876, 0.2782, 0.2822]),
    ])
    return train_transforms


def get_test_transforms(image_size=224):

    test_transforms=v2.Compose([
        v2.Resize((image_size,image_size)),
        v2.ToImage(),  
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=MEAN,
        std=STD),
    ])
    return test_transforms



def denormalize(tensor: torch.Tensor) -> torch.Tensor:

    device = tensor.device
    
    # Convert lists to tensors matching the input tensor's dtype and device
    mean = torch.tensor(MEAN, dtype=tensor.dtype, device=device)
    std = torch.tensor(STD, dtype=tensor.dtype, device=device)
    
    # Reshape mean and std for correct broadcasting depending on tensor dimensions
    if tensor.ndim == 3:  # (C, H, W)
        mean = mean.view(3, 1, 1)
        std = std.view(3, 1, 1)
    elif tensor.ndim == 4:  # (B, C, H, W)
        mean = mean.view(1, 3, 1, 1)
        std = std.view(1, 3, 1, 1)
    else:
        raise ValueError(f"Unsupported tensor dimension: {tensor.shape}. Only 3D or 4D tensors are supported.")
    
    # Denormalization formula: tensor * std + mean
    denorm_tensor = tensor * std + mean
    
    # Clamp values to strictly stay within the [0.0, 1.0] range
    return torch.clamp(denorm_tensor, 0.0, 1.0)
