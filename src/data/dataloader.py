# DataLoader setup and utilities
from torchvision.transforms import v2

def get_train_transforms(image_size=224):
    train_transforms=v2.Compose([
        v2.Resize(image_size+32),
        v2.RandomCrop(image_size),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(degrees=15),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.01),
        v2.ToTensor(),
        v2.Normalize(mean=[0.4177, 0.4200, 0.4048],
        std=[0.2876, 0.2782, 0.2822])
    ])
    return train_transforms

