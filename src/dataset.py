import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

IMG_SIZE = 224
BATCH_SIZE = 32

# Normalisation standard ImageNet (requise pour les modèles pré-entraînés)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]


def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ])


def get_dataloaders(data_dir="data", batch_size=BATCH_SIZE):
    train_dataset = datasets.ImageFolder(f"{data_dir}/train", transform=get_transforms(train=True))
    val_dataset = datasets.ImageFolder(f"{data_dir}/val", transform=get_transforms(train=False))
    test_dataset = datasets.ImageFolder(f"{data_dir}/test", transform=get_transforms(train=False))

    # train_dataset.class_to_idx donne par exemple {'cat': 0, 'dog': 1}
    class_names = train_dataset.classes

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader, class_names