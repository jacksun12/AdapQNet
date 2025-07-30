import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import logging


def create_train_val_test_loaders(batch_size=64, num_workers=8, input_size=112):
    """Create training, validation, and test data loaders with custom resolution support
    
    Args:
        batch_size: Batch size (optimized for single GPU with 96)
        num_workers: Number of worker processes for data loading
        input_size: Input image size, default 112x112
        
    Returns:
        tuple: (train_loader, val_loader, test_loader) - Data loaders for training, validation, and test sets
    """
    logger = logging.getLogger(__name__)
    
    # Training transforms: includes data augmentation
    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),  # Resize to target size
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),  # Add random rotation
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color augmentation
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Validation and test transforms: only resize and normalization
    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),  # Resize to target size
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Training set: use original train=True dataset (50,000 images)
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform)
    
    # Split validation and test sets from test dataset (train=False, 10,000 images)
    test_dataset_full = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=val_transform)
    
    # Take 6000 images as validation set, 4000 as test set
    val_size = 6000
    test_size = 4000
    
    val_dataset, test_dataset = random_split(
        test_dataset_full, [val_size, test_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True)
    
    logger.info(f"Dataset split ({input_size}x{input_size} resolution):")
    logger.info(f"  Training set: {len(train_dataset)} images (weight training)")
    logger.info(f"  Validation set: {len(val_dataset)} images (alpha training)")
    logger.info(f"  Test set: {len(test_dataset)} images (performance evaluation)")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Input size: {input_size}x{input_size}")
    
    return train_loader, val_loader, test_loader 
