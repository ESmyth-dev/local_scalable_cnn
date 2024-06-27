from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms


def prepare_test_data(batch_size):
    """Returns transformed cifar-10 data for testing."""
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])

    transformed_test_data = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=test_transform,
    )

    return DataLoader(transformed_test_data, batch_size=batch_size)


def prepare_train_data(batch_size):
    """Returns transformed cifar-10 data for training."""
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])

    transformed_training_data = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=train_transform,
    )

    return DataLoader(transformed_training_data, batch_size=batch_size)