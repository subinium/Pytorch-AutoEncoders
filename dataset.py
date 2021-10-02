from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def mnist_dataloader(dataset_path, batch_size):
    transform = transforms.Compose([
            transforms.ToTensor(),
            
    ])

    train_dataset = MNIST(dataset_path, transform=transform, train=True, download=True)
    test_dataset  = MNIST(dataset_path, transform=transform, train=False, download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    test_loader  = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
    return train_loader, test_loader