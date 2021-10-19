from torchvision.datasets import MNIST, ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomImageFolder, self).__init__(root, transform)

    def __getitem__(self, index):
        path = self.imgs[index][0]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img

def mnist_dataloader(dataset_path, batch_size):
    transform = transforms.Compose([
            transforms.ToTensor(),
            
    ])

    train_dataset = MNIST(dataset_path, transform=transform, train=True, download=True)
    test_dataset  = MNIST(dataset_path, transform=transform, train=False, download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
    test_loader  = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
    return train_loader, test_loader

def celebA_dataloader(dataset_path, batch_size): 
    image_size = 64
    transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            
    ])
    train_data = CustomImageFolder(root='./dataset/celeba/', transform=transform)
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=1,
                              pin_memory=True,
                              drop_last=True)


    
#     test_loader  = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
    return train_loader#, test_loader

if __name__ == '__main__':
    celebA_dataloader('./', 32)