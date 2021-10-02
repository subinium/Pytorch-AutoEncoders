import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision.utils import save_image, make_grid
from model import VanillaVAE
from dataset import *
import argparse



def train(args, device):
    model = VanillaVAE(args).to(device)
    optimizer = Adam(model.parameters(), lr=args['lr'])
    train_loader, valid_loader = mnist_dataloader('./dataset', 32)
    print('Number of Parameters: ', sum(param.numel() for param in model.parameters()))
    model.train()
    iteration = 0
    for epoch in range(100):
        for idx, (image, label) in enumerate(train_loader):
            iteration += 1
            image = image.to(device)
            image = image.view(32, -1)
            x_hat, loss = model(image)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iteration % 100 == 0:
                print(iteration, loss.item())
            
            

if __name__ == '__main__':
    args = {"input_dim":784, "hidden_dim":100, "latent_dim":50, "output_dim":784, "lr":0.0001}
    print(torch.cuda.is_available())
    device = 'cuda'
    train(args, device)