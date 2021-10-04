# utils
import argparse
import json
# torch modules
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image, make_grid
# custom modules
from model import *
from dataset import *
from logger import *


def train(args, device):
    model = VanillaVAE(args).to(device)
    optimizer = Adam(model.parameters(), lr=args['lr'])
    logger = prepare_logger()
    
    train_loader, valid_loader = mnist_dataloader('./dataset', args['batch_size'])
    print('Number of Parameters: ', sum(param.numel() for param in model.parameters()))
    
    
    model.train()
    iteration = 0
    for epoch in range(args['epochs']):
        for idx, (x, label) in enumerate(train_loader):
            iteration += 1
            x = x.view(32, -1).to(device)
            x_hat, loss = model(x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if iteration % 100 == 0:
                logger.add_scalar('Loss/train', loss, iteration)
                logger.add_images('Image/train', torch.stack([x.cpu(), x_hat.cpu()], dim=0).view(-1, 1, 28, 28), iteration)
            
                
            

if __name__ == '__main__':
    with open('config.json', 'r') as f:
        args = json.load(f)
    print('CUDA AVAILABLE :', torch.cuda.is_available())
    device = 'cuda'
    train(args, device)