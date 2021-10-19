# utils
import argparse
import json
# torch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image, make_grid
# custom modules
from models.VanillaVAE import VanillaVAE
from models.CVAE import CVAE
from models.BetaVAE  import BetaVAE
from dataset import *
from logger import *


def train(args, device):
    model_name = args['model']
    print(model_name)
    if model_name == 'VAE':
        model = VanillaVAE(args).to(device)
    elif model_name == 'CVAE':
        model = CVAE(args).to(device)
    elif model_name == 'BetaVAE':
        model = BetaVAE(args).to(device)
    optimizer = Adam(model.parameters(), lr=args['lr'])
    logger = prepare_logger()
    
    if model_name != 'BetaVAE':
        train_loader, valid_loader = mnist_dataloader('./dataset', args['batch_size'])
    else:
        train_loader = celebA_dataloader('./', 32)
    print('Number of Parameters: ', sum(param.numel() for param in model.parameters()))
    
    
    model.train()
    iteration = 0
    for epoch in range(args['epochs']):
        for idx, x in enumerate(train_loader):
            iteration += 1            
            if model_name == 'CVAE':
                x, label = x
            x = x.to(device)

            if model_name == 'VAE':
                x_hat, loss = model(x)
            if model_name == 'CVAE':
                label_ohe = F.one_hot(label, num_classes=10).to(device)
                x_hat, loss = model(x, label_ohe)
            if model_name == 'BetaVAE':
                x_hat, loss = model(x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if iteration % 100 == 0:
                logger.add_scalar('Loss/train', loss, iteration)
                logger.add_images('Image/train', torch.stack([x.cpu(), x_hat.cpu()], dim=0).view(-1, *(x.shape[1:])), iteration)
            
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/VanillaVAE.json')

    args = parser.parse_args()


    with open(args.filename, 'r') as f:
        args = json.load(f)
    print('CUDA AVAILABLE :', torch.cuda.is_available())
    device = 'cuda'
    train(args, device)
