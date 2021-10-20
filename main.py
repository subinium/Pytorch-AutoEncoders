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
from models.AAE import AAE, Discriminator
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
    elif model_name == 'AAE':
        model = AAE(args).to(device)
        model_D = Discriminator(args).to(device)

    optimizer = Adam(model.parameters(), lr=args['lr'])
    if model_name == 'AAE':
        optimizer_D = Adam(model_D.parameters(), lr=args['lr'])
    
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
            if model_name in ['VAE', 'CVAE', 'AAE']:
                x, label = x
            x = x.to(device)

            

            if model_name in ['VAE', 'BetaVAE']:
                x_hat, loss = model(x)
            if model_name == 'CVAE':
                label_ohe = F.one_hot(label, num_classes=10).to(device)
                x_hat, loss = model(x, label_ohe)
            if model_name == 'AAE':
                valid = torch.ones((x.shape[0], 1), requires_grad=False).to(device)
                fake =  torch.zeros((x.shape[0], 1), requires_grad=False).to(device)
                x_hat, fake_z = model(x)
                
                loss = 0.001*F.binary_cross_entropy(model_D(fake_z), valid) + 0.999*F.mse_loss(x_hat.cuda(), x.cuda())
                
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if model_name == 'AAE':
                optimizer_D.zero_grad()
                real_z = torch.normal(0, 1, size=(x.shape[0], args['latent_dim'])).to(device)
                real_loss = F.binary_cross_entropy(model_D(real_z), valid)
                fake_loss = F.binary_cross_entropy(model_D(fake_z.detach()), fake)
                D_loss = 0.5 * (real_loss + fake_loss)
                D_loss.backward()
                optimizer_D.step()

            if iteration % 100 == 0:
                logger.add_scalar(f'Loss/train_{model_name}', loss, iteration)
                logger.add_images(f'Image/train_{model_name}', torch.stack([x.cpu(), x_hat.cpu()], dim=0).view(-1, *(x.shape[1:])), iteration)
            
                

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
