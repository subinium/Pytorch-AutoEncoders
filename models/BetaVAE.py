import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        input_dim = args['input_dim']
        hidden_dim = args['hidden_dim']
        latent_dim = args['latent_dim']
        self.model = nn.Sequential(
            nn.Conv2d(input_dim, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 256, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, x):
        h = self.model(x)
        mu = self.fc_mean(h)
        logvar = self.fc_var(h)
        
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        input_dim = args['input_dim']
        hidden_dim = args['hidden_dim']
        latent_dim = args['latent_dim']
        output_dim = args['output_dim']
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.Unflatten(1, (256, 1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 64, kernel_size=4),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, output_dim, kernel_size=4, stride=2 , padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        
        return self.model(x)

class BetaVAE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)
        
    def reparametrize(self, mu, logvar):
        epsilon = torch.randn_like(logvar)
        z = mu + torch.exp(0.5*logvar)*epsilon
        return z

    def loss_calc(self, x, x_hat, mu, logvar, beta=10.0):
        BCE = F.binary_cross_entropy(x_hat, x, reduction='sum')
        KLD = - 0.5 * torch.sum(1+ logvar - mu.pow(2) - logvar.exp())
        return BCE + beta*KLD

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparametrize(mu, logvar)
        x_hat = self.decoder(z)
        
        loss = self.loss_calc(x, x_hat, mu, logvar)
        return x_hat, loss


if __name__ == '__main__':
    args = {"input_dim":3, "latent_dim":32, "output_dim":3, "hidden_dim":256}
    encoder = Encoder(args)
    input = torch.rand(32, 3, 64, 64)
    VAE = BetaVAE(args)
    x, _ = VAE(input)
    print(x.shape)
    