import torch 
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        input_dim = args['input_dim']
        hidden_dim = args['hidden_dim']
        latent_dim = args['latent_dim']
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=2),
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
            nn.Linear(latent_dim, 32*7*7),
            nn.Unflatten(1, (32, 7, 7)),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1 , padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, args):
        super().__init__()
        latent_dim = args['latent_dim']
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)

class AAE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)
        
    def reparametrize(self, mu, logvar):
        epsilon = torch.randn_like(logvar)
        z = mu + torch.exp(0.5*logvar)*epsilon
        return z

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparametrize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, z


if __name__ == '__main__':
    args = {"input_dim":784, "latent_dim":32, "output_dim":784, "hidden_dim":128}
    encoder = Encoder(args)
    input = torch.rand(32, 1, 28, 28)
    VAE = VanillaVAE(args)
    x, _ = VAE(input)
    print(x.shape)
    