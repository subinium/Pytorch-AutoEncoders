import torch 
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        input_dim = args['input_dim']
        condition_dim = args['condition_dim']
        hidden_dim = args['hidden_dim']
        latent_dim = args['latent_dim']
        self.model = nn.Sequential(
            nn.Conv2d(1+condition_dim, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, x, c):
        c = c.unsqueeze(2).unsqueeze(3)
        x = torch.cat((x, c.expand(32, 10, 28, 28)), 1)
        h = self.model(x)
        mu = self.fc_mean(h)
        logvar = self.fc_var(h)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        input_dim = args['input_dim']
        condition_dim = args['condition_dim']
        hidden_dim = args['hidden_dim']
        latent_dim = args['latent_dim']
        output_dim = args['output_dim']
        self.model = nn.Sequential(
            nn.Linear(latent_dim+condition_dim, 32*7*7),
            nn.Unflatten(1, (32, 7, 7)),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1 , padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, z, c):
        x = torch.cat((z, c), 1)
        return self.model(x)

class CVAE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)
        
    def reparametrize(self, mu, logvar):
        epsilon = torch.randn_like(logvar)
        z = mu + torch.exp(0.5*logvar)*epsilon
        return z

    def loss_calc(self, x, x_hat, mu, logvar):
        BCE = F.binary_cross_entropy(x_hat, x, reduction='sum')
        KLD = - 0.5 * torch.sum(1+ logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

    def forward(self, x, c):
        mu, logvar = self.encoder(x, c)
        z = self.reparametrize(mu, logvar)
        x_hat = self.decoder(z, c)
        loss = self.loss_calc(x, x_hat, mu, logvar)
        return x_hat, loss



if __name__ == '__main__':
    args = {"input_dim":784, "hidden_dim":100, "latent_dim":50, "output_dim":784}
    encoder = Encoder(args)
    input = torch.rand(32, 784)
    vae = VanillaVAE(args)
    x_hat, loss = vae(input)
    print(x_hat.shape, loss)
    