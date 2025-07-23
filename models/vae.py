from functools import reduce
import torch
from torch import nn
from torch.nn.modules import conv


class VariationalAutoEncoder(nn.Module):

    def __init__(self, in_channels, z_dim=60, flat_dim_tuple=(64, 4, 4)) -> None:
        super(VariationalAutoEncoder, self).__init__()

        self.flat_dim = reduce((lambda x, y: x * y), flat_dim_tuple)

        # ENCODER
        self.enc_conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=2, stride=1)
        self.enc_conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=2)
        self.enc_bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        # self.enc_maxpool1 = nn.MaxPool2d(kernel_size=3, stride=3)

        self.enc_conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=2)
        self.enc_conv4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

        self.enc_fc1 = nn.Linear(in_features=self.flat_dim, out_features=256)
        self.enc_bn2 = nn.BatchNorm1d(256)
        self.enc_fc2 = nn.Linear(in_features=256, out_features=128)
        self.lkrelu = nn.LeakyReLU()

        self.mu = nn.Linear(in_features=128, out_features=z_dim)
        self.sigma = nn.Linear(in_features=128, out_features=z_dim)

        # DECODER
        self.dec_fc1 = nn.Linear(in_features=z_dim, out_features=256)
        self.dec_bn1 = nn.BatchNorm1d(256)
        self.dec_fc2 = nn.Linear(in_features=256, out_features=self.flat_dim)
        self.unflatten = nn.Unflatten(1, flat_dim_tuple)

        self.dec_conv1 = nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=3, stride=2)
        self.dec_conv2 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2)
        self.dec_conv3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2)
        self.dec_conv4 = nn.ConvTranspose2d(in_channels=32, out_channels=in_channels, kernel_size=2, stride=1)

        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        """
        VAE Encoder q_phi(z|x):

        given an image x, returns its latent space encoding z.

        z is given followig a per-image gaussian.
        """
        # print(f'{x.shape}')
        # print(f'{conv1.shape}')
        # print(f'{conv2.shape}\n\n')

        conv1 = self.enc_conv1(x)
        conv2 = self.enc_conv2(conv1)
        x = self.relu(self.enc_bn1(conv2))

        conv3 = self.enc_conv3(x)
        conv4 = self.enc_conv4(conv3)
        x = self.enc_fc1(self.flatten(conv4))
        x = self.enc_fc2(self.enc_bn2(x))
        x = self.lkrelu(x)

        mu, logvar = self.mu(x), self.sigma(x)
        return mu, logvar, conv1, conv3

    def decode(self, z):
        """
        VAE Encoder p_theta(x|z):

        given a latent space encoding z, returns its image reconstruction x_hat.

        x_hat is meant to be a reconstruction of the image x passed to
        the encoder above OR sampled from p(z).
        """
        x = self.dec_fc2(self.dec_bn1(self.dec_fc1(z)))
        x = self.relu(x)
        x = self.unflatten(x)

        conv1 = self.dec_conv1(x)
        x = self.relu(conv1)
        x = self.dec_conv2(x)
        conv2 = self.dec_conv3(x)
        x = self.dec_conv4(conv2)
        x_hat = self.sigmoid(x)

        return x_hat, conv1, conv2

    def forward(self, x):
        """
        N.B. We learn the log(variance) instead of the variance itself
        to avoid it taking negative values, which, in turn, may lead
        to errors when computing the KL divergence.
        """
        # --- Encoder ---
        mu, logvar, c1, c2 = self.encode(x)
        sigma = torch.exp(0.5 * logvar)  # Compute real std
        eps = torch.randn_like(sigma)    # Add sampled randomness via epsilon
        z = mu + (eps * sigma)           # Reparameterization trick

        # --- Decoder ---
        x_hat, c3, c4 = self.decode(z)

        return mu, sigma, x_hat, (c1, c2, c3, c4)

    def sample(self, z_dim, n_samples=1):
        """
        VAE generation via sampling

        Samples z from the posterior p(z),
        where p(z) = N(0, I)

        z helps the decoder generate an artificial image
        from the learned latent distribution
        """
        z = torch.randn(n_samples, z_dim)
        x_hat = self.decode(z)
        return x_hat


if __name__ == "__main__":
    fake_img = torch.rand([2, 1, 28, 28])
    vae = VariationalAutoEncoder(in_channels=1)
    mu, sigma, x_hat, _ = vae.forward(fake_img)

    space = " "
    print(f'Reconstructed image \t{x_hat.shape}')
    print(f'Z mu shape {space*13}{mu.shape}')
    print(f'Z sigma shape {space*10}{sigma.shape}')
