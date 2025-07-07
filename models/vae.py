import torch
from torch import nn
from torch.nn.modules import conv


class VariationalAutoEncoder(nn.Module):

    def __init__(self, in_channels, z_dim=60, flat_dim=512) -> None:
        super(VariationalAutoEncoder, self).__init__()

        # ENCODER
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=flat_dim, out_features=128)
        self.bn2 = nn.BatchNorm1d(128)
        self.lkrelu = nn.LeakyReLU()

        self.mu = nn.Linear(in_features=128, out_features=z_dim)
        self.sigma = nn.Linear(in_features=128, out_features=z_dim)

        # DECODER
        self.fc2 = nn.Linear(in_features=z_dim, out_features=128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(in_features=128, out_features=flat_dim)
        self.unflatten = nn.Unflatten(1, (64, 4, 4))
        self.conv3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=3)
        self.conv4 = nn.ConvTranspose2d(in_channels=32, out_channels=in_channels, kernel_size=4, stride=2)
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        """
        VAE Encoder q_phi(z|x):

        given an image x, returns its latent space encoding z.

        z is given followig a per-image gaussian.
        """
        conv1 = self.conv1(x)
        x = self.maxpool1(self.relu(self.bn1(conv1)))
        conv2 = self.conv2(x)
        x = self.maxpool2(self.relu(conv2))
        x = self.lkrelu(self.bn2(self.fc1(self.flatten(x))))

        mu, logvar = self.mu(x), self.sigma(x)
        return mu, logvar, conv1, conv2

    def decode(self, z):
        """
        VAE Encoder p_theta(x|z):

        given a latent space encoding z, returns its image reconstruction x_hat.

        x_hat is meant to be a reconstruction of the image x passed to
        the encoder above OR sampled from p(z).
        """
        x = self.unflatten(self.fc3(self.lkrelu(self.bn3(self.fc2(z)))))
        conv3 = self.conv3(x)
        x = self.relu(conv3)
        conv4 = self.conv4(x)
        x_hat = self.sigmoid(conv4)

        return x_hat, conv3, conv4

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
