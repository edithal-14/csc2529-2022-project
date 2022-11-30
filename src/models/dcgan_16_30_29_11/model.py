import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

from data import BraTSDataset

from IPython.display import HTML

seed = 999
torch.manual_seed(seed)
np.random.seed(seed)

image_size = 240
batch_size = 16
num_workers = 16
device = torch.device("cuda:0")

# Learning rate
lr = 0.0002
# beta 1 for Adam
beta1 = 0.5

# Model hyperparameters
latent_size = 50

# Number of training epochs
num_epochs = 50

def weight_init(m):
    """Custom weight initialization called on netG and netD"""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.normal_(m.bias.data, 0)

# Generator code
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # state size: latent_size * 1 * 1
            nn.ConvTranspose2d(latent_size, 64, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # state size: 64 * 4 * 4
            nn.ConvTranspose2d(64, 16, 4, 4, 0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            # state size: 32 * 8 * 8
            # nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(16),
            # nn.ReLU(True),
            # state size: 16 * 16 * 16
            nn.ConvTranspose2d(16, 4, 4, 4, 0, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(True),
            # state size: 8 * 32 * 32
            # nn.ConvTranspose2d(8, 4, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(4),
            # nn.ReLU(True),
            # state size: 4 * 64 * 64
            nn.ConvTranspose2d(4, 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(True),
            # state size: 2 * 128 * 128
            nn.ConvTranspose2d(2, 1, 4, 2, 9, bias=False),
            nn.Tanh()
            # state size: 1 * 240 * 240
        )
    
    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # state size: 1 * 240 * 240
            nn.Conv2d(1, 2, 4, 2, 9, bias=False),
            nn.BatchNorm2d(2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: 2 * 128 * 128
            nn.Conv2d(2, 8, 4, 4, 0, bias=False),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, inplace=True),
            # # state size: 4 * 64 * 64
            # nn.Conv2d(4, 8, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(8),
            # nn.LeakyReLU(0.2, inplace=True),
            # state size: 8 * 32 * 32
            nn.Conv2d(8, 32, 4, 4, 0, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: 16 * 16 * 16
            # nn.Conv2d(16, 32, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(32),
            # nn.LeakyReLU(0.2, inplace=True),
            # state size: 32 * 8 * 8
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: 64 * 4 * 4
            nn.Conv2d(64, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # state size: 1 * 1 * 1
        )
    
    def forward(self, input):
        return self.main(input)

netD = Discriminator().to(device)
netD.apply(weight_init)
print(netD)

critetion = nn.BCELoss()

fixed_noise = torch.randn(32, latent_size, 1, 1, device=device)

real_label = 1.
fake_label = 0.

optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))