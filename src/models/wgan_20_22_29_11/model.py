import os

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.utils as vutils
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
import numpy as np

from data import BraTSDataset

from IPython.display import HTML

seed = 999
torch.manual_seed(seed)
np.random.seed(seed)

# Dataset configuration
dataset_root = "dataset"
t1_train_data = "data/MICCAI_BraTS2020/train/t1"

# Critic model configuration
use_gp = True
lambda_gp = 10
# clipping param won't be used if use_gp is True
clipping_param = 0.01

# Generator model configuration
latent_size = 200

# Training configuration
device = torch.device("cuda:0")
batch_size = 16
real_label = -1.0
fake_label = 1.0
n_epochs = 10
n_critic = 5

# Dataset configuration
image_size = 240
num_workers = 16

image_paths = [os.path.join(t1_train_data, impath) for impath in os.listdir(t1_train_data)]
tf = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    ])
dataset = BraTSDataset(image_paths, tf)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

real_batch = next(iter(dataloader))

# Plot some training images
print(real_batch.shape)
# Actually plot it
plt.figure(figsize=(15, 15))
plt.axis("off")
plt.title("Training Images")
plt.imshow(
    np.transpose(
        vutils.make_grid(real_batch.to(device)[:16], padding=2, normalize=True).cpu(),
        (1, 2, 0)
    )
)

def weight_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, 0.0, 0.02)

class Critic(nn.Module):
    def __init__(self, use_gp=False) -> None:
        super(Critic, self).__init__()
        if use_gp:
            self.main = nn.Sequential(
                # input: 1 x 240 x 240
                nn.Conv2d(1, 128, 4, stride=2, padding=9),
                nn.LeakyReLU(0.2),
                # input: 128 x 128 x 128
                nn.Conv2d(128, 64, 4, stride=2, padding=1),
                nn.LeakyReLU(0.2),
                # input: 64 x 64 x 64
                nn.Conv2d(64, 32, 4, stride=2, padding=1),
                nn.LeakyReLU(0.2),
                # input: 32 x 32 x 32
                nn.Conv2d(32, 16, 4, stride=2, padding=1),
                nn.LeakyReLU(0.2),
                # input: 16 x 16 x 16
                nn.Conv2d(16, 8, 4, stride=2, padding=1),
                nn.LeakyReLU(0.2),
                # input: 8 x 8 x 8
                nn.Conv2d(8, 4, 4, stride=2, padding=1),
                nn.LeakyReLU(0.2),
                # input: 4 x 4 x 4
                nn.Conv2d(4, 1, 4, stride=1, padding=0),
                # input: 1 x 1 x 1
            )
        else:
            self.main = nn.Sequential(
                # input: 1 x 240 x 240
                nn.Conv2d(1, 128, 4, stride=2, padding=9),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),
                # input: 128 x 128 x 128
                nn.Conv2d(128, 64, 4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2),
                # input: 64 x 64 x 64
                nn.Conv2d(64, 32, 4, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.2),
                # input: 32 x 32 x 32
                nn.Conv2d(32, 16, 4, stride=2, padding=1),
                nn.BatchNorm2d(16),
                nn.LeakyReLU(0.2),
                # input: 16 x 16 x 16
                nn.Conv2d(16, 8, 4, stride=2, padding=1),
                nn.BatchNorm2d(8),
                nn.LeakyReLU(0.2),
                # input: 8 x 8 x 8
                nn.Conv2d(8, 4, 4, stride=2, padding=1),
                nn.BatchNorm2d(4),
                nn.LeakyReLU(0.2),
                # input: 4 x 4 x 4
                nn.Conv2d(4, 1, 4, stride=1, padding=0),
                # input: 1 x 1 x 1
            )
    
    def forward(self, input):
        return self.main(input)

critic_model = Critic(use_gp=use_gp).to(device)
critic_model.apply(weight_init)

class Generator(nn.Module):
    def __init__(self) -> None:
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input: latent_size x 1 x 1
            nn.ConvTranspose2d(latent_size, 128, 4, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            # input: 128 x 4 x 4
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            # input: 64 x 8 x 8
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            # input: 32 x 16 x 16
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            # input: 16 x 32 x 32
            nn.ConvTranspose2d(16, 8, 4, 2, 1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2),
            # input: 8 x 64 x 64
            nn.ConvTranspose2d(8, 4, 4, 2, 1),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.2),
            # input: 4 x 128 x 128
            nn.ConvTranspose2d(4, 1, 4, 2, 9),
            nn.Tanh()
            # output: 1 x 240 x 240
        )
    
    def forward(self, input):
        return self.main(input)

generator_model = Generator().to(device)
generator_model.apply(weight_init)

optimizer_critic = optim.RMSprop(critic_model.parameters(), lr=5e-5)
optimizer_generator = optim.RMSprop(generator_model.parameters(), lr=5e-5)