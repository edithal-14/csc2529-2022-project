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
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from data import BraTSDataset

from IPython.display import HTML

# Dataset configuration
dataset_root = "dataset"
t1_train_data = "data/MICCAI_BraTS2020/train/t1"
# Original image size is 240, so compress by 4 time
image_size = 64
num_workers = 16

# Critic model configuration
use_gp = False
lambda_gp = 10
# clipping param won't be used if use_gp is True
clipping_param = 0.01

# Generator model configuration
latent_size = 128
feature_map_size = image_size

# Training configuration
device = torch.device("cuda:0")
batch_size = 32
n_epochs = 100
n_critic = 5
lr = 0.001
beta1 = 0.5

def weight_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, 0.0, 0.02)

class Critic(nn.Module):
    def __init__(self, use_gp=False) -> None:
        super(Critic, self).__init__()
        if use_gp:
            self.main = nn.Sequential(
                # input: 1 x image_size x image_size
                nn.Conv2d(1, feature_map_size, 4, 2, 1),
                nn.LeakyReLU(0.2),
                # input: feature_map_size x image_size/2 x image_size/2
                nn.Conv2d(feature_map_size, feature_map_size * 2, 4, 2, 1),
                nn.LeakyReLU(0.2),
                # input: feature_map_size*2 x image_size/4 x image_size/4
                nn.Conv2d(feature_map_size * 2, feature_map_size * 4, 4, 2, 1),
                nn.LeakyReLU(0.2),
                # input: feature_map_size*4 x image_size/8 x image_size/8
                nn.Conv2d(feature_map_size * 4, feature_map_size * 8, 4, 2, 1),
                nn.LeakyReLU(0.2),
                # input: feature_map_size*8 x image_size/16 x image_size/16
                nn.Conv2d(feature_map_size * 8, 1, 4, 1, 0),
                # output: 1 x 1 x 1
            )
        else:
            self.main = nn.Sequential(
                # input: 1 x image_size x image_size
                nn.Conv2d(1, feature_map_size, 4, 2, 1),
                nn.BatchNorm2d(feature_map_size),
                nn.LeakyReLU(0.2),
                # input: feature_map_size x image_size/2 x image_size/2
                nn.Conv2d(feature_map_size, feature_map_size * 2, 4, 2, 1),
                nn.BatchNorm2d(feature_map_size * 2),
                nn.LeakyReLU(0.2),
                # input: feature_map_size*2 x image_size/4 x image_size/4
                nn.Conv2d(feature_map_size * 2, feature_map_size * 4, 4, 2, 1),
                nn.BatchNorm2d(feature_map_size * 4),
                nn.LeakyReLU(0.2),
                # input: feature_map_size*4 x image_size/8 x image_size/8
                nn.Conv2d(feature_map_size * 4, feature_map_size * 8, 4, 2, 1),
                nn.BatchNorm2d(feature_map_size * 8),
                nn.LeakyReLU(0.2),
                # input: feature_map_size*8 x image_size/16 x image_size/16
                nn.Conv2d(feature_map_size * 8, 1, 4, 1, 0),
                # output: 1 x 1 x 1
            )
    
    def forward(self, input):
        return self.main(input)

class Generator(nn.Module):
    def __init__(self) -> None:
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input: latent_size x 1 x 1
            nn.ConvTranspose2d(latent_size, feature_map_size * 8, 4, 1, 0),
            nn.BatchNorm2d(feature_map_size * 8),
            nn.LeakyReLU(0.2),
            # input: feature_map_size*16 x image_size/16 x image_size/16
            nn.ConvTranspose2d(feature_map_size * 8, feature_map_size * 4, 4, 2, 1),
            nn.BatchNorm2d(feature_map_size * 4),
            nn.LeakyReLU(0.2),
            # input: feature_map_size*8 x image_size/8 x image_size/8
            nn.ConvTranspose2d(feature_map_size * 4, feature_map_size * 2, 4, 2, 1),
            nn.BatchNorm2d(feature_map_size * 2),
            nn.LeakyReLU(0.2),
            # input: feature_map_size*4 x image_size/4 x image_size/4
            nn.ConvTranspose2d(feature_map_size*2, feature_map_size, 4, 2, 1),
            nn.BatchNorm2d(feature_map_size),
            nn.LeakyReLU(0.2),
            # input: feature_map_size*2 x image_size/2 x image_size/2
            nn.ConvTranspose2d(feature_map_size, 1, 4, 2, 1),
            nn.Tanh(),
            # output: 1 x image_size x image_size
        )
    
    def forward(self, input):
        return self.main(input)