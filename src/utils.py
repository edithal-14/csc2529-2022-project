import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

# Normal weight init
def weights_init_norm(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Orthogonal weight init
def weights_init_ortho(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.orthogonal(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        nn.init.orthogonal(m.weight.data)
        nn.init.constant_(m.bias.data, 0)

def plot_train_metrics(G_losses,D_losses):
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()