#%matplotlib inline
import config.unetConfig as cfg
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
import random

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.utils as vutils

from data import BraTSDataset
from gan.GANTrainer import GANTrainer
from gan.UNetGenerator import UNetGenerator
from gan.UNetDiscriminator import UNetDiscriminator
from IPython.display import HTML
from torch.utils.data import DataLoader
from utils import weights_init_norm, weights_init_ortho, plot_train_metrics


# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

def main(bt_class):
    ############################################################
    ####################### LOAD DATA ##########################
    ############################################################
    dpath = cfg.DSET_CPATHS[f"{bt_class}"]

    # load the T1 image filepaths in a sorted manner
    image_paths = [os.path.join(dpath, impath) for impath in sorted(os.listdir(dpath))]

    tf = transforms.Compose([
        transforms.Resize((cfg.INPUT_IMAGE_HEIGHT,cfg.INPUT_IMAGE_WIDTH)),
        transforms.CenterCrop((cfg.INPUT_IMAGE_HEIGHT,cfg.INPUT_IMAGE_WIDTH)),
        transforms.ToTensor(),
        ])

    # Create the dataset
    dataset = BraTSDataset(image_paths, tf)

    ############################################################
    #################### Save Real Images ######################
    ############################################################

    dataset.save(bt_class=bt_class.lower())

    # Create the dataloader
    dataloader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE,
                            shuffle=True, num_workers=cfg.NUM_WORKERS)

    # Plot some training images
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(15,15))
    plt.axis("off")
    plt.title(f"Training Images ({bt_class})")
    img = np.transpose(vutils.make_grid(real_batch.to(cfg.DEVICE), normalize=True).cpu().numpy(),(1,2,0))
    plt.imsave(f"output/training/{cfg.WINIT}_winit/unet_samplereal_{bt_class}.png", img)

    ############################################################
    ############# Init Generator and weights ###################
    ############################################################

    # Create the generator
    netG = UNetGenerator(cfg.LATENT_SZ, cfg.NGF, cfg.NGC).to(cfg.DEVICE)

    # Apply the weights_init_ortho
    netG.apply(weights_init_norm)

    # Print the model
    print(netG)

    ############################################################
    ########### Init Discriminator and weights #################
    ############################################################

    # Create the Discriminator
    netD = UNetDiscriminator(ns=cfg.NEG_SLOPE).to(cfg.DEVICE)

    # Apply the weights_init_ortho
    netD.apply(weights_init_ortho)

    # Print the model
    print(netD)

    ############################################################
    ######################### Train ############################
    ############################################################

    trainer = GANTrainer(num_epochs=cfg.NUM_EPOCHS,
                            glr=cfg.GLR, dlr=cfg.DLR,
                            gbeta1=cfg.GBETA1, dbeta1=cfg.DBETA1,
                            dataloader=dataloader,
                            netG=netG, netD=netD,
                            device=cfg.DEVICE)

    trainer.train(nz=cfg.LATENT_SZ, batch_sz=cfg.BATCH_SIZE)

    ############################################################
    ################ Plot Training Results #####################
    ############################################################

    # Print and save max PSNR from training
    print(f"Highest Training PSNR: {trainer.best_g_psnr} | Highest_Training_SSIM: {trainer.best_g_ssim}")
    with open("output/training_mets.csv", 'a') as f:
        f.writelines(f"\nOrtho Weight init::{bt_class}::\t\
            NUM_EPOCHS==>{cfg.NUM_EPOCHS}\t\
            BATCH_SIZE=={cfg.BATCH_SIZE}\t\
            LATENT_SIZE==>{cfg.LATENT_SZ}\t\
            GLR==>{cfg.GLR}\t\
            DLR==>{cfg.DLR}\t\
            BEST_PSNR==>{trainer.best_g_psnr:.4f}\t\
            BEST_SSIM: {trainer.best_g_ssim:.4f}")

    ############################################################
    ################### Save trained model #####################
    ############################################################

    param_str = f"_{cfg.NUM_EPOCHS}epochs_{cfg.BATCH_SIZE}batch_G_{cfg.LATENT_SZ}z_{cfg.NGF}feat_{cfg.GLR}lr_D_{cfg.NUM_LEVELS}lvl_{cfg.NDF}feat_{cfg.DLR}lr_{cfg.NEG_SLOPE}lRelu"

    torch.save(trainer.best_d_state, f"models/unetgan/{cfg.WINIT}_winit/unetgan_{bt_class}_Dstate_{param_str}.pth")
    torch.save(trainer.best_g_state, f"models/unetgan/{cfg.WINIT}_winit/unetgan_{bt_class}_Gstate_{param_str}.pth")

    ############################################################
    ############ Plot Sample Synthetic Images ##################
    ############################################################

    netGval = UNetGenerator(cfg.LATENT_SZ,cfg.NGF,cfg.NGC).to(cfg.DEVICE)
    netGval.load_state_dict(torch.load(f"models/unetgan/{cfg.WINIT}_winit/unetgan_{bt_class}_Gstate_{param_str}.pth"))
    netGval.eval()

    fixed_noise = torch.randn(cfg.BATCH_SIZE, cfg.LATENT_SZ, 1, 1, device=cfg.DEVICE)
    with torch.no_grad():
        fake = netGval(fixed_noise).detach().cpu()

    img = np.transpose(
            vutils.make_grid(fake, padding=2, scale_each=True, normalize=True),
            (1, 2, 0)
        ).cpu().numpy()
    title_str = f"{cfg.NUM_EPOCHS}epochs_{cfg.BATCH_SIZE}batch_G_{cfg.LATENT_SZ}z_{cfg.GLR}lr_D_{cfg.DLR}lr"
    plt.figure(figsize=(10,10))
    plt.title(title_str)
    plt.imshow(img)
    plt.axis("off")
    plt.imsave(f"output/training/{cfg.WINIT}_winit/unet_samplefake_{bt_class}_{title_str}.png",img)

    ############################################################
    ################ Save Synthetic Images #####################
    ############################################################

    with torch.no_grad():
        fixed_noise = torch.randn(369, cfg.LATENT_SZ, 1, 1, device=cfg.DEVICE)

    fake = netGval(fixed_noise).detach().cpu()

    for i in range(fake.size(0)):
            vutils.save_image(fake[i, :, :, :], f'output/unetgan/{cfg.WINIT}_winit/{bt_class.lower()}/fake/{i}.png')

if __name__ == "__main__":
    for bt_class in cfg.DSET_CPATHS:
        print(f"Training UNET-GAN for Brain Tumor Class: {bt_class}")
        main(bt_class)