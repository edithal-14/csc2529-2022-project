import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

class GANTrainer():
    def __init__(self, num_epochs, glr, dlr, gbeta1, dbeta1, dataloader, netG, netD, device) -> None:
        self.num_epochs = num_epochs
        self.device = device
        self.glr = glr
        self.dlr = dlr
        self.gbeta1 = gbeta1
        self.dbeta1 = dbeta1

        # Establish convention for real and fake labels during training
        self.real_label = 1.
        self.fake_label = 0.

        # Dataloader
        self.dataloader = dataloader

        # Generator
        self.netG = netG
        # Discriminator
        self.netD = netD

        # Loss function
        self.criterion = nn.BCELoss()
        # Optimizers
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=self.dlr, betas=(self.dbeta1, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=self.glr, betas=(self.gbeta1, 0.999))

        # Lists to keep track of progress
        self.img_list = []
        self.G_losses = []
        self.D_losses = []
        self.psnr_hist = []
        self.ssim_hist = []
        self.iters = 0
        self.best_g_psnr = 0
        self.best_g_ssim = 0
        self.best_g_state = None
        self.best_d_state = None

    def getBatchMeanPSNR(self, rbatch, fbatch):
        mpsnr = 0
        b_size = rbatch.shape[0]
        for i in range(b_size):
            mpsnr += np.round(psnr(
                np.transpose(rbatch[i,:,:,:],(1,2,0)),
                np.transpose(fbatch[i,:,:,:],(1,2,0)),
                data_range=1), decimals=4)
        return mpsnr/b_size

    def getBatchMeanSSIM(self, rbatch, fbatch):
        mssim = 0
        b_size = rbatch.shape[0]
        for i in range(b_size):
            mssim += np.round(ssim(
                rbatch[i,:,:,:],
                fbatch[i,:,:,:],
                channel_axis=0,
                data_range=1.), decimals=4)
        return mssim/b_size

    def train(self, nz, batch_sz):
        # Create batch of latent vectors that we will use to visualize
        #  the progression of the generator
        fixed_noise = torch.randn(batch_sz, nz, 1, 1, device=self.device)

        print("Starting Training Loop...")
        # For each epoch
        for epoch in range(self.num_epochs):
            # For each batch in the dataloader
            for i, data in enumerate(self.dataloader, 0):

                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                ## Train with all-real batch
                self.netD.zero_grad()
                # Format batch
                real = data.to(self.device)
                b_size = real.size(0)
                label = torch.full((b_size,), self.real_label, dtype=torch.float, device=self.device)
                # Forward pass real batch through D
                output = self.netD(real).view(-1)
                # Calculate loss on all-real batch
                errD_real = self.criterion(output, label)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()

                ## Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, nz, 1, 1, device=self.device)
                # Generate fake image batch with G
                fake = self.netG(noise)
                label.fill_(self.fake_label)
                # Classify all fake batch with D
                output = self.netD(fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                errD_fake = self.criterion(output, label)
                # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                # Compute error of D as sum over the fake and the real batches
                errD = errD_real + errD_fake
                # Update D
                self.optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                self.netG.zero_grad()
                label.fill_(self.real_label)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = self.netD(fake).view(-1)
                # Calculate G's loss based on this output
                errG = self.criterion(output, label)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                self.optimizerG.step()

                # Check PSNR and SSIM scores and update best_G_weights accordingly
                # generate fake images after updating generator weights
                fake = self.netG(fixed_noise).detach().cpu().numpy()
                real = real.cpu().numpy()
                mpsnr = self.getBatchMeanPSNR(real,fake)
                mssim = self.getBatchMeanSSIM(real,fake)

                # Save best model based on PSNR every iteration
                self.psnr_hist.append(mpsnr)
                self.ssim_hist.append(mssim)

                # Currently, updating based on PSNR values
                if mpsnr > self.best_g_psnr:
                    self.best_g_state = self.netG.state_dict()
                    self.best_d_state = self.netD.state_dict()
                    self.best_g_psnr = mpsnr
                    self.best_g_ssim = mssim

                # Save Losses for plotting later
                self.G_losses.append(errG.item())
                self.D_losses.append(errD.item())

                # Output training stats
                if i % 12 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f\tMPSNR: %.4f\tMSSIM: %.4f'
                        % (epoch, self.num_epochs, i, len(self.dataloader),
                            errD.item(), errG.item(), D_x, D_G_z1, D_G_z2,
                            mpsnr, mssim))

                # Check how the generator is doing by saving G's output on fixed_noise
                if (self.iters % 12 == 0) or ((epoch == self.num_epochs-1) and (i == len(self.dataloader)-1)):
                    with torch.no_grad():
                        fake = self.netG(fixed_noise).detach().cpu()
                    self.img_list.append(vutils.make_grid(fake, scale_each=True, normalize=True))

                self.iters += 1