import torch.nn as nn

class UNetGenerator(nn.Module):
    """
    (UpConv => Normalization => Non-Linearity)*repeat_x_times
    """
    def __init__(self, nz, ngf, nc) -> None:
        super(UNetGenerator, self).__init__()

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            # ngf = 32, nz = 128, nc=1
            nn.ConvTranspose2d(in_channels=nz,
                                out_channels=ngf*8,
                                kernel_size=4,
                                stride=1,
                                padding=0,
                                bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace=True),
            # state size. (ngf*8)x4x4
            nn.ConvTranspose2d(in_channels=ngf*8,
                                out_channels=ngf*4,
                                kernel_size=4,
                                stride=2,
                                padding=1,
                                bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace=True),
            # state size. (ngf*4)x8x8
            nn.ConvTranspose2d(in_channels=ngf*4,
                                out_channels=ngf*2,
                                kernel_size=4,
                                stride=2,
                                padding=1,
                                bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
            # state size. (ngf*2)x16x16
            nn.ConvTranspose2d(in_channels=ngf*2,
                                out_channels=ngf,
                                kernel_size=4,
                                stride=2,
                                padding=1,
                                bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),
            # state size. (ngf)x32x32
            nn.ConvTranspose2d(in_channels=ngf,
                                out_channels=nc,
                                kernel_size=4,
                                stride=2,
                                padding=1,
                                bias=False),
            nn.Tanh()
            # state size. (1)x64x64
        )

    def forward(self, input):
        return self.main(input)