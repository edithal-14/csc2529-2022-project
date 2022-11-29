import torch.nn as nn

# Convolution Block for Generator
class GBlock(nn.Module):
	def __init__(self, ic, oc) -> None:
		super().__init__()

		# store the convolution and ReLU layers
		# let input dims be (D,D)
		self.main = nn.Sequential(
			# state size. (ic)xDxD
			nn.ConvTranspose2d(in_channels=ic,
								out_channels=oc,
								kernel_size=3,
								stride=1,
								padding=0,
								bias=False),
			nn.BatchNorm2d(oc),
			nn.LeakyReLU(inplace=True),
			# state size. (oc)x(D-2)x(D-2)
			nn.ConvTranspose2d(in_channels=oc,
								out_channels=oc,
								kernel_size=3,
								stride=1,
								padding=0,
								bias=False),            
		)
		# state size. (oc)x(D-4)x(D-4)
	
	def forward(self, input):
		return self.main(input)

# Convolution Block for Discriminator
class DBlock(nn.Module):
	def __init__(self, ic, oc, ns=0.01) -> None:
		super().__init__()

		# store the convolution and ReLU layers
		# let input dims be (D,D)
		self.main = nn.Sequential(
			# state size. (ic)xDxD
			nn.Conv2d(in_channels=ic,
						out_channels=oc,
						kernel_size=3,
						stride=1,
						padding=0,
						bias=False),
			nn.LeakyReLU(ns, inplace=True),
			# state size. (oc)x(D-2)x(D-2)
			nn.Conv2d(in_channels=oc,
						out_channels=oc,
						kernel_size=3,
						stride=1,
						padding=0,
						bias=False),            
		)
		# state size. (oc)x(D-4)x(D-4)
	
	def forward(self, input):
		return self.main(input)