import torch.nn as nn

from ConvBlock import GBlock as Block

# Generator2 Class
class UNetGenerator2(nn.Module):
	def __init__(self, channels=(1, 1024, 512, 256, 128, 64)) -> None:
		super().__init__()

		self.channels = channels
		# store generator blocks and transpose conv layers (for upsampling)
		self.upsamplings = nn.ModuleList(
			[nn.ConvTranspose2d(in_channels=self.channels[i],
									out_channels=self.channels[i+1],
									kernel_size=2,
									stride=2,
									padding=0,
									bias=False)
				for i in range(len(self.channels)-1)]
		)
		self.g_blocks = nn.ModuleList(
			[Block(self.channels[i], self.channels[i+1])
				for i in range(len(self.channels)-1)]
		)

	def forward(self,x):
		# loop through number of channels
		for i in range(len(self.channels)-1):
			# pass the inputs through the upsampler block
			x = self.upsamplings[i](x)
			# pass concatenated features to generator block
			x = self.g_blocks[i](x)
		
		return x