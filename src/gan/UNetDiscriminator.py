import torch
import torch.nn as nn
import torchvision.transforms as transforms

# Convolution Block
class Block(nn.Module):
	def __init__(self, ic, oc) -> None:
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
			nn.LeakyReLU(0.2,inplace=True),
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

# Encoder Class
class Encoder(nn.Module):
	def __init__(self, channels=(1, 64, 128, 256, 512)) -> None:
		super().__init__()

		self.channels = channels
		# store encoder blocks and downsampling layer
		self.enc_blocks = nn.ModuleList(
			[Block(self.channels[i], self.channels[i+1]) 
				for i in range(len(self.channels)-1)]
		)
		self.downsample = nn.AvgPool2d(2)

	def forward(self, x):
		# store intermediate block outputs for skip connections
		block_outputs = []

		# loop through the encoder blocks
		for block in self.enc_blocks:
			# pass input ([ic]xDxD) to current encoder block
			x = block(x)
			# store the output of encoder block
			block_outputs.append(x)
			# downsample
			x = self.downsample(x)
			# state size. (oc)x(D-4)/2x(D-4)/2
		
		# return the block_outputs
		return block_outputs
	
# Decoder Class
class Decoder(nn.Module):
	def __init__(self, channels=(512, 256, 128, 64)) -> None:
		super().__init__()

		self.channels = channels
		# store decoder blocks and transpose conv layers (for upsampling)
		self.upsamplings = nn.ModuleList(
			[nn.ConvTranspose2d(in_channels=self.channels[i],
									out_channels=self.channels[i+1],
									kernel_size=2,
									stride=2,
									padding=0,
									bias=False)
				for i in range(len(self.channels)-1)]
		)
		self.dec_blocks = nn.ModuleList(
			[Block(self.channels[i], self.channels[i+1])
				for i in range(len(self.channels)-1)]
		)
	
	def crop(self, enc_features, x):
		# grab the dimensions of the inputs, and crop the encoder
		# features to match the dimensions
		(_, _, H, W) = x.shape
		enc_features = transforms.CenterCrop((H, W))(enc_features)
		# return the cropped features
		return enc_features

	def forward(self,x,enc_features):
		# loop through number of channels
		for i in range(len(self.channels)-1):
			# pass the inputs through the upsampler block
			x = self.upsamplings[i](x)
			# crop the current features from the encoder block
			enc_feature = self.crop(enc_features[i], x)
			# concatenate to the current upsampled features
			x = torch.cat([x,enc_feature], dim=1)
			# pass concatenated features to decoder block
			x = self.dec_blocks[i](x)
		
		return x

# UNet Class
class UNetDiscriminator(nn.Module):
	def __init__(self, enc_channels=(1, 16, 32, 64),
		 dec_channels=(64, 32, 16),
		 nb_classes=1):
		super().__init__()
		
		# initialize the encoder and decoder
		self.encoder = Encoder(enc_channels)
		self.decoder = Decoder(dec_channels)
		
		# initialize the regression head and store the class variables
		self.head = nn.Conv2d(in_channels=dec_channels[-1],
								out_channels=nb_classes,
								kernel_size=88,
								stride=1,
								padding=0,
								bias=False)

	def forward(self, x):
		# grab the features from the encoder
		enc_features = self.encoder(x)
		
		# pass the encoder features through decoder making sure that
		# their dimensions are suited for concatenation
		dec_features = self.decoder(enc_features[::-1][0],
			enc_features[::-1][1:])
		
		# pass the decoder features through the regression head to
		# obtain the output probability
		out = self.head(dec_features)
		
		# Pass it through sigmoid activation
		pre_activation = nn.Sigmoid()
		out = pre_activation(out)
		
		# return the segmentation map
		return out