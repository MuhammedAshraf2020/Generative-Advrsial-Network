import torch.nn as nn

class Critic(nn.Module):
	def __init__(self , n_channels , width):
		super(Critic , self).__init__()
		self.model = nn.Sequential(
					nn.Conv2d(n_channels , width  , 4 , 2 , 1 , bias = False) , 
					nn.LeakyReLU(0.2 , inplace = True) ,
					# 32 , 32 , 64
					nn.Conv2d(width , width * 2 , 4 , 2 , 1 , bias = False),
					nn.BatchNorm2d(width * 2),
					nn.LeakyReLU(0.2),
					#16 , 16 , 128
					nn.Conv2d(width * 2 , width * 4 , 4 , 2 , 1 , bias = False),
					nn.BatchNorm2d(width * 4),
					nn.LeakyReLU(0.2),
					#8 , 8 , 256
					nn.Conv2d(width * 4 , width * 8 , 4 , 2 , 1 , bias = False),
					nn.BatchNorm2d(width * 8),
					nn.LeakyReLU(0.2),
					#4 , 4 , 512
					nn.Conv2d(width * 8 , 1 , 4 , 2 , 0 , bias = False)
					#1 , 1 , 1
			)
	def forward(self , x):
		return self.model(x).view(-1)


class Generator(nn.Module):
	def __init__(self , n_dim , width , n_channels):
		super(Generator , self).__init__()
		self.model = nn.Sequential(
					nn.ConvTranspose2d(n_dim , width * 8 , 4 , 1 , 0 , bias = False),
					nn.BatchNorm2d(width * 8),
					nn.ReLU(True),
					#4 , 4 , 512
					nn.ConvTranspose2d(width * 8  , width * 4 , 4 , 2 , 1 , bias = False),
					nn.BatchNorm2d(width * 4),
					nn.ReLU(True),
					#8 , 8 , 256
					nn.ConvTranspose2d(width *4   , width * 2 , 4 , 2 , 1 , bias = False),
					nn.BatchNorm2d(width * 2),
					nn.ReLU(True),
					#16 , 16 , 64
					nn.ConvTranspose2d(width * 2  , width  , 4 , 2 , 1 , bias = False),
					nn.BatchNorm2d(width),
					nn.ReLU(True),
					#32 , 32 , 64
					nn.ConvTranspose2d(width   , n_channels , 4 , 2 , 1 , bias = False),
					nn.Tanh(),
			)
	def forward(self , x):
		return self.model(x)