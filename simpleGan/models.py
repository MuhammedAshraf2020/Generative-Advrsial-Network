import torch.nn as nn


						                                     #=====================#
						                                     #       Generator     #
						                                     #=====================#
class Generator(nn.Module):
  def __init__(self , z_dim , img_size):
    super(Generator , self).__init__()
    layers = [
					*self.block(z_dim , 128) ,
					*self.block(128   , 256 , normalize = True) ,
					*self.block(256   , 512 , normalize = True) ,
					nn.Linear(512    , img_size ) ,
					nn.Tanh()]
    self.model = nn.Sequential(*layers)
  
  def forward(self , x):
    return self.model(x)
  
  def block(self , in_channels , out_channels , normalize = False):
    layers = [nn.Linear(in_channels , out_channels)]
    if normalize:
      layers.append(nn.BatchNorm1d(out_channels , 0.8))
    layers.append(nn.LeakyReLU(0.1 , inplace = True))
    return layers



						                                     #=====================#
						                                     #    Discriminator    #
						                                     #=====================#

class Discriminator(nn.Module):
	def __init__(self , img_size):
		
		super(Discriminator , self).__init__()
		self.model = nn.Sequential(
			nn.Linear(img_size , 512) ,
			nn.LeakyReLU(0.01)   ,
			nn.Linear(512 , 128) ,
			nn.LeakyReLU(0.1),
			nn.Linear(128 , 1),
			nn.Sigmoid())
	
	def forward(self , x):
		return self.model(x)