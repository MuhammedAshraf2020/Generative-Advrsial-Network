import torch.nn as nn
import torch


class Discriminator(nn.Module):
  def __init__(self , in_units):
    super(Discriminator , self).__init__()
    self.model = nn.Sequential(
				*self.disc_block(in_units , 512 ),
        		*self.disc_block(512 , 128) ,
				 nn.Linear(128 , 1),
				 nn.Sigmoid()
			)
  
  def disc_block(self , in_units , out_units , normalize = True):
    layers =  []
    layers.append(nn.Linear(in_features = in_units , out_features = out_units)) 
    layers.append(nn.LeakyReLU(0.01))
    return layers
  
  def forward(self , x):
    return self.model(x)


class Generator(nn.Module):
	def __init__(self , z_dim , out_shape):
		super(Generator , self).__init__()
		self.model = nn.Sequential(
			*self.gen_block(z_dim , 128 , normalize = False) ,
			*self.gen_block(128 , 256)  ,
			*self.gen_block(256 , 512)  ,
			 nn.Linear(512 , out_shape),
			 nn.Tanh())

	def gen_block(self , in_units , out_units , normalize = True):
		layers = []
		layers.append(nn.Linear(in_units , out_units))
		layers.append(nn.LeakyReLU(0.01 , inplace = True))
		if normalize:
			layers.append(nn.BatchNorm1d(out_units))
		return layers

	def forward(self , x):
		return self.model(x)
