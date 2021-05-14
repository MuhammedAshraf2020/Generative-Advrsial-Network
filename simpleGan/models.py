import torch.nn as nn
import torch


class Discriminator(nn.Module):
  def __init__(self , in_units):
    super(Discriminator , self).__init__()
    self.model = nn.Sequential(
        nn.Linear(in_features = in_units , out_features = 512)
				nn.LeakyReLU(0.01)
        nn.Linear(in_features = 512 , out_features = 128)
				nn.LeakyReLU(0.1)        
        nn.Linear(128 , 1),
				nn.Sigmoid()
			)
  
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
