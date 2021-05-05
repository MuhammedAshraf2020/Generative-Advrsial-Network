import torch.nn as nn

class Generator(nn.Module):
  def __init__(self , z_dim , img_channels , img_size):
    super(Generator , self).__init__()
    self.model = nn.Sequential(
			self.block(z_dim         , img_size*16 , 4 , 1 , 0 ),			
			self.block(img_size*16   , img_size*8  , 4 , 2 , 1 ),
			self.block(img_size*8    , img_size*4  , 4 , 2 , 1 ),
			self.block(img_size*4    , img_size*2  , 4 , 2 , 1 ),
			nn.ConvTranspose2d(img_size * 2 , img_channels , kernel_size = 4 , stride = 2 , padding = 1) ,
			nn.Tanh()
			)						                        
  
  def block(self ,in_channels , out_channels  , kernel_size, stride , padding):
    return nn.Sequential(
			nn.ConvTranspose2d(in_channels , out_channels , kernel_size , stride , padding , bias = False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU()
			)
  
  def forward(self , img):
    return self.model(img)



class Discriminator(nn.Module):
	def __init__(self , img_channels , img_size):
		super(Discriminator , self).__init__()
		self.model = nn.Sequential(
			nn.Conv2d(img_channels , img_size , kernel_size = 4 , stride = 2 , padding = 1) ,
			nn.LeakyReLU(0.02) ,
			self.block(img_size   , img_size*2 , 4 , 2 , 1 ),
			self.block(img_size*2 , img_size*4 , 4 , 2 , 1 ),
			self.block(img_size*4 , img_size*8 , 4 , 2 , 1 ),
			nn.Conv2d(img_size *8 , 1 , kernel_size = 4 , stride = 2 , padding = 0) ,
			nn.Sigmoid()
			)

	
	def block(self , in_channels , out_channels  , kernel_size, stride , padding ):
		return nn.Sequential(
			nn.Conv2d(in_channels , out_channels , kernel_size , stride , padding , bias = False) ,
			nn.BatchNorm2d(out_channels),
			nn.LeakyReLU(0.2) ,

			)

	def forward(self , x):
		return self.model(x)