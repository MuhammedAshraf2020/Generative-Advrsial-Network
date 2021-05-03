import torch 
import argparse
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
from  torch.utils.data import DataLoader
import torchvision.transforms as transforms
from models import Discriminator , Generator
from torch.utils.tensorboard import SummaryWriter 
from torch.utils.tensorboard import SummaryWriter 

parser = argparse.ArgumentParser()
parser.add_argument("--img_size"      , type = int   ,  default = 28     , help = "The number of pixels in each channel")
parser.add_argument("--num_channels"  , type = int   ,  default = 1      , help = "The number of channel 1 for grayscale , 3 for color image")
parser.add_argument("--epochs"        , type = int   ,  default = 50     , help = "The number of epochs")
parser.add_argument("--batch_size"    , type = int   ,  default = 64     , help = "Latent numbers ")
parser.add_argument("--lr"            , type = float ,  default = 2e-4   , help = "Learning Rate" )
parser.add_argument("--z_dim"         , type = int   ,  default = 64     , help = "Latent variable")
parser.add_argument("--write_path"    ,                 default = "logs" , )
args = parser.parse_args()

lr          = args.lr
writer_fake = SummaryWriter("{log}/fake".format(log = args.write_path))
writer_real = SummaryWriter("{log}/real".format(log = args.write_path))
img_shape   = [args.img_size , args.img_size , args.num_channels]
batch_size  = args.batch_size
epochs      = args.epochs
z_dim       = args.z_dim
device      = "cuda" if torch.cuda.is_available() else "cpu"
img_size    = np.prod(img_shape)
step        = 0

print("Learning rate   = " , lr)
print("Batch Size      = " , batch_size)
print("Epochs          = " , epochs)
print("Latent variable = " , z_dim)
print("Device used     = " , device)
print("Image shape     = " , img_shape)

discriminator = Discriminator(img_size).to(device)
generator     = Generator(z_dim , img_size).to(device)
fixed_noise = torch.randn((batch_size, z_dim)).to(device)
transform     = transforms.Compose([
	transforms.ToTensor() ,
	transforms.Normalize((0.5,), (0.5,))])

opt_disc = optim.Adam(discriminator.parameters() , lr = lr) 
opt_gen  = optim.Adam(generator.parameters() , lr = lr)
advarsial_loss = torch.nn.BCELoss()
dataset = datasets.MNIST(root="dataset/", transform=transform, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
criterion = nn.BCELoss()
#Train Model

for epoch in range(epochs):
  for batch_idx , (real , _ ) in enumerate(loader):

		#create real , fake data
    real  = real.view(-1 , img_size).to(device)
    noise = torch.randn(batch_size , z_dim).to(device)
    fake  = generator(noise)
		
		#Discriminator Trainig
    disc_real  = discriminator(real).view(-1)
    disc_fake  = discriminator(fake).view(-1)

    real_label = torch.ones_like(disc_real) 
    fake_label = torch.zeros_like(disc_fake)
    
    lossD_real =  criterion(disc_real , real_label)
    lossD_fake =  criterion(disc_fake , fake_label)
    lossD      = (lossD_real + lossD_fake) / 2 
    discriminator.zero_grad()
    lossD.backward(retain_graph=True)
    opt_disc.step()

		#Generator trainig
    disc_fake = discriminator(fake).view(-1)
    lossG = criterion(disc_fake , torch.ones_like(disc_fake))
    generator.zero_grad()
    lossG.backward()
    opt_gen.step()
    if batch_idx == 0:
      print(f"Epoch [{epoch}/{epochs}] Batch {batch_idx}/{len(loader)} \ Loss D: {lossD:.4f}, loss G: {lossG:.4f}")
      with torch.no_grad():
        fake = generator(fixed_noise).reshape(-1, 1, 28, 28)
        data = real.reshape(-1, 1, 28, 28)
        img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
        img_grid_real = torchvision.utils.make_grid(data, normalize=True)
        writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )
        writer_real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )
        step += 1

torch.save(generator.state_dict() , "weights.pt")