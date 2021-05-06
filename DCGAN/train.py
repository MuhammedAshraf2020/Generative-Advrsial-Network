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
parser.add_argument("--img_size"      , type = int   ,  default = 64     , help = "The number of pixels in each channel")
parser.add_argument("--num_channels"  , type = int   ,  default = 1      , help = "The number of channel 1 for grayscale , 3 for color image")
parser.add_argument("--epochs"        , type = int   ,  default = 20     , help = "The number of epochs")
parser.add_argument("--batch_size"    , type = int   ,  default = 32     , help = "Latent numbers ")
parser.add_argument("--lr"            , type = float ,  default = 2e-4   , help = "Learning Rate" )
parser.add_argument("--z_dim"         , type = int   ,  default = 100    , help = "Latent variable")
parser.add_argument("--data_path"                                        , help = "Path of data")
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
img_size    = args.img_size
step        = 0
num_channels= args.num_channels

print("Learning rate   = " , lr)
print("Batch Size      = " , batch_size)
print("Epochs          = " , epochs)
print("Latent variable = " , z_dim)
print("Device used     = " , device)
print("Image shape     = " , img_shape)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

discriminator = Discriminator(num_channels , img_size).to(device)
generator     = Generator(z_dim ,num_channels , img_size ).to(device)

generator.apply(weights_init)
discriminator.apply(weights_init)

fixed_noise = torch.randn((batch_size, z_dim , 1 , 1)).to(device)

transform     = transforms.Compose([
    transforms.Resize(img_size) ,
    transforms.CenterCrop(img_size),
    transforms.ToTensor() ,
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

opt_disc = optim.Adam(discriminator.parameters() , lr = lr , betas=(0.5 , 0.999)) 
opt_gen  = optim.Adam(generator.parameters() , lr = lr , betas=(0.5 , 0.999))
criterion = nn.BCELoss()

dataset = datasets.ImageFolder(root = args.data_path , transform = transform)
loader  = DataLoader(dataset, batch_size = 32 , shuffle=True)

#Train Model
generator.train()
discriminator.train()

for epoch in range(epochs):
  for batch_idx , (real , _ ) in enumerate(loader):

        #create real , fake data
    real  = real.to(device)
    noise = torch.randn(batch_size , z_dim , 1 , 1).to(device)
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
    if batch_idx %200 == 0:
      print(f"Epoch [{epoch}/{epochs}] Batch {batch_idx}/{len(loader)} \ Loss D: {lossD:.4f}, loss G: {lossG:.4f}")
      with torch.no_grad():
        fake = generator(fixed_noise)
        data = real
        img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
        img_grid_real = torchvision.utils.make_grid(data[:32], normalize=True)
        writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )
        writer_real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )
        step += 1

torch.save(generator.state_dict() , "weights.pt")
