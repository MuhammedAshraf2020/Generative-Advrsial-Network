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
parser.add_argument("--batch_size"    , type = int   ,  default = 64     , help = "Latent numbers ")
parser.add_argument("--lr"            , type = float ,  default = 5e-5   , help = "Learning Rate" )
parser.add_argument("--z_dim"         , type = int   ,  default = 100    , help = "Latent variable")
parser.add_argument("--data_path"                                        , help = "Path of data")
parser.add_argument("--write_path"    ,                 default = "logs" , )
parser.add_argument("--critic_iter"   , type = int   ,  default = 5      , help = "")
parser.add_argument("--weight_clip"   , type = float ,  default = 0.01   , help = "")
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
weight_clip = args.weight_clip
critic_iter = args.critic_iter


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

critic = Discriminator(num_channels , img_size).to(device)
generator     = Generator(z_dim ,num_channels , img_size ).to(device)

generator.apply(weights_init)
critic.apply(weights_init)

fixed_noise = torch.randn((batch_size, z_dim , 1 , 1)).to(device)

transform     = transforms.Compose([
    transforms.Resize(img_size) ,
    transforms.CenterCrop(img_size),
    transforms.ToTensor() ,
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

opt_critic = optim.RMSprop(critic.parameters() , lr = lr ) 
opt_gen  = optim.RMSprop(generator.parameters() , lr = lr )

dataset = datasets.ImageFolder(root = args.data_path , transform = transform)
loader  = DataLoader(dataset, batch_size = 32 , shuffle=True)

#Train Model
generator.train()
critic.train()

for epoch in range(epochs):
  for batch_idx , (real , _ ) in enumerate(loader):

    #create real , fake data
    real  = real.to(device)

    for _ in range(critic_iter):
      noise = torch.randn(batch_size , z_dim , 1 , 1).to(device)
      fake  = generator(noise)
      critic_real = critic(real).view(-1)
      critic_fake = critic(fake).view(-1)
      loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake))
      critic.zero_grad()
      loss_critic.backward(retain_graph=True)
      opt_critic.step()

      for p in critic.parameters():
        p.data.clamp_(-weight_clip , weight_clip)

    #Generator trainig
    gen_fake = critic(fake).view(-1)
    loss_gen = -torch.mean(gen_fake)
    generator.zero_grad()
    loss_gen.backward()
    opt_gen.step()
    
    if batch_idx %100 == 0:
      print(f"Epoch [{epoch}/{epochs}] Batch {batch_idx}/{len(loader)} \ Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}")
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