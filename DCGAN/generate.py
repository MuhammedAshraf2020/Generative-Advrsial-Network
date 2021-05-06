
import argparse
from models import Generator
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

parser = argparse.ArgumentParser()
parser.add_argument("--path"                                         ,  help = "The path to weights of the generator model")
parser.add_argument("--nums"         , type = int   , default = 1    ,  help = "Number of examples that you want to generate")
parser.add_argument("--z_dim"        , type = int   , default = 64   ,  help = "latent dimentions")
parser.add_argument("--img_size"     , type = int   , default = 64   ,  help = "image size")
parser.add_argument("--num_channels" , type = int   , default = 1    ,  help = "number of number of channels")
args = parser.parse_args()

path    = args.path
numbers = args.nums
z_dim   = args.z_dim

os.makedirs("new" , exist_ok = True)
noise       = torch.randn(args.nums , z_dim , 1 , 1)
generator   = Generator(z_dim , args.num_channels , args.img_size)
generator.load_state_dict(torch.load(args.path))
generator.eval()
outputs = generator(noise)
outputs = outputs.permute(0 , 2 , 3 , 1)
outputs = outputs.cpu().detach().numpy()
outputs = (1/(2*2.25)) * outputs + 0.5
for idx , out in tqdm(enumerate(outputs)):
  plt.imsave("new/{}.jpg".format(idx) , out)
