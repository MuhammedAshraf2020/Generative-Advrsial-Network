import torch
import argparse
import torchvision
import torch.nn as nn 
import torch.optim as optim
import torchvision.datasets as datasets
from   models import Critic , Generator
from   torch.utils.data import DataLoader
import torchvision.transforms as transforms
from   torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser()
parser.add_argument("--Load"  , default = False , help = "Dose You Want To Load Old Weights ? ")

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


step = 0
Load = parser.parse_args().Load
n_dimension = 100
width_shape = 64
batch_size  = 128
n_channels  = 3
epochs_nums = 20
learning_rate = 5e-5
cliping_rate = 0.01
Writer_fake = SummaryWriter("logs/fake")
Writer_real = SummaryWriter("logs/real")
data_path  = "/content/img_align_celeba" # TODO
device = "cuda" if torch.cuda.is_available() else "cpu"


transform = transforms.Compose([
			transforms.Resize(width_shape),
			#transforms.CenterCrop(width_shape - 24),
			transforms.ToTensor(),
			transforms.Normalize((0.5 , 0.5 , 0.5) , (0.5 , 0.5 , 0.5))

		])

dataset = datasets.ImageFolder(root = data_path , transform = transform)
loader  = DataLoader(dataset , batch_size = batch_size , shuffle = True)

critic    = Critic(n_channels , width_shape).to(device)
generator = Generator(n_dimension , width_shape , n_channels).to(device)

generator.apply(weights_init)
critic.apply(weights_init)

opt_critic = optim.RMSprop(critic.parameters() , lr = learning_rate)
opt_gen    = optim.RMSprop(generator.parameters() , lr = learning_rate)


fixed_sample = torch.randn(batch_size , n_dimension , 1 , 1).to(device)

if Load == "True":
  print("Load Weights...")
  critic.load_state_dict(torch.load("critic_weights.pt"))
  generator.load_state_dict(torch.load("gen_weights.pt"))

for epoch in range(epochs_nums):
	for batch_idx , (real , _) in enumerate(loader):
		real = real.to(device)
		for itr in range(5):
			noise = torch.randn(batch_size , n_dimension , 1 , 1).to(device)
			fake  = generator(noise)
			critic_real = critic(real)
			critic_fake = critic(fake)
			loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake))
			critic.zero_grad()
			loss_critic.backward(retain_graph = True)
			opt_critic.step()

			for p in critic.parameters():
				p.data.clamp_(-cliping_rate , cliping_rate)

		gen_fake = critic(fake)
		loss_gen = - torch.mean(gen_fake)
		generator.zero_grad()
		loss_gen.backward()
		opt_gen.step()

		if batch_idx % 100 == 0:
			print(f"Epoch [{epoch} / {epochs_nums}] Batch [{batch_idx} / {len(loader)}]  Loss C: {-loss_critic:.4f} , Loss G: {loss_gen:.4f}")
			with torch.no_grad():
				fake = generator(fixed_sample).to(device)
				data = real.to(device)
				img_grid_fake = torchvision.utils.make_grid(fake[:32] , normalize = True)
				img_grid_real = torchvision.utils.make_grid(real[:32] , normalize = True)

				Writer_fake.add_image("Images Fake" , img_grid_fake  , global_step = step)
				Writer_real.add_image("Images Real" , img_grid_real  , global_step = step)
				step = step + 1 
				torch.save(generator.state_dict() , "gen_weights.pt")
				torch.save(critic.state_dict()    , "critic_weights.pt")