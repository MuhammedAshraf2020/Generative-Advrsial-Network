import torch
import torch.optim as optim
import torchvision.datasets as datasets
from torch.utils.data import  DataLoader
import torchvision.transforms as transforms
from models import Discriminator , Generator
from torch.utils.tensorboard import SummaryWriter 
from torchvision.utils import make_grid


#parameters
BATCH_SIZE  = 64
LR          = 2e-4
IMG_SIZE    = 28 * 28
EPOCHS      = 50
Z_DIM       = 100
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
WRITER_FAKE = SummaryWriter("logs/fake")
WRITER_REAL = SummaryWriter("logs/real")
STEPS       = 0
LOAD_MODELS = True

#transformers which we need to apply on the image
transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.5,) , (0.5,))
	])


#load Data
dataset = datasets.MNIST(root = "dataset" , transform = transform)
loader  = DataLoader(dataset , shuffle = True , batch_size = BATCH_SIZE)

#Prepare model
disc = Discriminator(IMG_SIZE).to(DEVICE)
gen  = Generator(Z_DIM , IMG_SIZE).to(DEVICE)
disc_opt  = optim.Adam(disc.parameters() , lr = LR)
gen_opt   = optim.Adam(gen.parameters()  , lr = LR)
criterion = nn.BCELoss()
NOISE = torch.randn(BATCH_SIZE , Z_DIM).to(DEVICE)

if LOAD_MODELS:
  gen.load_state_dict(torch.load("gen_weights.pt"))
  disc.load_state_dict(torch.load("disc_weights.pt"))

for epoch in range(EPOCHS):
  for batch_idx , (x , _) in enumerate(loader):
    real  = x.view(-1 , 28*28).to(DEVICE)
    noise = torch.randn(BATCH_SIZE , Z_DIM).to(DEVICE)
    fake  = gen(noise)
    #print(real.shape)
    disc_fake   = disc(fake).view(-1)
    disc_real   = disc(real).view(-1)

    real_label  = torch.ones_like(disc_real)
    fake_label  = torch.zeros_like(disc_fake)

    loss_real   = criterion(disc_real , real_label)
    loss_fake   = criterion(disc_fake , fake_label)
    D_loss      = (loss_fake + loss_real) / 2
    disc.zero_grad()
    D_loss.backward(retain_graph = True)
    disc_opt.step()
    
    disc_fake = disc(fake).view(-1)
    loss_G = criterion(disc_fake , torch.ones_like(disc_fake))

    gen.zero_grad()
    loss_G.backward()
    gen_opt.step()
  print("epoch[{}:{}] , D_LOSS = {:.4f} , G_LOSS = {:.4f}".format(epoch , EPOCHS , D_loss , loss_G))
  with torch.no_grad():
    fake = gen(NOISE).reshape(-1, 1, 28, 28)
    data = real.reshape(-1, 1, 28, 28)
    img_grid_fake = make_grid(fake, normalize=True)
    img_grid_real = make_grid(data, normalize=True)
    WRITER_FAKE.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=STEPS
                )
    WRITER_REAL.add_image(
                    "Mnist Real Images", img_grid_real, global_step=STEPS
                )
    STEPS += 1
    torch.save(gen.state_dict()  , "gen_weights.pt")
    torch.save(disc.state_dict() , "disc_weights.pt")
