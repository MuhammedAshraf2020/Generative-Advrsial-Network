import torch
import torchvision
import torch.nn as nn
from   math import log2
from   tqdm import tqdm
import torch.optim as optim
from   models import Generator, Critic
import torchvision.datasets as datasets
from   torch.utils.data import DataLoader
import torchvision.transforms as transforms
from   torch.utils.tensorboard import SummaryWriter

def Gradient_penality(critic , real , fake ,  alpha , step , device = "cpu" ):
  Batch_size , C , H , W = real.shape
  epsilon = torch.rand((Batch_size , 1 , 1 , 1)).repeat(1 , C , H , W).to(device)
  interpolated_images = real * epsilon + fake * (1 - epsilon)
  mixed_scores = critic(interpolated_images , alpha , step)
  gradient = torch.autograd.grad(inputs  = interpolated_images ,
                                 outputs = mixed_scores,
                                 grad_outputs = torch.ones_like(mixed_scores),
                                 create_graph = True ,
                                 retain_graph = True)[0]

  gradient = gradient.view(gradient.shape[0] , -1)
  gradient_norm = gradient.norm(2 , dim = 1)
  gradient_penality = torch.mean((gradient_norm - 1)**2)
  return gradient_penality 



def plot_to_tensorboard(writer, loss_critic, loss_gen, real, fake, tensorboard_step):
    writer.add_scalar("Loss Critic", loss_critic, global_step=tensorboard_step)
    with torch.no_grad():
      img_grid_real = torchvision.utils.make_grid(real[:8], normalize=True)
      img_grid_fake = torchvision.utils.make_grid(fake[:8], normalize=True)
      writer.add_image("Real", img_grid_real, global_step=tensorboard_step)
      writer.add_image("Fake", img_grid_fake, global_step=tensorboard_step)




def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location="cuda")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


lr = 1e-3
Z_DIM = 256  
LAMBDA_GP = 10
CHANNELS_IMG  = 3
SAVE_MODEL = True
LOAD_MODEL = True
IN_CHANNELS = 256 
START_TRAIN_AT_IMG_SIZE = 16
CHECKPOINT_GEN    = "generator.pth"
CHECKPOINT_CRITIC = "critic.pth"
BATCH_SIZES   = [32, 32, 32, 16, 16, 16, 16, 8, 4]
PROGRESSIVE_EPOCHS = [10] * len(BATCH_SIZES)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FIXED_NOISE = torch.randn(8, Z_DIM, 1, 1).to(DEVICE)


def get_loader(image_size , CHANNELS_IMG = 3  , data_path = '/content/celeba_hq/train'):
    
    transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize((0.5 , 0.5 , 0.5) ,
                                 (0.5 , 0.5 , 0.5)) ])

    batch_size = BATCH_SIZES[int(log2(image_size / 4))]
    dataset    = datasets.ImageFolder( root = data_path , transform = transform)
    loader     = DataLoader(dataset , batch_size = batch_size , shuffle = True)
    return loader , dataset

def train(critic , generator , loader , dataset , 
    step , alpha , opt_crit , opt_gen , tensorboard_step , 
    writer , scrit , sgen , device = "cuda" , Z_DIM = 256):

    loop = tqdm(loader , leave = True)
    for batch_idx , (real , _) in enumerate(loop):
        real = real.to(device)

        cur_bacth_size = real.shape[0]
        noise = torch.randn(cur_bacth_size , Z_DIM , 1 , 1).to(device)

        with torch.cuda.amp.autocast():
            fake  = gen(noise , alpha , step)
            crt_real = critic(real , alpha , step)
            crt_fake = critic(fake.detach() , alpha , step)
            gp = Gradient_penality(critic , real , fake , alpha , step , device )
            loss_critic = (-(torch.mean(crt_real) - torch.mean(crt_fake)) + 10 * gp )

        opt_crit.zero_grad()
        scrit.scale(loss_critic).backward(retain_graph = True)
        scrit.step(opt_crit)
        scrit.update()

        with torch.cuda.amp.autocast():
            gen_fake = critic(fake , alpha , step)
            loss_gen = - torch.mean(gen_fake)

        opt_gen.zero_grad()
        sgen.scale(loss_gen).backward()
        sgen.step(opt_gen)
        sgen.update()

        alpha += cur_bacth_size /((PROGRESSIVE_EPOCHS[step] * 0.5) * len(dataset))
        alpha  = min(alpha , 1)

        if batch_idx % 500 == 0:
            with torch.no_grad():
                fixed_fakes = gen(FIXED_NOISE, alpha, step) * 0.5 + 0.5
            plot_to_tensorboard(
                writer,
                loss_critic.item(),
                loss_gen.item(),
                real.detach(),
                fixed_fakes.detach(),
                tensorboard_step,
            )
            tensorboard_step += 1

        loop.set_postfix(
            gp = gp.item(),
            loss_critic=loss_critic.item(),
        )

    return tensorboard_step, alpha

gen   = Generator(Z_DIM , IN_CHANNELS , CHANNELS_IMG).to(DEVICE)
crt   = Critic(Z_DIM , IN_CHANNELS , CHANNELS_IMG).to(DEVICE)

opt_gen = optim.Adam(gen.parameters() , lr = lr , betas = (0.0 , 0.99))
opt_crt = optim.Adam(crt.parameters() , lr = lr , betas = (0.0 , 0.99))

scalar_crt  = torch.cuda.amp.GradScaler()
scalar_gen  = torch.cuda.amp.GradScaler()

writer = SummaryWriter("logs/gan")

if LOAD_MODEL == True:
    load_checkpoint(CHECKPOINT_GEN , gen , opt_gen , lr)
    load_checkpoint(CHECKPOINT_CRITIC , crt , opt_crt , lr)

gen.train()
crt.train()

tensorboard_step = 0

step = int(log2(START_TRAIN_AT_IMG_SIZE / 4))

for num_epochs in PROGRESSIVE_EPOCHS[step:]:
    alpha = 1e-5
    loader , dataset = get_loader(4 * 2**step)
    print(f"Current image size: {4 * 2 ** step}")
    
    for epoch in range(num_epochs):
        print(f"Epoch [{epoch+1}/{num_epochs}]") 
        tensorboard_step , alpha = train(crt , gen , loader , dataset , step , alpha , opt_crt , opt_gen , tensorboard_step , writer , scalar_crt , scalar_gen)
        if SAVE_MODEL == True:
            save_checkpoint(gen, opt_gen, filename = CHECKPOINT_GEN)
            save_checkpoint(crt, opt_crt, filename = CHECKPOINT_CRITIC)

    step = step + 1