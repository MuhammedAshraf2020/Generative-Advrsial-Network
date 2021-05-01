
import numpy as np
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense , LeakyReLU , Flatten , Input
from keras.optimizers import Adam
from keras.datasets.mnist import load_data

def discriminator():
	model = Sequential([
		  Dense(128  , input_shape = (784,)) ,
      LeakyReLU(0.1),
		  Dense(1 , activation = "sigmoid")])
	opt = Adam(lr = 3e-4)
	model.compile(optimizer = opt , loss = "binary_crossentropy" , metrics = ["accuracy"])
	return model


def generator(z_dim):
	model = Sequential([
    Dense(128 , input_shape = (z_dim,) ) ,
    LeakyReLU(0.1),
		Dense(256) ,
    LeakyReLU(0.1),
		Dense(28 * 28  , activation = "tanh")
  ])
	return model


def GAN(discriminator , generator):
  model = Sequential()
  discriminator.trainable = False
  model.add(generator)
  model.add(discriminator)
  opt = Adam(lr = 3e-4)
  model.compile(optimizer = opt , loss = "binary_crossentropy" , metrics = ["accuracy"])
  return model

def load_real_samples():
	(trainX, _), (_, _) = load_data()
	X = trainX.astype('float32')
	X = X / 255.0
	return X


def generate_real_sample(batch_size , dataset):
  idxs = np.random.randint(0 , dataset.shape[0] , batch_size)
  X = dataset[idxs]
  X = X.reshape(batch_size , 28 * 28)
  y = np.ones((batch_size , 1))
  return X , y

generate_latent_points = lambda batch_size , z_dim : np.random.randn(batch_size , z_dim) 

def generate_fake_samples(latent_dim , generator , batch_size):
  points     = generate_latent_points(batch_size , latent_dim)
  fake_data  = generator.predict(points)
  fake_label = np.zeros((batch_size , 1))
  return fake_data , fake_label


def train(epochs , img_shape , z_dim , dataset , batch_size):
  disc = discriminator()
  gen  = generator(z_dim)
  gan  = GAN(disc , gen)
  for epoch in range(epochs):
    real_x , real_y = generate_real_sample(batch_size , dataset)
    fake_x , fake_y = generate_fake_samples(z_dim , gen , batch_size)
    batch = np.concatenate((real_x , fake_x))
    y     = np.concatenate((real_y , fake_y))
    d_loss, _  = disc.train_on_batch(batch , y) 
    latent_gan =  generate_latent_points(batch_size , z_dim)
    g_loss , _  = gan.train_on_batch(latent_gan , np.ones(batch_size))
    print(f"Epoch [{epoch}/{epochs}] \ Loss D: {d_loss:.4f}, loss G: {g_loss:.4f}")
  return disc , gen

epochs     = 100
img_shape  = 28 * 28
z_dim      = 100
dataset    = load_real_samples()
batch_size = 32

disc , gen = train(epochs , img_shape , z_dim , dataset , batch_size)
