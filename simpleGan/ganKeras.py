import numpy as np
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense , LeakyReLU , Flatten , Input , Dropout
from keras.optimizers import Adam
from keras.datasets.mnist import load_data

def discriminator():
  d = Sequential()
  d.add(Dense(1024, input_dim=784, activation=LeakyReLU(alpha=0.2)))
  d.add(Dropout(0.3))
  d.add(Dense(512, activation=LeakyReLU(alpha=0.2)))
  d.add(Dropout(0.3))
  d.add(Dense(256, activation=LeakyReLU(alpha=0.2)))
  d.add(Dropout(0.3))
  d.add(Dense(1, activation='sigmoid'))  # Values between 0 and 1
  d.compile(loss='binary_crossentropy', optimizer = "adam", metrics=['accuracy'])
  return d


def generator(z_dim):
  model = Sequential([
                      Dense(256, input_dim=z_dim, activation=LeakyReLU(alpha=0.2)) ,
                      Dense(512, activation=LeakyReLU(alpha=0.2)),
                      Dense(1024, activation=LeakyReLU(alpha=0.2)),
                      Dense(784 , activation = "sigmoid")])
  model.compile(loss='binary_crossentropy', optimizer = "adam", metrics=['accuracy'])
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
  X = X.reshape((X.shape[0] , -1))
  return X


"""def generate_real_sample(batch_size , dataset):
  idxs = np.random.randint(0 , dataset.shape[0] , batch_size)
  X = dataset[idxs]
  X = X.reshape(batch_size , 28 * 28)
  y = np.ones((batch_size))
  return X , y

def generate_latent_points( batch_size , z_dim):
	x_input = np.random.randn(z_dim * batch_size)
	x_input = x_input.reshape(batch_size, z_dim)
	return x_input

def generate_fake_samples(latent_dim , generator , batch_size):
  points     = generate_latent_points(batch_size , latent_dim)
  fake_data  = generator.predict(points)
  fake_label = np.zeros((batch_size))
  return fake_data , fake_label
"""

def train(epochs , img_shape , z_dim , dataset , batch_size):
  disc = discriminator()
  gen  = generator(z_dim)
  gan  = GAN(disc , gen)
  batches = len(dataset) // 32
  for epoch in range(epochs):
    for batch in range(batches):
      image_batch = dataset[np.random.randint(0, dataset.shape[0], size=batch_size)]
      noise = np.random.normal(0, 1, size=(batch_size, z_dim))
      generated_images = gen.predict(noise)
      X = np.concatenate((image_batch, generated_images))
      y = np.zeros(2*batch_size)
      y[:batch_size] = 0.9  
      disc.trainable = True
      d_loss = disc.train_on_batch(X, y)
      disc.trainable = True
      d_loss , _ = disc.train_on_batch(X, y)
      noise = np.random.normal(0, 1, size=(batch_size, z_dim))
      y2 = np.ones(batch_size)
      disc.trainable = False
      g_loss , _ = gan.train_on_batch(noise, y2)

    print(f"Epoch [{epoch}/{epochs}] \ Loss D: {d_loss:.4f}, loss G: {g_loss:.4f}")
  return disc , gen

epochs     = 10
img_shape  = 28 * 28
z_dim      = 100
dataset    = load_real_samples()
batch_size = 32

disc , gen = train(epochs , img_shape , z_dim , dataset , batch_size)
