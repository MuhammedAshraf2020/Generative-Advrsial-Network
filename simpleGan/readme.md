## Generative Advarsial Network MLP
![alt text](https://github.com/MuhammedAshraf2020/Generative-Advrsial-Network/blob/main/assets/gan_example_4.png)
generative advrsial network consist of two networks 
## Generator
![alt text](https://github.com/MuhammedAshraf2020/Generative-Advrsial-Network/blob/main/assets/nn%20(1).svg)
its MLP network in our example here whith [256 , 512 , 1024 , 784] in order 
with LeakyReLU and Tanh in the last layer to activate output between [-1 , 1]

## Discriminator
![alt text](https://github.com/MuhammedAshraf2020/Generative-Advrsial-Network/blob/main/assets/nn.svg)
Discriminator is ordinary classifier whith classify input [real , fake] with layers [1024 , 512 , 256 , 1] with dropout
