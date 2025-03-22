from utils import *

print("Loading Module.....")
import torch
from torch import nn
from torch.utils.data import DataLoader,random_split
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import NeuralNetwork as net
import numpy as np

device  = "cuda"

BATCH_SIZE = 1024
LEARNING_RATE_DISCRI = 0.0002
LEARNING_RATE_GENER = 0.0008
num_epochs = 2000
NUM_OF_DISCRI_STEP = 2
NUM_OF_GENER_STEP = 1

ImageDim = (64,64,3)
zdim = 100
mean_gen_loss = 0
mean_disc_loss = 0
info_iter = 1
show_iter = 30
beta1 = 0.5
beta2 = 0.999




data_transform = transforms.Compose([transforms.Resize((ImageDim[0],ImageDim[1])),transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

train_data = datasets.ImageFolder("T",transform=data_transform)

dataloader = DataLoader(dataset=train_data,shuffle=True,batch_size=BATCH_SIZE)

# Define the generator and discriminator
# Initialize generator and discriminator
generator = net.Generator(latent_dim=zdim).to(device)
discriminator = net.Discriminator().to(device)

generator.load_state_dict(torch.load("Models/generator.pth",weights_only=True))
discriminator.load_state_dict(torch.load("Models/discriminator.pth",weights_only=True))

# Loss function
adversarial_loss = nn.BCELoss()
# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters()\
                         , lr=LEARNING_RATE_GENER, betas=(beta1, beta2))
optimizer_D = torch.optim.Adam(discriminator.parameters()\
                         , lr=LEARNING_RATE_DISCRI, betas=(beta1, beta2))


for epoch in range(num_epochs):
    for i, batch in enumerate(dataloader):
       # Convert list to tensor
        real_images = batch[0].to(device) 
        # Adversarial ground truths
        valid = torch.ones(real_images.size(0), 1, device=device)
        fake = torch.zeros(real_images.size(0), 1, device=device)
        # Configure input
        real_images = real_images.to(device)
        for i in range(NUM_OF_DISCRI_STEP):
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            # Sample noise as generator input
            z = torch.randn(real_images.size(0), zdim,1,1, device=device)
            # Generate a batch of images
            fake_images = generator(z)

            # Measure discriminator's ability 
            # to classify real and fake images
            real_loss = adversarial_loss(discriminator\
                                         (real_images), valid)
            fake_loss = adversarial_loss(discriminator\
                                         (fake_images.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            # Backward pass and optimize
            d_loss.backward()
            optimizer_D.step()

        for i in range(NUM_OF_GENER_STEP):
            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()
            # Generate a batch of images
            gen_images = generator(z)
            # Adversarial loss
            g_loss = adversarial_loss(discriminator(gen_images), valid)
            # Backward pass and optimize
            g_loss.backward()
            optimizer_G.step()
        # ---------------------
        #  Progress Monitoring
        
    print(
                f"Epoch [{epoch+1}/{num_epochs}]"
                f"Discriminator Loss: {d_loss.item():.4f} "
                f"Generator Loss: {g_loss.item():.4f}"
    )
    # Save generated images for every epoch
    if (epoch + 1) % 20 == 0:
        with torch.no_grad():
            z = torch.randn(81, zdim,1,1, device=device)
            generated = generator(z).detach().cpu()
            grid = make_grid(generated,\
                                        nrow=9, normalize=True)
            plt.imshow(np.transpose(grid, (1, 2, 0)))
            plt.axis("off")
            plt.show()
            print("Do u want to save?")
            if(input() != "n"): 
                torch.save(generator.state_dict(),"Models/generator.pth")
                torch.save(discriminator.state_dict(),"Models/discriminator.pth")
                print("Model saved.")


        
        

