

print("Loading Modules.....")
from utils import *
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
import NeuralNetwork as net
import numpy as np
print("Modules Loaded")

BATCH_SIZE = 9
ImageDim = (64,64,3)
zdim = 100


data_transform = transforms.Compose([transforms.Resize((ImageDim[0],ImageDim[1])),transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

generator = net.Generator(latent_dim=zdim).to(device)
generator.load_state_dict(torch.load("WorkingModels/Planes_generator.pth",weights_only=True))

plt.figure(figsize=(10,10))

generator.eval()

with torch.no_grad():
    z = torch.randn(25, zdim,1,1, device=device)
    generated = generator(z).detach().cpu()
    print(generated.shape)
    grid = make_grid(generated,nrow=5, normalize=True)
    plt.imshow(np.transpose(grid, (1, 2, 0)),interpolation='bilinear')
    plt.axis("off")
    plt.show()
