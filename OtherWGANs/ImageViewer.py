print("Loading Modules.....")
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as fn
import matplotlib.pyplot as plt
import Networks as net
from torchvision.utils import make_grid

from PIL import Image

import numpy as np
print("Modules Loaded")

#-----------
#Paste this in MODEL_PATH to view output

#"WorkingModels/Planes_generator.pth"
#"WorkingModels/Paintings_generator.pth"
#"WorkingModels/Pokemon_generator.pth"
#"WorkingModels/Planes_with_small_batch_gen.pth"

#----------

MODEL_PATH = "WorkingModels/Pokemon_generator.pth"

BATCH_SIZE = 9
FEATURES_DISC = 64
FEATURES_GEN = 64
ZDIM = 100

IMAGE_HEIGHT,IMAGE_WIDTH,COLOR_CHANNELS = 64,64,3
LATENT_DIM = 100

gen = net.Generator(LATENT_DIM, COLOR_CHANNELS,FEATURES_GEN)
gen.load_state_dict(torch.load(MODEL_PATH,weights_only=True))

plt.figure(figsize=(10,10))
gen.eval()

#Generated Images
with torch.no_grad():
    z = torch.randn(49, ZDIM,1,1)
    generated = gen(z).detach().cpu()
    print(generated.shape)
    grid = make_grid(generated,nrow=7, normalize=True)
    img = grid
    tran = fn.ToPILImage()
    img = tran(img)
    img.save("Test.jpg")
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.axis("off")
    plt.show()


