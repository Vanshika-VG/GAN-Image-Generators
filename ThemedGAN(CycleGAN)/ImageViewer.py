

print("Loading Modules.....")
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
import NeuralNetworks as net
from torchvision.utils import make_grid
from utils import CGANDataset
import torchvision.transforms as fn

import numpy as np
print("Modules Loaded")

BATCH_SIZE = 9

FEATURES_DISC = 64

FEATURES_GEN = 64

IMAGE_HEIGHT,IMAGE_WIDTH,COLOR_CHANNELS = 128,128,3
LATENT_DIM = 100

data_transform = transforms.Compose([transforms.Resize((IMAGE_HEIGHT,IMAGE_WIDTH)),transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

train_data = CGANDataset(root_A="Training/trainA",root_B="Training/trainB",transform=data_transform)

dataloader = DataLoader(dataset=train_data,shuffle=True,batch_size=BATCH_SIZE)

gen_Z = net.Generator(img_channels=3,num_residuals=9)
gen_Z.load_state_dict(torch.load("Training_Models/gen_A.pth",weights_only=True))

gen_H = net.Generator(img_channels=3,num_residuals=9)
gen_H.load_state_dict(torch.load("Training_Models/gen_B.pth",weights_only=True))


plt.figure(figsize=(10,10))
gen_H.eval()

real_image,_ = next(iter(dataloader))
nrows = 4
ncols = 4
with torch.no_grad():
    images = []
    for i in range(nrows * ncols):
        test_data_features_A,_ = train_data[np.random.randint(0,len(train_data))]
        test_data_features_display = test_data_features_A * 0.5 + 0.5
        images.append(make_grid([test_data_features_display,gen_H(test_data_features_A) * 0.5 + 0.5],nrow=2))

    img_grid = make_grid(
                    images ,nrow=nrows
                )
    img = img_grid
    tran = fn.ToPILImage()
    img = tran(img)
    img.save("Test.jpg")
    plt.imshow(np.transpose(img_grid, (1, 2, 0))) 
    plt.axis(False)  
    plt.show()

    images = []
    for i in range(nrows * ncols):
        _,test_data_features = train_data[np.random.randint(0,len(train_data))]
        test_data_features_display = test_data_features * 0.5 + 0.5
        images.append(make_grid([test_data_features_display,gen_Z(test_data_features) * 0.5 + 0.5],nrow=2))

    img_grid = make_grid(
                    images ,nrow=nrows
                )
    
    img = img_grid
    tran = fn.ToPILImage()
    img = tran(img)
    img.save("Test_2.jpg")
    plt.imshow(np.transpose(img_grid, (1, 2, 0)))
    plt.axis(False)    
    plt.show()
        