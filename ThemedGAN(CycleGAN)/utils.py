import torch
from torch import nn
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image

class CGANDataset(Dataset):
    def __init__(self,root_A,root_B,transform=None):
        self.root_A = root_A
        self.root_B = root_B
        self.transform = transform

        self.A_images = os.listdir(root_A)
        self.B_images = os.listdir(root_B)
        self.length_dataset = (max(len(self.A_images),len(self.B_images)))

        self.A_len = len(self.A_images)
        self.B_len = len(self.B_images)

    def __len__(self):
        return self.length_dataset
    
    def __getitem__(self, index):
        A_image_path = self.A_images[index % self.A_len]
        B_image_path = self.B_images[index % self.B_len]

        A_path = os.path.join(self.root_A,A_image_path)
        B_path = os.path.join(self.root_B,B_image_path)

        A_image = ((Image.open(A_path).convert("RGB")))
        B_image = ((Image.open(B_path).convert("RGB")))

        if self.transform:
            
            A_image = self.transform(A_image)
            B_image = self.transform(B_image)

        return A_image,B_image

def gradient_penalty(critic,real,fake,device):
    BATCH_SIZE,C,H,W = real.shape
    epsilon = torch.rand((BATCH_SIZE,1,1,1)).repeat(1,C,H,W).to(device)
    interpolated_images = real * epsilon + fake*(1-epsilon)

    mixed_scores = critic(interpolated_images)

    gradient = torch.autograd.grad(
        inputs = interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True
    )[0]

    gradient = gradient.view(gradient.shape[0],-1).to(device)
    gradient_norm = gradient.norm(2,dim = 1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)

    return gradient_penalty

def initialize_weights(model:nn.Module):
    for m in model.modules():
        if isinstance(m,(nn.Conv2d,nn.ConstantPad2d,nn.BatchNorm2d)):
           nn.init.normal_(m.weight.data,0.0,0.02) 