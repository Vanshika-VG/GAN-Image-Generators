import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self,zdim,channels_img,features_g,*args, **kwargs):
        super(Generator,self).__init__(*args, **kwargs)
        self.gen = nn.Sequential(
            self.block(zdim,features_g*16,4,1,0),
            self.block(features_g*16,features_g*8,4,2,1),
            self.block(features_g*8,features_g*4,4,2,1),
            self.block(features_g*4,features_g*2,4,2,1),
            nn.ConvTranspose2d(
                features_g*2,
                channels_img,
                4,
                2,
                1,
            ),
            nn.Tanh()
        )
        
    def block(self,in_channels,out_channels,kernel_size,stride,padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False
            ),
            nn.InstanceNorm2d(out_channels,affine=True),
            nn.ReLU()
        )


    def forward(self,x:torch.Tensor) -> torch.Tensor:
        return self.gen(x)
    
class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d,*args, **kwargs):
        super(Discriminator,self).__init__(*args, **kwargs)
        self.disc = nn.Sequential(
            nn.Conv2d(channels_img,features_d,kernel_size=4,stride=2,padding=1),
            nn.LeakyReLU(0.2),
            self.block(features_d,features_d*2,4,2,1),
            self.block(features_d*2,features_d*4,4,2,1),
            self.block(features_d*4,features_d*8,4,2,1),
            nn.Conv2d(features_d*8,1,kernel_size=4,stride=2,padding=0),
            
        )

    def block(self,in_channels,out_channels,kernel_size,stride,padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False
            ),
            nn.InstanceNorm2d(out_channels,affine=True),
            nn.LeakyReLU(0.2)
        )

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        return self.disc(x)
    
def initialize_weights(model:nn.Module):
    for m in model.modules():
        if isinstance(m,(nn.Conv2d,nn.ConstantPad2d,nn.BatchNorm2d)):
           nn.init.normal_(m.weight.data,0.0,0.02) 

