import torch
from torch import nn

class Discriminator(nn.Module):
    def __init__(self,in_channels = 3,features = [64,128,256,512],*args, **kwargs):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels,
                features[0],
                4,2,1,
                padding_mode="reflect"
            ),
            nn.LeakyReLU(0.2)
        )
        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(Block(in_channels,feature,stride = 1 if features == features[-1] else 2))
            in_channels=feature
        layers.append(nn.Conv2d(in_channels,1,kernel_size=4,stride=1,padding=1,padding_mode="reflect") 
                      )
        self.model = nn.Sequential(*layers)

    def forward(self,x):
        x = self.initial(x)
        return torch.sigmoid(self.model(x))
    
class Critic(nn.Module):
    def __init__(self,in_channels = 3,features = [64,128,256,512],*args, **kwargs):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels,
                features[0],
                4,2,1,
                padding_mode="reflect"
            ),
            nn.LeakyReLU(0.2)
        )
        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(Block(in_channels,feature,stride = 1 if features == features[-1] else 2))
            in_channels=feature
        layers.append(nn.Conv2d(in_channels,1,kernel_size=4,stride=1,padding=1,padding_mode="reflect") 
                      )
        self.model = nn.Sequential(*layers)

    def forward(self,x):
        x = self.initial(x)
        return self.model(x)
    
class Generator(nn.Module):
    def __init__(self,img_channels,num_features=64,num_residuals=9, *args, **kwargs):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(img_channels,64,7,1,3,padding_mode="reflect"),
            nn.ReLU(inplace=True)
        )
        self.down_blocks = nn.ModuleList(
            [
                ConvBlock(num_features,num_features*2,kernel_size=3,stride=2,padding=1),
                ConvBlock(num_features*2,num_features*4,kernel_size=3,stride=2,padding=1)
            ]
        )

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_features*4) for _ in range(num_residuals)]
        )

        self.up_blocks = nn.ModuleList(
            [
                ConvBlock(num_features*4,num_features*2,down=False,kernel_size=3,stride=2,padding=1,output_padding=1),
                ConvBlock(num_features*2,num_features,down=False,kernel_size=3,stride=2,padding=1,output_padding=1)
            ]
        )

        self.last = nn.Conv2d(num_features*1, img_channels,kernel_size=7,stride=1,padding=3,padding_mode="reflect" )

    def forward(self,x):
        x = self.initial(x)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.residual_blocks(x)
        for layer in self.up_blocks:
            x = layer(x)
        return torch.tanh(self.last(x))


class Block(nn.Module):
    def __init__(self, in_channels,out_channels,stride,*args, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      4,
                      stride,
                      1,
                      padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self,x):
        return self.conv(x)
    
class ConvBlock(nn.Module):
    def __init__(self,in_channels, out_channels,down=True,use_act=True ,*args, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,padding_mode="reflect",**kwargs)
            if down
            else nn.ConvTranspose2d(in_channels,out_channels,**kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity()
        )

    def forward(self,x):
        return self.conv(x)
    
class ResidualBlock(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels,channels,kernel_size=3,padding=1),
            ConvBlock(channels,channels,use_act=False,kernel_size=3,padding=1)           
        )

    def forward(self,x):
        return x + self.block(x)
