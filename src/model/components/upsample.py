import math

import torch.nn as nn
import torch.nn.functional as F


class InterpolationUpSampler(nn.Module):
    def __init__(self, nb_patch : int, embed_dim : int, output_size : int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.output_size = output_size
        self.input_size = int(math.sqrt(nb_patch)) 

        assert self.input_size in [14, 16], "input_size must be either 14 or 16 (nb_patches must be 196 or 256) !"

        self.upsample = nn.Upsample(size = (output_size, output_size), mode = "bilinear", align_corners = False)
        self.depthwise = nn.Conv2d(embed_dim, embed_dim, kernel_size = 3, stride = 1, padding = 1, groups = embed_dim) # per channel
        self.pointwise = nn.Conv2d(embed_dim, embed_dim, kernel_size = 1) # across channel

    def forward(self, x):
        B, _, embed_dim = x.shape

        H = self.input_size
        W = self.input_size

        x = x.permute(0, 2, 1).reshape(B, embed_dim, H, W)
        output = self.pointwise(self.depthwise(self.upsample(x)))

        return output.permute(0, 2, 3, 1) # Shape: B x 64 x 64 x embed_dim
    

class DeconvolutionUpSampler(nn.Module):
    def __init__(self, nb_patch : int, embed_dim : int, output_size : int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.output_size = output_size
        self.input_size = int(math.sqrt(nb_patch))

        assert self.input_size in [14, 16], "input_size must be either 14 or 16 (nb_patches must be 196 or 256) !"
        
        if self.input_size == 14:
            self.deconv1 = nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size = 3, stride = 1, padding = 0)  # 14 to 16
            self.deconv2 = nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size = 4, stride = 4, padding = 0)  # 16 to 64
        else:
            self.deconv1 = nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size = 4, stride = 4, padding = 0) # 16 to 64

    def forward(self, x):
        B, _, embed_dim = x.shape

        H = self.input_size
        W = self.input_size

        x = x.permute(0, 2, 1).reshape(B, embed_dim, H, W)

        if self.input_size == 14:
            output = self.deconv2(self.deconv1(x))
        else:
            output = self.deconv1(x)

        return output.permute(0, 2, 3, 1)  # Shape: B x 64 x 64 x embed_dim