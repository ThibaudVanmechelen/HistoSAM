import timm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple

class HistEncoder(nn.Module):
    "Wrapper class for the different histopathology encoders."
    def __init__(self, model, img_res : Tuple[int, int], norm_mean : float, norm_std : float, type_ : str, device : str):
        """
        Constructor.

        Args:
            model: the encoder model.
            img_res (Tuple[int, int]): the resolution of the image for that particular encoder.
            norm_mean (float): the mean for normalizing with that encoder.
            norm_std (float): the std for normalizing with that encoder.
            type_ (str): name of that encoder.
            device (str): the device where the encoder is located.
        """
        super().__init__()

        self.model = model
        self.model_type = type_

        self.img_res = img_res
        self.norm_mean = torch.tensor(norm_mean).view(3, 1, 1).to(device)
        self.norm_std = torch.tensor(norm_std).view(3, 1, 1).to(device)

        self.device = device

        for param in self.model.parameters():
            param.requires_grad = False

    def preprocess(self, images : list):
        """
        Function to preprocess the image for the encoder.

        Args:
            images (list): list of images to preprocess.

        Returns:
            (Tensor): preprocessed images.
        """
        processed_images = []

        for img in images:
            if isinstance(img, np.ndarray):
                if img.shape[0] == 3 and img.shape[1] > 3 and img.shape[2] > 3: # case: img format = CxHxW numpy array
                    img_tensor = torch.from_numpy(img).float()

                elif img.shape[-1] == 3 and img.shape[0] > 3 and img.shape[1] > 3: # case: img format = HxWxC numpy array
                    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()

                else:
                    raise ValueError(f"Wrong image shape (np.ndarray): {img.shape}")
            
            elif isinstance(img, torch.Tensor): # case: img format = CxHxW tensor
                if img.shape[0] == 3 and img.shape[1] > 3 and img.shape[2] > 3:
                    img_tensor = img.float()
                else:
                    raise ValueError(f"Wrong image shape (Tensor): {img.shape}")
            
            else:
                raise TypeError(f"Wrong image type: {type(img)}")

            img_tensor = F.interpolate(img_tensor.unsqueeze(0), size = self.img_res, mode = "bilinear", align_corners = False).squeeze(0)
            img_tensor = img_tensor / 255.0
            img_tensor = (img_tensor - self.norm_mean) / self.norm_std

            processed_images.append(img_tensor)

        batch = torch.stack(processed_images, dim = 0).to(self.device)

        return batch

    def forward(self, x : torch.Tensor):
        """
        Forward function for the encoder.

        Args:
            x (torch.Tensor): input

        Returns:
            (Tensor): the prediction.
        """
        with torch.inference_mode():
            features = self.model.forward_features(x)

        return features[:, self.model.num_prefix_tokens:, :] # Shape: B x number_patches x encoder_dim


def get_histo_encoder(weight_path : str, type_ : str, device : str = 'cuda', verbose : bool = False):
    """
    Function to get an histo encoder model.

    Args:
        weight_path (str): path to the weights.
        type_ (str): type of the model, must be in ['uni', 'uni-2h', 'h-optimus-0'].
        device (str, optional): device where to put the encoder. Defaults to 'cuda'.
        verbose (bool, optional): whether to print additional info. Defaults to False.

    Returns:
        the model
    """
    assert type_ in ['uni', 'uni-2h', 'h-optimus-0'], f"Type must be 'uni','uni-2h or 'h-optimus-0', got: {type_}."

    if type_ == 'uni':
        model = timm.create_model("vit_large_patch16_224", img_size = 224, patch_size = 16, init_values = 1e-5, num_classes = 0, dynamic_img_size = True)

        mean_ = (0.485, 0.456, 0.406)
        std_ = (0.229, 0.224, 0.225)
        img_res = (224, 224)
        
    elif type_ == 'h-optimus-0':
        model = timm.create_model("hf-hub:bioptimus/H-optimus-0", pretrained = False, init_values = 1e-5, dynamic_img_size = False)

        mean_ = (0.707223, 0.578729, 0.703617)
        std_ = (0.211883, 0.230117, 0.177517)
        img_res = (224, 224)
    
    else:
        timm_kwargs = {
            'model_name': 'vit_giant_patch14_224',
            'img_size': 224, 
            'patch_size': 14, 
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5, 
            'embed_dim': 1536,
            'mlp_ratio': 2.66667*2,
            'num_classes': 0, 
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked, 
            'act_layer': torch.nn.SiLU, 
            'reg_tokens': 8, 
            'dynamic_img_size': True
        }

        model = timm.create_model(pretrained = False, **timm_kwargs)

        mean_ = (0.485, 0.456, 0.406)
        std_ = (0.229, 0.224, 0.225)
        img_res = (224, 224)

    model.to(device)
    model = load_weights(model, weight_path, device, verbose)

    return HistEncoder(model, img_res, mean_, std_, type_, device).eval()

def load_weights(model, weight_path : str, device : str = 'cuda', verbose : bool = False):
    """
    Function to load weights for the encoder.

    Args:
        model: the model
        weight_path (str): path to the weights.
        device (str, optional): device where to put the encoder. Defaults to 'cuda'.
        verbose (bool, optional): whether to print additional info. Defaults to False.

    Returns:
        the model with loaded weights.
    """
    state_dict = torch.load(weight_path, map_location = device)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict = True)

    if verbose:
        print("Missing keys:", missing_keys)        # This corresponds to layers in model that are missing in the checkpoint
        print("Unexpected keys:", unexpected_keys)  # This corresponds to layers in checkpoint that are missing in the model

    return model