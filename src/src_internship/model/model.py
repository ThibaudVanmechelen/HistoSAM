from argparse import ArgumentParser
from functools import partial
from typing import Any, Dict, List

import torch

#from segment_anything.build_sam import build_sam_vit_b, build_sam_vit_h, build_sam_vit_l
from segment_anything.modeling import (
    ImageEncoderViT,
    MaskDecoder,
    PromptEncoder,
    Sam,
    TwoWayTransformer,
)
from torch import nn


class TrainableSam(Sam):
    '''A trainable version of the Sam that allows backpropagation on the model.
    Only the forward method differs slightly'''
    def __init__(self, img_embeddings_as_input: bool = False, return_iou:bool=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.img_embeddings_as_input = img_embeddings_as_input
        self.return_iou = return_iou
    def forward(self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
        binary_mask_output: bool = False):
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.
          img_embeddings_as_input (bool): If True, the model expects the
            input to be image embeddings instead of raw images. Default: False.

        Returns:
            (torch.Tensor): Batched binary mask predictions,
              with shape BxCxHxW, where B is the number of input prompts,
              C is determined by multimask_output, and (H, W) is the
              original size of the image.
        """
        if self.img_embeddings_as_input:
            input_images = torch.stack([x["image"] for x in batched_input], dim=0).squeeze(0).squeeze(1)
            image_embeddings = input_images
        else:
            input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
            image_embeddings = self.image_encoder(input_images)
        outputs = []
        for i, (image_record, curr_embedding) in enumerate(zip(batched_input, image_embeddings)):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["original_size"],
                original_size=image_record["original_size"],
            )
            outputs.append(masks[0][torch.argmax(iou_predictions)])

        outputs = torch.stack(outputs, dim=0)
        if binary_mask_output:
            outputs = torch.where(outputs > self.mask_threshold, 1, 0).float()
        if self.return_iou:
            return outputs, torch.max(iou_predictions)
        return outputs

    def get_image_embeddings(self, batched_input: List[Dict[str, Any]]) -> torch.Tensor:
        '''Take batch_input and return the image embeddings.'''
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        return self.image_encoder(input_images)
    
    def get_nb_parameters(self, img_encoder=True):
        '''Function to get the number of parameters of the model.
        img_encoder: bool, If True, return the number of parameters of the image encoder else ignore it. Default: True
        Returns: int, number of parameters of the model'''
        if img_encoder:
            return sum(p.numel() for p in self.parameters())
        else:
            return sum(p.numel() for p in self.mask_decoder.parameters()) + sum(p.numel() for p in self.prompt_encoder.parameters())

def load_model(model_path:str, model_type:str='vit_b', img_embeddings_as_input:bool=False, return_iou:bool=False,) -> TrainableSam:
    '''Function to load a trained model. The function returns a torch.nn.Module model.
    model_path: str, path to the model
    model_type: str in ["vit_b", "vit_h", "vit_l"], type of the model to load. Default: "vit_b"'''
    if model_type == 'vit_b':
        model = build_sam_vit_b(model_path)
    elif model_type == 'vit_h':
        model = build_sam_vit_h(model_path)
    elif model_type == 'vit_l':
        model = build_sam_vit_l(model_path)
    else:
        raise ValueError(f'Model {model_type} not supported')
    model.requires_grad_(True)
    model.img_embeddings_as_input = img_embeddings_as_input
    model.return_iou = return_iou
    return model

def build_sam_vit_h(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
    )


build_sam = build_sam_vit_h


def build_sam_vit_l(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
    )


def build_sam_vit_b(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )


sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}


def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = TrainableSam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        '''for key in list(state_dict.keys()):
            if key.startswith('module.'):
                state_dict[key[7:]] = state_dict.pop(key)'''
        sam.load_state_dict(state_dict)
    return sam




if __name__ == '__main__':
    model = load_model('../../checkpoints/sam_vit_b_01ec64.pth', 'vit_b')
    print(model)