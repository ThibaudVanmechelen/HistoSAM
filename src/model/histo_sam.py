import os
import tqdm
import torch
import torch.nn as nn

from typing import Tuple, Any, Dict, List
from segment_anything import sam_model_registry
from segment_anything.modeling.common import LayerNorm2d
from dataset_processing.dataset import SAMDataset

from components.histo_encoder import get_histo_encoder
from components.custom_fusion_sam import fusion_sam_model_registry
from components.upsample import DeconvolutionUpSampler, InterpolationUpSampler
from components.cross_attention_fusion import CrossAttentionFusionModule
from components.attention_refinement_module import AttentionRefinementModule

class HistoSAM(nn.Module):
    def __init__(self, 
             model_type : str,
             checkpoint_path : str,
             hist_encoder_type : str = None,
             hist_encoder_checkpoint_path : str = None,
             not_use_sam_encoder : bool = False,
             embedding_as_input : bool = False,
             up_sample_with_deconvolution : bool = False,
             fuse_with_attention : bool = False,
             refine_with_attention : bool = False,
             sam_weights_for_refinement : str = None,
             freeze_sam_img_encoder : bool = True,
             freeze_prompt_encoder : bool = False,
             freeze_mask_decoder : bool = False,
             resolution : Tuple[int, int] = (1024, 1024),
             return_iou : bool = False,
             device : str = 'cuda'
             ):
        """
        Constructor of HistoSAM.

        Args:
            model_type (str): model type for SAM.
            checkpoint_path (str): path to the SAM checkpoint.
            hist_encoder_type (str, optional): Encoder type of the histoencoder (if used). Defaults to None.
            hist_encoder_checkpoint_path (str, optional): path to the checkpoint of the histoencoder. Defaults to None.
            not_use_sam_encoder (bool, optional): whether to use sam image encoder. Defaults to False.
            embedding_as_input (bool, optional): whether the inputs are embeddings instead of images. Defaults to False.
            up_sample_with_deconvolution (bool, optional): whether to upsample with deconv instead of efficient upsampling. Defaults to False.
            fuse_with_attention (bool, optional): whether to merge embeddings of the 2 encoders with attention. Defaults to False.
            refine_with_attention (bool, optional): whether to refine the mask with attention (cannot be true with fuse_with_attention). Defaults to False.
            sam_weights_for_refinement (str, optional): path to the weights if refinement is performed. Defaults to None.
            freeze_sam_img_encoder (bool, optional): whether to freeze the sam img encoder. Defaults to True.
            freeze_prompt_encoder (bool, optional): whether to freeze the prompt encoder. Defaults to False.
            freeze_mask_decoder (bool, optional): whether to freeze the mask decoder. Defaults to False.
            resolution (Tuple[int, int], optional): resolution of the image. Defaults to (1024, 1024).
            return_iou (bool, optional): whether to return the iou with the output mask. Defaults to False.
            device (str, optional): device where to put the model Defaults to 'cuda'.
        """
        super().__init__()

        assert model_type in ['default', 'vit_b', 'vit_l', 'vit_h'], f"Model type must be 'default', 'vit_b', 'vit_l' or 'vit_h': received {model_type}."
        assert hist_encoder_type in ['uni', 'uni-2h', 'h-optimus-0'], f"Encoder type must be 'uni','uni-2h or 'h-optimus-0': received {hist_encoder_type}."

        assert (model_type is None) or (checkpoint_path is not None), "Must specify the path of SAM."

        assert (hist_encoder_checkpoint_path is None) or (hist_encoder_type is not None), "Must specify the type of the hist_encoder."
        assert (hist_encoder_type is None) or (hist_encoder_checkpoint_path is not None), "Must specify the path of the hist_encoder."

        assert (fuse_with_attention is False) or (refine_with_attention is False), "Impossible to use the hist encoder to both fuse and refine."

        self.model_type = model_type
        self.hist_encoder_type = hist_encoder_type

        if hist_encoder_type and not_use_sam_encoder == True:
            self.do_merge = False # if only one encoder, no need to merge
        
        else:
            self.do_merge = True

        if not self.do_merge:
            assert not fuse_with_attention, "fuse_with_attention must be False if only one encoder."
            assert not refine_with_attention, "refine_with_attention must be False if only one encoder."

        if self.do_merge and not refine_with_attention:
            self.model = fusion_sam_model_registry[model_type](checkpoint = checkpoint_path) # output shape: B x 64 x 64 x sam_embed_dim
        else:
            self.model = sam_model_registry[model_type](checkpoint = checkpoint_path)

            if refine_with_attention and sam_weights_for_refinement:
                print(f"Loading sam model weights from {sam_weights_for_refinement}")

                state_dict = torch.load(sam_weights_for_refinement)
                self.model.load_state_dict(state_dict, strict = True)

        self.model.to(device)
        self.hist_encoder = get_histo_encoder(hist_encoder_checkpoint_path, hist_encoder_type, device, False)

        self.not_use_sam_encoder = not_use_sam_encoder
        self.embedding_as_input = embedding_as_input
        self.up_sample_with_deconvolution = up_sample_with_deconvolution
        self.fuse_with_attention = fuse_with_attention
        self.refine_with_attention = refine_with_attention

        self.resolution = resolution
        self.return_iou = return_iou

        self.device = device

        if freeze_sam_img_encoder:
            for param in self.model.image_encoder.parameters():
                param.requires_grad = False

        if freeze_prompt_encoder:
            for param in self.model.prompt_encoder.parameters():
                param.requires_grad = False

        if freeze_mask_decoder:
            for param in self.model.mask_decoder.parameters():
                param.requires_grad = False

        if self.model_type == 'vit_b':
            sam_embed_dim = 768

        elif self.model_type == 'vit_l':
            sam_embed_dim = 1024

        else:
            sam_embed_dim = 1280

        if self.fuse_with_attention:
            self.fuse_module = CrossAttentionFusionModule(
                query_dim = sam_embed_dim,
                context_dim = self.hist_encoder.model.embed_dim                                    
            )

            self.fuse_module.to(device)

        elif self.refine_with_attention:
            self.refinement_module = AttentionRefinementModule(
                mask_channels = 3,
                hist_dim = self.hist_encoder.model.embed_dim
            )

            self.refinement_module.to(device)

        elif self.up_sample_with_deconvolution:
            self.upsample = DeconvolutionUpSampler(
                nb_patch = self.hist_encoder.model.patch_embed.num_patches, 
                embed_dim = self.hist_encoder.model.embed_dim, 
                output_size = 64
            ) # output shape: B x 64 x 64 x embed_dim

            self.upsample.to(device)
        
        else:
            self.upsample = InterpolationUpSampler(
                nb_patch = self.hist_encoder.model.patch_embed.num_patches,
                embed_dim = self.hist_encoder.model.embed_dim, 
                output_size = 64
            ) # output shape: B x 64 x 64 x embed_dim

            self.upsample.to(device)

        if self.refine_with_attention is False and self.fuse_with_attention is False:
            out_chans = 256
            if self.do_merge:
                total_embed_dim = sam_embed_dim + self.hist_encoder.model.embed_dim
            else:
                total_embed_dim = self.hist_encoder.model.embed_dim

            self.neck = nn.Sequential(
                nn.Conv2d(
                    total_embed_dim,
                    out_chans,
                    kernel_size = 1,
                    bias = False,
                ),
                LayerNorm2d(out_chans),
                nn.Conv2d(
                    out_chans,
                    out_chans,
                    kernel_size = 3,
                    padding = 1,
                    bias = False,
                ),
                LayerNorm2d(out_chans),
            )

            self.neck.to(device)

    def forward(self, batched_input : List[Dict[str, Any]], multimask_output : bool = True, binary_mask_output: bool = False):
        """
        Forward method of HistoSAM.

        Args:
            batched_input (List[Dict[str, Any]]): the batched inputs (same structure as for SAM), see model.py.
            multimask_output (bool, optional): whether to output multiple masks. Defaults to True.
            binary_mask_output (bool, optional): whether to binarize the masks. Defaults to False.

        Returns:
            the predicted mask as Tensor (and potentially the iou, if return_iou is True)
        """
        if self.embedding_as_input and self.do_merge == False: # case 1: only hist_encoder is used
            hist_image_embeddings = torch.stack([x["image"] for x in batched_input], dim = 0) # Shape: B x number_patches x encoder_dim because embeddings stored as number_patches x encoder_dim
            
            upsampled_image_embeddings = self.upsample(hist_image_embeddings) # Shape: B x 64 x 64 x encoder_dim
            image_embeddings = self.neck(upsampled_image_embeddings.permute(0, 3, 1, 2)) # Shape: B x 256 x 64 x 64

        elif self.embedding_as_input and self.do_merge == True: # case 2: both encoder are used
            input_images = [x["image"] for x in batched_input] # list of dict {"sam": tensor (64, 64, sam_embed_dim), "encoder": tensor (number_patches x encoder_dim)}

            sam_embeddings = torch.stack([d["sam"] for d in input_images], dim = 0) # Shape: B x 64 x 64 x sam_embed_dim
            hist_image_embeddings = torch.stack([d["encoder"] for d in input_images], dim = 0) # Shape: B x number_patches x encoder_dim

            if self.refine_with_attention:
                image_embeddings = sam_embeddings # Shape: B x 256 x 64 x 64 because used original SAM here with its neck
                encoder_embeddings = hist_image_embeddings # Shape: B x number_patches x encoder_dim

            elif self.fuse_with_attention:
                B, H, W, sam_dim = sam_embeddings.shape
                image_embeddings = self.fuse_module(query = sam_embeddings.reshape(B, H * W, sam_dim), context = hist_image_embeddings) # Shape: B x 256 x 64 x 64

            else:
                upsampled_hist_embeddings = self.upsample(hist_image_embeddings) # Shape: B x 64 x 64 x encoder_dim
                concat_embeddings = torch.cat((sam_embeddings, upsampled_hist_embeddings), dim = -1) # Shape: B x 64 x 64 x (sam_embed_dim + encoder_dim)
                image_embeddings = self.neck(concat_embeddings.permute(0, 3, 1, 2)) # Shape: B x 256 x 64 x 64

        elif self.do_merge == True: # case 3: both encoder are used but without embeddings as input
            input_images_sam = torch.stack([self.model.preprocess(x["image"]) for x in batched_input], dim = 0) # Shape: BxCxHxW
            input_images_encoder = self.hist_encoder.preprocess([x["image"] for x in batched_input]) # Shape: BxCxHxW

            with torch.no_grad():
                image_embeddings = self.model.image_encoder(input_images_sam) # Shape: B x 64 x 64 x sam_embed_dim
                hist_image_embeddings = self.hist_encoder(input_images_encoder) # Shape: B x number_patches x encoder_dim

            if self.refine_with_attention:
                encoder_embeddings = hist_image_embeddings # Shape: B x number_patches x encoder_dim
                # image_embeddings has shape B x 256 x 64 x 64

            elif self.fuse_with_attention:
                B, H, W, sam_dim = image_embeddings.shape
                image_embeddings = self.fuse_module(query = image_embeddings.reshape(B, H * W, sam_dim), context = hist_image_embeddings) # Shape: B x 256 x 64 x 64

            else:
                upsampled_hist_embeddings = self.upsample(hist_image_embeddings) # Shape: B x 64 x 64 x encoder_dim
                concat_embeddings = torch.cat((image_embeddings, upsampled_hist_embeddings), dim = -1) # Shape: B x 64 x 64 x (sam_embed_dim + encoder_dim)
                image_embeddings = self.neck(concat_embeddings.permute(0, 3, 1, 2)) # Shape: B x 256 x 64 x 64

        else: # case 4: only hist_encoder is used but without embeddings as input
            input_images_encoder = self.hist_encoder.preprocess([x["image"] for x in batched_input]) # Shape: BxCxHxW

            with torch.no_grad():
                hist_image_embeddings = self.hist_encoder(input_images_encoder) # Shape: B x number_patches x encoder_dim

            upsampled_image_embeddings = self.upsample(hist_image_embeddings) # Shape: B x 64 x 64 x encoder_dim
            image_embeddings = self.neck(upsampled_image_embeddings.permute(0, 3, 1, 2)) # Shape: B x 256 x 64 x 64

        outputs = []
        iou_scores = []

        for i, (image_record, curr_embedding) in enumerate(zip(batched_input, image_embeddings)):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])

            else:
                points = None

            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points = points,
                boxes = image_record.get("boxes", None),
                masks = image_record.get("mask_inputs", None),
            )

            low_res_masks, iou_predictions = self.model.mask_decoder(
                image_embeddings = curr_embedding.unsqueeze(0),
                image_pe = self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings = sparse_embeddings,
                dense_prompt_embeddings = dense_embeddings,
                multimask_output = multimask_output,
            )

            if self.refine_with_attention:
                low_res_masks = self.refinement_module(low_res_masks, encoder_embeddings[i].unsqueeze(0))

            masks = self.model.postprocess_masks(
                low_res_masks,
                input_size = image_record["original_size"],
                original_size = image_record["original_size"],
            )

            iou_predictions = iou_predictions.squeeze(0) 
            best_mask_idx = torch.argmax(iou_predictions)

            outputs.append(masks[0][best_mask_idx]) # outputs the mask with highest iou
            iou_scores.append(iou_predictions[best_mask_idx])

        outputs = torch.stack(outputs, dim = 0)

        if binary_mask_output:
            outputs = torch.where(outputs > self.model.mask_threshold, 1, 0).float() # make the mask binary (in the original SAM, always binary)

        if self.return_iou:
            return outputs, torch.stack(iou_scores, dim = 0)
        
        return outputs
    
    def get_image_embedding(self, x : dict):
        """
        Function to get the embeddings for one image.

        Args:
            x (dict): the dict containing the input image, same shape as input in model.py

        Returns:
            (Tensor): the embedding
        """
        if self.do_merge == True: # case 1: both encoder are used
            input_images_sam = torch.stack([self.model.preprocess(x["image"])], dim = 0) # Shape: 1xCxHxW
            input_images_encoder = self.hist_encoder.preprocess([x["image"]]) # Shape: 1xCxHxW

            with torch.no_grad():
                image_embeddings = self.model.image_encoder(input_images_sam).squeeze(0) # Shape: 64 x 64 x sam_embed_dim
                hist_image_embeddings = self.hist_encoder(input_images_encoder).squeeze(0) # Shape: number_patches x encoder_dim

            return {"sam": image_embeddings, "encoder": hist_image_embeddings}

        else: # case 2: only hist_encoder is used
            input_images_encoder = self.hist_encoder.preprocess([x["image"]]) # Shape: 1xCxHxW

            with torch.no_grad():
                hist_image_embeddings = self.hist_encoder(input_images_encoder).squeeze(0) # Shape: number_patches x encoder_dim

            return hist_image_embeddings
    
    def compute_all_img_embeddings(self, dataset : SAMDataset):
        """
        Function to compute all the embeddings for the images in a dataset.

        Args:
            dataset (SAMDataset): the dataset object.
        """
        self.eval()

        with torch.no_grad():
            for j, (data, _, _) in tqdm(enumerate(dataset), total = len(dataset), desc = 'Saving img embeddings'):
                img_embedding = self.get_image_embedding(data)
                img_dir = os.path.dirname(dataset.images[j])

                img_save_path = os.path.join(img_dir, "img_embedding.pt")
                torch.save(img_embedding, img_save_path)
