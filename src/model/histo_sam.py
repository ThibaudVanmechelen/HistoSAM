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

class HistoSAM(nn.Module):
    def __init__(self, 
             model_type : str,
             checkpoint_path : str,
             hist_encoder_type : str = None,
             hist_encoder_checkpoint_path : str = None,
             not_use_sam_encoder : bool = False,
             embedding_as_input : bool = False,
             up_sample_with_deconvolution : bool = False,
             freeze_sam_img_encoder : bool = True,
             freeze_prompt_encoder : bool = False,
             freeze_mask_decoder : bool = False,
             resolution : Tuple[int, int] = (1024, 1024),
             return_iou : bool = False,
             device : str = 'cuda'
             ):
        super().__init__()

        assert model_type in ['default', 'vit_b', 'vit_l', 'vit_h'], f"Model type must be 'default', 'vit_b', 'vit_l' or 'vit_h': received {model_type}."
        assert hist_encoder_type in ['uni', 'uni-2h', 'h-optimus-0'], f"Encoder type must be 'uni','uni-2h or 'h-optimus-0': received {hist_encoder_type}."

        assert (model_type is None) or (checkpoint_path is not None), "Must specify the path of SAM."

        assert (hist_encoder_checkpoint_path is None) or (hist_encoder_type is not None), "Must specify the type of the hist_encoder."
        assert (hist_encoder_type is None) or (hist_encoder_checkpoint_path is not None), "Must specify the path of the hist_encoder."

        self.model_type = model_type
        self.hist_encoder_type = hist_encoder_type

        if hist_encoder_type and not_use_sam_encoder == True:
            self.do_merge = False # if only one encoder, no need to merge
        
        else:
            self.do_merge = True

        if self.do_merge:
            self.model = fusion_sam_model_registry[model_type](checkpoint = checkpoint_path) # output shape: B x 64 x 64 x sam_embed_dim
        else:
            self.model = sam_model_registry[model_type](checkpoint = checkpoint_path)

        self.model.to(device)
        self.hist_encoder = get_histo_encoder(hist_encoder_checkpoint_path, hist_encoder_type, device, False)

        self.not_use_sam_encoder = not_use_sam_encoder
        self.embedding_as_input = embedding_as_input
        self.up_sample_with_deconvolution = up_sample_with_deconvolution 

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

        if self.up_sample_with_deconvolution:
            self.upsample = DeconvolutionUpSampler(nb_patch = self.hist_encoder.model.patch_embed.num_patches, 
                                                    embed_dim = self.hist_encoder.model.embed_dim, 
                                                    output_size = 64) # output shape: B x 64 x 64 x embed_dim
        
        else:
            self.upsample = InterpolationUpSampler(nb_patch = self.hist_encoder.model.patch_embed.num_patches,
                                                    embed_dim = self.hist_encoder.model.embed_dim, 
                                                    output_size = 64) # output shape: B x 64 x 64 x embed_dim

        self.upsample.to(device)

        out_chans = 256
        if self.model_type == 'vit_b':
            sam_embed_dim = 768

        elif self.model_type == 'vit_l':
            sam_embed_dim = 1024

        else:
            sam_embed_dim = 1280

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
        if self.embedding_as_input and self.do_merge == False: # case 1: only hist_encoder is used
            hist_image_embeddings = torch.stack([x["image"] for x in batched_input], dim = 0) # Shape: B x number_patches x encoder_dim because embeddings stored as number_patches x encoder_dim
            
            upsampled_image_embeddings = self.upsample(hist_image_embeddings) # Shape: B x 64 x 64 x encoder_dim
            image_embeddings = self.neck(upsampled_image_embeddings.permute(0, 3, 1, 2)) # Shape: B x 256 x 64 x 64

        elif self.embedding_as_input and self.do_merge == True: # case 2: both encoder are used
            input_images = [x["image"] for x in batched_input] # list of dict {"sam": tensor (64, 64, sam_embed_dim), "encoder": tensor (number_patches x encoder_dim)}

            sam_embeddings = torch.stack([d["sam"] for d in input_images], dim = 0) # Shape: B x 64 x 64 x sam_embed_dim
            hist_embeddings = torch.stack([d["encoder"] for d in input_images], dim = 0) # Shape: B x number_patches x encoder_dim

            upsampled_hist_embeddings = self.upsample(hist_embeddings) # Shape: B x 64 x 64 x encoder_dim
            concat_embeddings = torch.cat((sam_embeddings, upsampled_hist_embeddings), dim = -1) # Shape: B x 64 x 64 x (sam_embed_dim + encoder_dim)
            image_embeddings = self.neck(concat_embeddings.permute(0, 3, 1, 2)) # Shape: B x 256 x 64 x 64

        elif self.do_merge == True: # case 3: both encoder are used but without embeddings as input
            input_images_sam = torch.stack([self.model.preprocess(x["image"]) for x in batched_input], dim = 0) # Shape: BxCxHxW
            input_images_encoder = self.hist_encoder.preprocess([x["image"] for x in batched_input]) # Shape: BxCxHxW

            with torch.no_grad():
                image_embeddings = self.model.image_encoder(input_images_sam) # Shape: B x 64 x 64 x sam_embed_dim
                hist_image_embeddings = self.hist_encoder(input_images_encoder) # Shape: B x number_patches x encoder_dim

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
        if self.do_merge == True: # case 1: both encoder are used
            input_images_sam = torch.stack([self.model.preprocess(x["image"])], dim = 0) # Shape: 1xCxHxW
            input_images_encoder = self.hist_encoder.preprocess([x["image"]]) # Shape: 1xCxHxW

            with torch.no_grad():
                image_embeddings = self.model.image_encoder(input_images_sam).squeeze(0) # TODO check: Shape: 64 x 64 x sam_embed_dim
                hist_image_embeddings = self.hist_encoder(input_images_encoder).squeeze(0) # Shape: number_patches x encoder_dim

            return {"sam": image_embeddings, "encoder": hist_image_embeddings}

        else: # case 2: only hist_encoder is used
            input_images_encoder = self.hist_encoder.preprocess([x["image"]]) # Shape: 1xCxHxW

            with torch.no_grad():
                hist_image_embeddings = self.hist_encoder(input_images_encoder).squeeze(0) # Shape: number_patches x encoder_dim

            return hist_image_embeddings
    
    def compute_all_img_embeddings(self, dataset : SAMDataset):
        self.eval()

        with torch.no_grad():
            for i, (data, _, _) in tqdm(enumerate(dataset), total = len(dataset), desc = 'Saving img embeddings'):
                img_embedding = self.get_image_embedding(data)
                img_dir = os.path.dirname(dataset.images[i])

                img_save_path = os.path.join(img_dir, "img_embedding.pt")
                torch.save(img_embedding, img_save_path)