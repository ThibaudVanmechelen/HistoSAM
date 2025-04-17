"""File to define the attention module that is used to refine the mask at the output of the HistoSAM model."""

import torch.nn as nn

class AttentionRefinementModule(nn.Module):
    def __init__(self, mask_channels, hist_dim, hidden_dim = 256, num_heads = 8):
        """
        Constructor for the module.

        Args:
            mask_channels (int): the number of channels of the masks given as input.
            hist_dim (int): _the dimension of the embedding provided by the domain encoder.
            hidden_dim (int, optional): the number of dimensions to output. Defaults to 256 (to be compatible with the original size used in SAM).
            num_heads (int, optional): the number of head to be used for the MultiheadAttention. Defaults to 8.
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        self.query_conv = nn.Conv2d(mask_channels, hidden_dim, kernel_size = 1) # B x 3 x 256 x 256 -> B x 256 x 256 x 256
        self.hist_layer = nn.Linear(hist_dim, hidden_dim) # B x number_patches x encoder_dim -> B x number_patches x hidden_dim
        self.attention = nn.MultiheadAttention(embed_dim = hidden_dim, num_heads = num_heads, batch_first = True)

        self.refinement_layer = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size = 3, padding = 1), # B x 256 x 256 x 256 -> B x 256 x 256 x 256
            nn.ReLU(),
            nn.Conv2d(hidden_dim, mask_channels, kernel_size = 1) # B x 256 x 256 x 256 -> B x 3 x 256 x 256
        )

    def forward(self, low_res_masks, hist_embeddings):
        """
        Forwarding function for this module.

        Args:
            low_res_masks (Tensor): the low resolution masks given by SAM.
            hist_embeddings (Tensor): the embeddings of the image given by the domain encoder.

        Returns:
            Tensor: the refined low resolution masks.
        """
        B, _, H, W = low_res_masks.shape

        query = self.query_conv(low_res_masks) # Shape: B x 256 x 256 x 256
        query_flat = query.view(B, self.hidden_dim, H * W).permute(0, 2, 1) # Shape: B x (256 x 256) x hidden_dim

        key_value = self.hist_layer(hist_embeddings) # Shape: B x number_patches x hidden_dim

        attention_output, _ = self.attention(query_flat, key_value, key_value) # Shape: B x (256 x 256) x hidden_dim
        temp_attention = attention_output.permute(0, 2, 1).view(B, self.hidden_dim, H, W)

        output = self.refinement_layer(temp_attention) + low_res_masks

        return output
