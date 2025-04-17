"""File to define the cross attention module that is used to fuse the domain encoder embeddings with SAM image encoder embeddings."""

import torch.nn as nn

class CrossAttentionFusionModule(nn.Module):
    def __init__(self, query_dim, context_dim, embed_dim = 256, num_heads = 8, output_size = 64):
        """
        Constructor for the module.

        Args:
            query_dim (int): the dimension of the embeddings of SAM.
            context_dim (int): the dimension of the embeddings of the domain encoder.
            embed_dim (int, optional): the dimension of the embeddings to output. 
                                        Defaults to 256 (to be compatible with the original size used in SAM mask decoder).
            num_heads (int, optional): the number of head to be used for the MultiheadAttention. Defaults to 8.
            output_size (int, optional): the number of embeddings of size embed_dim to output. 
                                        Defaults to 64 (again to be compatible with SAM mask decoder).
        """
        super().__init__()
        self.output_size = output_size
        self.embed_dim = embed_dim

        self.query_layer = nn.Linear(query_dim, embed_dim)
        self.context_layer = nn.Linear(context_dim, embed_dim)
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first = True)
        self.output_layer = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, context):
        """
        Forwarding function for this module.

        Args:
            query(Tensor): the SAM image encoder embeddings (used as query).
            context (Tensor): the domain encoder embeddings (used as context).

        Returns:
            Tensor: the fused embeddings.
        """
        # query, shape: B x 4096 x sam_encoder_dim.             query should be the features from the sam encoder
        # context, shape: B x number_patches x encoder_dim.     context should be the features from the additional domain encoder
        B, _, _ = query.shape

        temp_query = self.query_layer(query)  # Shape: B x 4096 x embed_dim
        temp_context = self.context_layer(context)  # Shape: B x number_patches x embed_dim
        
        output, _ = self.cross_attention(temp_query, temp_context, temp_context)
        
        fused_embeddings = self.output_layer(output + temp_query)  # Shape: B x 4096 x embed_dim, residual connection is to ensure the gradient flow
        
        return fused_embeddings.permute(0, 2, 1).reshape(B, self.embed_dim, self.output_size, self.output_size)
