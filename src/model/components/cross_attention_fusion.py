import torch.nn as nn

class CrossAttentionFusionModule(nn.Module):
    def __init__(self, query_dim, context_dim, embed_dim = 256, num_heads = 8, output_size = 64):
        super().__init__()
        self.output_size = output_size
        self.embed_dim = embed_dim

        self.query_layer = nn.Linear(query_dim, embed_dim)
        self.context_layer = nn.Linear(context_dim, embed_dim)
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first = True)
        self.output_layer = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, context): # 
        # query, shape: B x 4096 x sam_encoder_dim.             query should be the features from the sam encoder
        # context, shape: B x number_patches x encoder_dim.     context should be the features from the additional domain encoder
        B, _, _ = query.shape

        temp_query = self.query_layer(query)  # Shape: B x 4096 x embed_dim
        temp_context = self.context_layer(context)  # Shape: B x number_patches x embed_dim
        
        output, _ = self.cross_attention(temp_query, temp_context, temp_context)
        
        fused_embeddings = self.output_layer(output + temp_query)  # Shape: B x 4096 x embed_dim, residual connection is to ensure the gradient flow
        
        return fused_embeddings.permute(0, 2, 1).reshape(B, self.embed_dim, self.output_size, self.output_size)