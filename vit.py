import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size_x: int, patch_size_y: int, img_size_x: int, img_size_y: int, embedding_dim: int, in_channels: int):
        super(PatchEmbedding, self).__init__()
        
        self.img_size_x = img_size_x
        self.img_size_y = img_size_y
        self.patch_size_x = patch_size_x
        self.patch_size_y = patch_size_y
        self.num_patches = (img_size_x // patch_size_x) * (img_size_y // patch_size_y)

        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embedding_dim,
            kernel_size=(patch_size_x, patch_size_y),
            stride=(patch_size_x, patch_size_y)
        )

    def forward(self, x: torch.Tensor):
        """
        Shape of x : (batch_size, in_channels, img_size_x, img_size_y)

        We want to return : (batch_size, num_patches, embedding_dim)
        """

        x = self.proj(x)

        x = x.flatten(2)
        x = x.transpose(1, 2)

        return x


class Attention(nn.Module):
    def __init__(self, embedding_dim, n_heads, qkv_bias=False):
        super(Attention, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.heads = n_heads
        self.head_dim = embedding_dim // n_heads
        
        assert (self.head_dim * n_heads == embedding_dim), "Embedding dimension must be perfectly divisible by number of heads"
        
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=qkv_bias)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=qkv_bias)
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=qkv_bias)
        
        self.fc_out = nn.Linear(embedding_dim, embedding_dim)
        
    def forward(self, key, query, value, mask):
        N = self.query.shape[1]
        key_len, query_len, value_len = key.shape[1], query.shape[1], value.shape[1]
        embedding_dim = key.shape[2]
        
        query = query.reshape(N, query_len, self.heads, self.head_dim)
        key = key.reshape(N, key_len, self.heads, self.head_dim)
        value = value.reshape(N, value_len, self.heads, self.head_dim)
        
        query = self.queries(query)
        key = self.keys(key)
        value = self.values(value)
        
        energy = torch.einsum("nqhd,nkhd->nhqk", [query, key])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
            
        attention = torch.softmax(energy / (self.embedding_dim ** (1/2)), dim=3)
        
        attention = torch.einsum("nhql,nlhd->nqhd", [attention, value]).reshape(
            N, query_len, self.embedding_dim
        )
        
        out = self.fc_out(attention)
        
        return out

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, prob=0.):
        super(MLP, self).__init__()
        
        self.fc_1 = nn.Linear(in_features, hidden_features)
        self.activation = nn.GELU()
        self.fc_2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(prob)
        
    def forward(self, x):
        x = self.fc_1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.fc_2(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        return x
        
class Block(nn.Module):
    def __init__(self, embedding_dim, n_heads, hidden_dim, prob=0., attn_prob=0., qkv_bias=True):
        super(Block, self).__init__()
        
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.attention = Attention(
            embedding_dim,
            n_heads,
            qkv_bias
        )
        
        self.norm2 = nn.LayerNorm(embedding_dim)
        mlp_ratio = hidden_dim // embedding_dim
        self.mlp = MLP(
            embedding_dim,
            hidden_dim,
            embedding_dim
        )
        
    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        
        return x
    
class VisionTransformer(nn.Module):
    def __init__(self, img_size_x, img_size_y, patch_size_x, patch_size_y, in_channels, n_classes, embedding_dim=512, depth=12, n_heads=12, hidden_dim=2048, qkv_bias=True, prob=0., attn_prob=0.):
        super(VisionTransformer, self).__init__()
        
        self.patch_embeddings = PatchEmbedding(patch_size_x, patch_size_y, img_size_x, img_size_y, embedding_dim, in_channels)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.positional_embedding = nn.Parameter(torch.zeros(1, 1 + self.patch_embeddings.num_patches, embedding_dim))
        self.pos_drop = nn.Dropout(prob)
        
        self.blocks = nn.ModuleList(
            [
                Block(
                    embedding_dim,
                    n_heads,
                    hidden_dim,
                    prob,
                    attn_prob,
                    qkv_bias
                )
                for _ in range(depth)
            ]
        )
        
        self.norm = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, n_classes)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.patch_embeddings(x)
        
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.positional_embedding
        x = self.pos_drop(x)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        
        cls_token_final = x[:, 0]
        x = self.head(cls_token_final)
        
        return x

