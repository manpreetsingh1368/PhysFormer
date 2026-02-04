# physformer/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PhysicsMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch, seq_len, _ = x.shape
        q = self.W_q(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        k = self.W_k(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        v = self.W_v(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        attn = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        x = torch.matmul(attn, v)
        x = x.transpose(1,2).contiguous().view(batch, seq_len, self.head_dim*self.num_heads)
        return self.out(x)

class PhysicsTransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = PhysicsMultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm1(x + self.dropout(self.attn(x)))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x

class PhysFormer(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_heads=4, ff_dim=512, num_layers=3, patch_dim=32*32):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([PhysicsTransformerLayer(d_model, num_heads, ff_dim) for _ in range(num_layers)])
        self.patch_decoder = nn.Linear(d_model, patch_dim)

    def forward(self, tokens, num_patches):
        batch = tokens.shape[0]
        x = self.token_embed(tokens)
        for layer in self.layers:
            x = layer(x)

        patches = self.patch_decoder(x[:, :num_patches, :])
        patch_len = patches.shape[-1]
        patch_size = int(math.sqrt(patch_len))
        if patch_size**2 != patch_len:
            pad = patch_size**2 - patch_len
            patches = F.pad(patches, (0, pad))

        num_patches_sqrt = int(math.ceil(math.sqrt(patches.shape[1])))
        total_patches = num_patches_sqrt**2
        pad_patches = total_patches - patches.shape[1]
        if pad_patches > 0:
            extra = torch.zeros((patches.shape[0], pad_patches, patches.shape[2]),
                                device=patches.device, dtype=patches.dtype)
            patches = torch.cat([patches, extra], dim=1)

        images = patches.view(batch, num_patches_sqrt, num_patches_sqrt, patch_size, patch_size)
        images = images.permute(0,1,3,2,4).contiguous()
        images = images.view(batch, 1, num_patches_sqrt*patch_size, num_patches_sqrt*patch_size)
        return images
