import math

import torch
import torch.nn as nn


class PositionEmbeddingSine3D(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.max_freq = cfg.MAX_FREQ
        self.dimensionality = cfg.DIMENSIONALITY
        self.num_bands = math.floor(cfg.FEAT_SIZE / cfg.DIMENSIONALITY / 2)
        self.base = cfg.BASE
        pad = cfg.FEAT_SIZE - self.num_bands * 2 * cfg.DIMENSIONALITY
        self.zero_pad = nn.ZeroPad2d((pad, 0, 0, 0))  # left padding

    def forward(self, _x):
        """
        _x [B,N,3]: batched point coordinates
        returns: [B,N,C]: positional encoding of dimension C
        """
        x = _x.clone()
        x[:, :, 0] = x[:, :, 0] / 48
        x[:, :, 1] = x[:, :, 1] / 48
        x[:, :, 2] = x[:, :, 2] / 4
        x = x.unsqueeze(-1)
        scales = torch.logspace(
            0.0,
            math.log(self.max_freq / 2) / math.log(self.base),
            self.num_bands,
            base=self.base,
            device=x.device,
            dtype=x.dtype,
        )
        # reshaping
        scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]
        x = x * scales * math.pi
        x = torch.cat([x.sin(), x.cos()], dim=-1)
        x = x.flatten(2)
        enc = self.zero_pad(x)
        return enc


# todo: add positional encoding with learnable parameters
class PositionEmbeddingLearned2D(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, cfg):
        super().__init__()
        self.num_pos_feats = cfg.NUM_POS_FEATS
        self.row_embed = nn.Embedding(50, self.num_pos_feats)
        self.col_embed = nn.Embedding(50, self.num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list):
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos
