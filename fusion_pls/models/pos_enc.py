import math

import torch
import torch.nn as nn


class PositionEmbeddingSine3D(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.range = cfg.RANGE
        self.max_freq = cfg.MAX_FREQ
        self.dimensionality = cfg.DIMENSIONALITY
        self.num_bands = math.floor(cfg.FEAT_SIZE / cfg.DIMENSIONALITY / 2)
        self.base = cfg.BASE
        pad = cfg.FEAT_SIZE - self.num_bands * 2 * cfg.DIMENSIONALITY
        self.zero_pad = nn.ZeroPad2d((pad, 0, 0, 0))  # left padding

    def forward(self, _x, coor_type="cart"):
        """
        _x [B,N,3]: batched point coordinates
        returns: [B,N,C]: positional encoding of dimension C
        """
        x = _x.clone()
        if coor_type == "cart":
            x[:, :, 0] = x[:, :, 0] / self.range[0]
            x[:, :, 1] = x[:, :, 1] / self.range[1]
            x[:, :, 2] = x[:, :, 2] / self.range[2]
        elif coor_type == "polar":
            # transform x to polar coordinates
            rho = torch.sqrt(x[:, :, 0] ** 2 + x[:, :, 1] ** 2)
            phi = torch.atan2(x[:, :, 1], x[:, :, 0])
            x = torch.stack((rho, phi, x[:, :, 2]), dim=-1)
            # normalize
            x[:, :, 0] = x[:, :, 0] / max(self.range[:2])
            x[:, :, 1] = x[:, :, 1] / torch.tensor([2 * math.pi]).to(x.device)
            x[:, :, 2] = x[:, :, 2] / self.range[2]
        else:
            raise NotImplementedError

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


class PositionEmbeddingLearned3D(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, cfg):
        super().__init__()
        self.in_dim = cfg.DIMENSIONALITY
        self.out_dim = cfg.FEAT_SIZE
        self.proj = nn.Linear(self.in_dim, self.out_dim)
        self.ln = nn.LayerNorm(self.out_dim)
        self.range = cfg.RANGE

    def forward(self, _x, coor_type="cart"):
        """
        _x [B,N,3]: batched point coordinates (cartesian)
        returns: [B,N,C]: positional encoding of dimension C
        """
        x = _x.clone()
        if coor_type == "cart":
            x[:, :, 0] = x[:, :, 0] / self.range[0]
            x[:, :, 1] = x[:, :, 1] / self.range[1]
            x[:, :, 2] = x[:, :, 2] / self.range[2]
        elif coor_type == "polar":
            # transform x to polar coordinates
            rho = torch.sqrt(x[:, :, 0] ** 2 + x[:, :, 1] ** 2)
            phi = torch.atan2(x[:, :, 1], x[:, :, 0])
            x = torch.stack((rho, phi, x[:, :, 2]), dim=-1)
            # normalize
            x[:, :, 0] = x[:, :, 0] / max(self.range[:2])
            x[:, :, 1] = x[:, :, 1] / torch.tensor([2 * math.pi]).to(x.device)
            x[:, :, 2] = x[:, :, 2] / self.range[2]
        else:
            raise NotImplementedError

        enc = self.proj(x)
        enc = self.ln(enc)

        return enc


class MixPositionEmbedding(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.pe_cart = PositionEmbeddingSine3D(cfg)
        self.pe_polar = PositionEmbeddingSine3D(cfg)
        self.in_dim = cfg.DIMENSIONALITY
        self.out_dim = cfg.FEAT_SIZE
        self.proj = nn.Linear(self.out_dim, self.out_dim)
        self.ln = nn.LayerNorm(self.out_dim)

    def forward(self, _x):
        enc_cart = self.pe_cart(_x, type="cart")
        enc_polar = self.pe_polar(_x, type="polar")
        enc = enc_cart + enc_polar
        enc = self.proj(enc)
        enc = self.ln(enc)
        return enc
