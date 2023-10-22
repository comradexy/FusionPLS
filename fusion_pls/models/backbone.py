import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from typing import Optional
from fusion_pls.models.color_encoder import ColorPointEncoder
from fusion_pls.models.mink import MinkEncoderDecoder
from fusion_pls.models.positional_encoder import PositionalEncoder
import fusion_pls.models.blocks as blocks


class FusionEncoder(nn.Module):
    """
    Fuse mono and early-fused pts features
    """

    def __init__(self, cfg: object, data_cfg: object) -> object:
        super().__init__()

        # init raw pts encoder
        cfg.PCD.CHANNELS = cfg.CHANNELS
        cfg.PCD.KNN_UP = cfg.KNN_UP
        self.pcd_enc = MinkEncoderDecoder(cfg.PCD, data_cfg)

        # init img_to_pcd pts encoder
        cfg.IMG.CHANNELS = cfg.CHANNELS
        cfg.IMG.KNN_UP = cfg.KNN_UP
        self.img_enc = MinkEncoderDecoder(cfg.IMG, data_cfg)

        # init fusion encoder
        self.n_levels = cfg.FUSION.N_LEVELS
        self.out_dim = cfg.FUSION.OUT_DIM
        if isinstance(self.out_dim, int):
            self.out_dim = [cfg.FUSION.OUT_DIM] * self.n_levels
        self.pcd_out_dim = cfg.PCD.CHANNELS[-self.n_levels:]
        self.img_out_dim = cfg.IMG.CHANNELS[-self.n_levels:]
        self.feats_proj = nn.ModuleList()
        self.fusion = nn.ModuleList()
        for level in range(self.n_levels):
            self.feats_proj.append(
                nn.Linear(
                    cfg.CHANNELS[level - self.n_levels],
                    self.out_dim[level],
                )
            )
            self.fusion.append(
                AutoWeightedFeatureFusion(
                    c_in_m1=self.out_dim[level],
                    c_in_m2=self.out_dim[level],
                    c_out=self.out_dim[level],
                )
            )

        sem_head_pcd_in_dim = cfg.CHANNELS[-1]
        sem_head_img_in_dim = cfg.CHANNELS[-1]
        sem_head_in_dim = self.out_dim[-1]
        self.sem_head_pcd = nn.Linear(sem_head_pcd_in_dim, 20)
        self.sem_head_img = nn.Linear(sem_head_img_in_dim, 20)
        self.sem_head = nn.Linear(sem_head_in_dim, 20)

    def forward(self, x):
        camera_fov_mask = [torch.from_numpy(ind).bool().cuda() for ind in x["uvrgb_ind"]]
        pcd_feats, coords = self.pcd_enc(x)
        img_feats, _ = self.img_enc(x)

        # project feats to out_dim
        pcd_feats = [
            [
                self.feats_proj[l](batch)
                for batch in pcd_feats[l]
            ]
            for l in range(self.n_levels)
        ]
        img_feats = [
            [
                self.feats_proj[l](batch)
                for batch in img_feats[l]
            ]
            for l in range(self.n_levels)
        ]

        pcd_feats_cf = [
            [
                batch[mask]
                for batch, mask in zip(pcd_feats[l], camera_fov_mask)
            ]
            for l in range(self.n_levels)
        ]
        pcd_feats_cf, _, pad_masks_cf = self.pad_batch(coords, pcd_feats_cf)
        img_feats, _, _ = self.pad_batch(coords, img_feats)

        # auto-weighted feature fusion
        fused_feats = []
        for l in range(self.n_levels):
            feats_cf = self.fusion[l](
                img_feats[l],
                pcd_feats_cf[l],
            )
            # unbatch
            fused_feats.append(
                [
                    batch[~mask]
                    for batch, mask in zip(feats_cf, pad_masks_cf[l])
                ]
            )

        # replace pcd_feats in camera fov with fused_feats
        for l in range(self.n_levels):
            for pf, ff, mask in zip(pcd_feats[l], fused_feats[l], camera_fov_mask):
                pf[mask] = ff

        feats, coords, pad_masks = self.pad_batch(coords, pcd_feats)
        bb_logits = self.sem_head(feats[-1])

        return feats, coords, pad_masks, bb_logits

    def pad_batch(self, coors, feats):
        """
        From a list of multi-level features create a list of batched tensors with
        features padded to the max number of points in the batch.

        returns:
            feats: List of batched feature Tensors per feature level
            coors: List of batched coordinate Tensors per feature level
            pad_masks: List of batched bool Tensors indicating padding
        """
        # get max number of points in the batch for each feature level
        maxs = [max([level.shape[0] for level in batch]) for batch in feats]
        # pad and batch each feature level in a single Tensor
        coors = [
            torch.stack([F.pad(f, (0, 0, 0, maxs[i] - f.shape[0])) for f in batch])
            for i, batch in enumerate(coors)
        ]
        pad_masks = [
            torch.stack(
                [
                    F.pad(
                        torch.zeros_like(f[:, 0]), (0, maxs[i] - f.shape[0]), value=1
                    ).bool()
                    for f in batch
                ]
            )
            for i, batch in enumerate(feats)
        ]
        feats = [
            torch.stack([F.pad(f, (0, 0, 0, maxs[i] - f.shape[0])) for f in batch])
            for i, batch in enumerate(feats)
        ]
        return feats, coors, pad_masks


class AutoWeightedFeatureFusion(nn.Module):
    def __init__(self, c_in_m1, c_in_m2, c_out):
        super().__init__()
        c_in = c_in_m1 + c_in_m2
        self.weights_generator = blocks.MLP(c_in, c_in, 2, 3)
        self.activation = torch.sigmoid
        self.encoder = blocks.MLP(c_in, c_in, c_out, 3)

    def forward(
            self,
            feats_m1,
            feats_m2,
    ):
        assert feats_m1.shape == feats_m2.shape, \
            "feats_m1 and feats_m2 must have the same shape"

        # MLP
        weights = self.weights_generator(torch.cat([feats_m1, feats_m2], dim=2))
        # sigmoid
        weights = self.activation(weights)

        weighted_feats_m1 = weights[..., 0].unsqueeze(-1) * feats_m1
        weighted_feats_m2 = weights[..., 1].unsqueeze(-1) * feats_m2
        fused_feats = torch.cat([weighted_feats_m1, weighted_feats_m2], dim=2)
        # MLP
        fused_feats = self.encoder(fused_feats)
        # residual
        fused_feats = fused_feats + feats_m1

        return fused_feats
