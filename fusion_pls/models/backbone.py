import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
import torchvision.models as models

from typing import Optional
from fusion_pls.models.mink import MinkEncoderDecoder
from fusion_pls.models.resnet import ResNetEncoderDecoder
import fusion_pls.models.blocks as blocks


class FusionEncoder(nn.Module):
    """
    Fuse pcd and img features
    """

    def __init__(self, cfg: object, data_cfg: object):
        super().__init__()

        # init raw pts encoder
        self.pcd_enc = MinkEncoderDecoder(cfg.PCD)

        # init img_to_pcd pts encoder
        # self.img_enc = ResNetEncoderDecoder(cfg.IMG)
        img_enc_cfg = cfg.PCD
        img_enc_cfg.INPUT_DIM = 3
        img_enc_cfg.FREEZE = False
        self.img_enc = MinkEncoderDecoder(img_enc_cfg)

        # init fusion encoder
        self.n_levels = cfg.FUSION.N_LEVELS
        self.out_dim = cfg.FUSION.OUT_DIM
        if isinstance(self.out_dim, int):
            self.out_dim = [cfg.FUSION.OUT_DIM] * self.n_levels
        # self.pcd_feats_proj = nn.ModuleList()
        self.img_feats_proj = nn.ModuleList()
        self.fusion = nn.ModuleList()
        for level in range(self.n_levels):
            # self.pcd_feats_proj.append(
            #     nn.Linear(
            #         cfg.PCD.CHANNELS[level - self.n_levels],
            #         self.out_dim[level],
            #     )
            # )
            # self.img_feats_proj.append(
            #     nn.Linear(
            #         cfg.IMG.HIDDEN_DIM,
            #         cfg.PCD.CHANNELS[level - self.n_levels],
            #     )
            # )
            self.fusion.append(
                AutoWeightedFeatureFusion(
                    c_in_m1=cfg.PCD.CHANNELS[level - self.n_levels],
                    c_in_m2=cfg.PCD.CHANNELS[level - self.n_levels],
                    c_out=self.out_dim[level],
                )
            )

        sem_head_in_dim = self.out_dim[-1]
        # sem_head_in_dim = cfg.PCD.CHANNELS[-1]
        self.sem_head = nn.Linear(sem_head_in_dim, data_cfg.NUM_CLASSES)

    def forward(self, x):
        # get pcd feats
        pcd_feats = [torch.from_numpy(f).float().cuda() for f in x["feats"]]
        pcd_feats, coords = self.pcd_enc(pcd_feats, x["pt_coord"])

        # get img feats
        image = [torch.from_numpy(i).float().cuda() for i in x["image"]]
        img_feats = self.proj_img2pcd(x['map_img2pcd'], image)
        img_feats, _ = self.img_enc(img_feats, x["pt_coord"])
        # img_sizes = [tuple(i.shape[-2:]) for i in x['image']]
        # img_feats = self.img_enc(x['image'], img_sizes)
        # # project img_feats to pcd
        # img_feats = [
        #     self.proj_img2pcd(x['map_img2pcd'], level)
        #     for level in img_feats
        # ]
        # img_feats = [
        #     [
        #         self.img_feats_proj[l](batch)
        #         for batch in img_feats[l]
        #     ]
        #     for l in range(self.n_levels)
        # ]

        # pad batch
        pcd_feats, batched_coords, pad_masks = self.pad_batch(coords, pcd_feats)
        img_feats, _, _ = self.pad_batch(coords, img_feats)

        assert pcd_feats[0].shape[0] == img_feats[0].shape[0], \
            "pcd_feats and img_feats must have the same number of points"

        # auto-weighted feature fusion
        fused_feats = []
        for l in range(self.n_levels):
            fused_feats.append(self.fusion[l](img_feats[l], pcd_feats[l]))

        bb_logits = self.sem_head(fused_feats[-1])

        return fused_feats, batched_coords, pad_masks, bb_logits

        # bb_logits = self.sem_head(pcd_feats[-1])
        #
        # return pcd_feats, batched_coords, pad_masks, bb_logits

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

    def proj_img2pcd(self, map_img2pcd, img_feats):
        """
        Project img_feats to pcd_feats
        Args:
            map_img2pcd: list of [Ni, 2] torch.Tensor
            img_feats: list of [C, H, W] torch.Tensor
        Returns:
            img2pcd_feats: list of [Ni, C] torch.Tensor
        """
        assert len(map_img2pcd) == len(img_feats), \
            "Batch size of map_img2pcd and img_feats must be the same"

        img2pcd_feats = []
        for b, m in enumerate(map_img2pcd):
            img2pcd_feats.append(img_feats[b].permute(1, 2, 0)[m[:, 1], m[:, 0]])

        return img2pcd_feats


class AutoWeightedFeatureFusion(nn.Module):
    def __init__(self, c_in_m1, c_in_m2, c_out):
        super().__init__()
        self.fusion_leaner = blocks.MLP(
            input_dim=c_in_m1 + c_in_m2,
            hidden_dim=c_in_m1 + c_in_m2,
            output_dim=c_in_m1,
            num_layers=3,
        )
        self.weight_leaner = nn.Sequential(
            blocks.MLP(c_in_m1, c_in_m1, 1, 3),
            nn.Sigmoid(),
        )
        self.feats_proj = nn.Linear(c_in_m1, c_out)

    def forward(
            self,
            feats_m1,
            feats_m2,
    ):
        """
        Args:
            feats_m1: [B, N, C_in_m1]
            feats_m2: [B, N, C_in_m2]
        Returns:
            fused_feats: [B, N, C_out]
        """
        # MLP
        fusion_feats = self.fusion_leaner(torch.cat([feats_m1, feats_m2], dim=2))
        # sigmoid
        weights = self.weight_leaner(fusion_feats)
        # weighted sum
        fusion_feats = weights * fusion_feats + feats_m1
        # project
        fusion_feats = self.feats_proj(fusion_feats)

        return fusion_feats
