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

        # init img encoder
        # self.img_enc = ResNetEncoderDecoder(cfg.IMG)
        # project img feats to pcd, then use mink encoder
        img_enc_cfg = cfg.PCD
        img_enc_cfg.RESOLUTION = cfg.IMG.RESOLUTION
        img_enc_cfg.INPUT_DIM = cfg.IMG.INPUT_DIM
        img_enc_cfg.FREEZE = cfg.IMG.FREEZE
        self.img_enc = MinkEncoderDecoder(img_enc_cfg)

        # init fusion encoder
        self.n_levels = cfg.FUSION.N_LEVELS
        self.out_dim = cfg.PCD.CHANNELS[-self.n_levels:]
        self.enable_fusion = cfg.FUSION.ENABLE
        if self.enable_fusion:
            self.fusion = nn.ModuleList()
            for level in range(self.n_levels):
                self.fusion.append(
                    AutoWeightedFeatureFusion(
                        c_in=cfg.PCD.CHANNELS[level - self.n_levels],
                        c_out=self.out_dim[level],
                        enable_ca=cfg.FUSION.ENABLE_CA,
                    )
                )

    def forward(self, x):
        # get pcd feats
        pcd_feats = [torch.from_numpy(f).float().cuda() for f in x["feats"]]
        pcd_feats, coors = self.pcd_enc(pcd_feats, x["pt_coord"])
        pcd_feats, coors, pad_masks = self.pad_batch(coors, pcd_feats)

        if self.enable_fusion:
            # get img feats by mink
            image = [torch.from_numpy(i).float().cuda() for i in x["image"]]
            img_feats = self.proj_img2pcd(x['map_img2pcd'], image)
            img_feats, _ = self.img_enc(img_feats, x["pt_coord"])
            img_feats, _, _ = self.pad_batch(coors, img_feats)

            # multi-modal feature fusion
            fused_feats = []
            for l in range(self.n_levels):
                fused_feats.append(
                    self.fusion[l](
                        pcd_feats[l],
                        img_feats[l],
                    )
                )

            return fused_feats, coors, pad_masks

        else:
            return pcd_feats, coors, pad_masks

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
            img2pcd_feats.append(
                img_feats[b].permute(1, 2, 0).contiguous()[m[:, 1], m[:, 0]]
            )

        return img2pcd_feats


class AutoWeightedFeatureFusion(nn.Module):
    def __init__(self, c_in, c_out, enable_ca=False):
        super().__init__()
        self.spatial_weight_leaner = nn.Sequential(
            blocks.MLP(c_in * 2, c_in * 2, c_in, 3),
            blocks.MLP(c_in, c_in, 2, 3),
            nn.Sigmoid(),
        )

        self.enable_ca = enable_ca
        if self.enable_ca:
            self.channel_attn_m1 = nn.Sequential(
                blocks.MLP(c_in * 2, c_in * 2, c_in, 3),
                ChannelAttention(c_in),
            )
            self.channel_attn_m2 = nn.Sequential(
                blocks.MLP(c_in * 2, c_in * 2, c_in, 3),
                ChannelAttention(c_in),
            )

        self.fusion_leaner = blocks.MLP(c_in * 2, c_in * 2, c_in, 3)

        self.output_proj = nn.Linear(c_in, c_out)

    def forward(
            self,
            feats_m1,
            feats_m2,
    ):
        """
        Args:
            feats_m1: [B, N, C_in] torch.Tensor
            feats_m2: [B, N, C_in] torch.Tensor
        Returns:
            output: [B, N, C_out] torch.Tensor
        """
        assert feats_m1.shape[-1] == feats_m2.shape[-1]

        spatial_weights = self.spatial_weight_leaner(
            torch.cat([feats_m1, feats_m2], dim=2)
        )
        weights_m1 = spatial_weights[..., 0].unsqueeze(-1)
        weights_m2 = spatial_weights[..., 1].unsqueeze(-1)

        if self.enable_ca:
            channel_weights_m1 = self.channel_attn_m1(
                torch.cat([feats_m1, feats_m2], dim=2)
            )
            channel_weights_m2 = self.channel_attn_m2(
                torch.cat([feats_m1, feats_m2], dim=2)
            )
            weights_m1 = weights_m1 * channel_weights_m1
            weights_m2 = weights_m2 * channel_weights_m2

        fused_feats = self.fusion_leaner(torch.cat(
            [feats_m1 * weights_m1, feats_m2 * weights_m2],
            dim=2)
        )

        output = self.output_proj(fused_feats + feats_m1)

        return output


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, output_size=1):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(output_size)
        self.max_pool = nn.AdaptiveMaxPool1d(output_size)

        self.fc1 = nn.Conv1d(in_channels, in_channels * 2, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv1d(in_channels * 2, in_channels, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: [B, N, C] torch.Tensor
        Return:
            [B, 1, C] torch.Tensor
        """
        x = x.permute(0, 2, 1).contiguous()
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)
        return out.permute(0, 2, 1).contiguous()


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv1d(2, 1, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: [B, N, C] torch.Tensor
        Return:
            [B, N, 1] torch.Tensor
        """
        x = x.permute(0, 2, 1).contiguous()
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv1(x))
        return out.permute(0, 2, 1).contiguous()
