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
        self.fusion = nn.ModuleList()
        for level in range(self.n_levels):
            self.fusion.append(
                AutoWeightedFeatureFusion(
                    c_in_m1=cfg.PCD.CHANNELS[level - self.n_levels],
                    c_in_m2=cfg.PCD.CHANNELS[level - self.n_levels],
                    c_out=self.out_dim[level],
                    enable_ca=cfg.FUSION.ENABLE_CA,
                )
                # MixDimensionAttentionFusion(
                #     c_in=cfg.PCD.CHANNELS[level - self.n_levels],
                #     n_heads=cfg.FUSION.N_HEADS,
                # )
            )

        # sem_head_in_dim = self.out_dim[-1]
        # self.sem_head = nn.Linear(sem_head_in_dim, data_cfg.NUM_CLASSES)
        # self.sem_head_img = nn.Linear(cfg.PCD.CHANNELS[-1], data_cfg.NUM_CLASSES)
        # self.sem_head_pcd = nn.Linear(cfg.PCD.CHANNELS[-1], data_cfg.NUM_CLASSES)

    def forward(self, x):
        # get pcd feats
        pcd_feats = [torch.from_numpy(f).float().cuda() for f in x["feats"]]
        # in_field, pcd_out_feats = self.pcd_enc(pcd_feats, x["pt_coord"])
        # pcd_vox_feats = [vf.decomposed_features for vf in pcd_out_feats]
        pcd_feats, coors = self.pcd_enc(pcd_feats, x["pt_coord"])
        pcd_feats, coors, pad_masks = self.pad_batch(coors, pcd_feats)

        # get img feats by mink
        image = [torch.from_numpy(i).float().cuda() for i in x["image"]]
        img_feats = self.proj_img2pcd(x['map_img2pcd'], image)
        # _, img_out_feats = self.img_enc(img_feats, x["pt_coord"])
        # img_vox_feats = [vf.decomposed_features for vf in img_out_feats]
        img_feats, _ = self.img_enc(img_feats, x["pt_coord"])
        img_feats, _, _ = self.pad_batch(coors, img_feats)

        # multi-modal feature fusion
        fused_feats = []
        for l in range(self.n_levels):
            fused_feats.append(
                self.fusion[l](
                    # pcd_vox_feats[l],
                    # img_vox_feats[l],
                    pcd_feats[l],
                    img_feats[l],
                )
            )

        # # project vox to pts ,and batch norm
        # fused_feats, coors = self.pcd_enc.voxel_to_point(in_field, pcd_out_feats, fused_feats)

        # # pad batch
        # fused_feats, batched_coors, pad_masks = self.pad_batch(coors, fused_feats)

        # bb_logits = self.sem_head(fused_feats[-1])
        bb_logits = torch.tensor([]).cuda()

        return fused_feats, coors, pad_masks, bb_logits

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
    def __init__(self, c_in_m1, c_in_m2, c_out, enable_ca=False):
        super().__init__()
        if c_in_m1 != c_in_m2:
            self.proj_enabled = True
            c_in = max(c_in_m1, c_in_m2)
            self.proj_m1 = nn.Linear(c_in_m1, c_in)
            self.proj_m2 = nn.Linear(c_in_m2, c_in)
        else:
            self.proj_enabled = False
            c_in = c_in_m1

        self.spatial_weight_leaner = nn.Sequential(
            blocks.MLP(
                input_dim=c_in * 2,
                hidden_dim=c_in * 2,
                output_dim=c_out,
                num_layers=3,
            ),
            nn.LayerNorm(c_out),
            nn.Linear(c_out, 2),
            nn.Sigmoid(),
        )

        self.enable_ca = enable_ca
        if self.enable_ca:
            self.channel_attn_m1 = ChannelAttention(c_in)
            self.channel_attn_m2 = ChannelAttention(c_in)

        self.fusion_leaner = nn.Sequential(
            blocks.MLP(
                input_dim=c_in * 2,
                hidden_dim=c_in * 2,
                output_dim=c_out,
                num_layers=3,
            ),
            nn.LayerNorm(c_out),
        )

    def forward(
            self,
            feats_m1,
            feats_m2,
    ):
        """
        Args:
            feats_m1: [B, N, C_in_m1] torch.Tensor
            feats_m2: [B, N, C_in_m2] torch.Tensor
        Returns:
            fused_feats: [B, N, C_out] torch.Tensor
        """
        if self.proj_enabled:
            feats_m1 = self.proj_m1(feats_m1)
            feats_m2 = self.proj_m2(feats_m2)

        weights = self.spatial_weight_leaner(torch.cat([feats_m1, feats_m2], dim=2))

        if self.enable_ca:
            feats_m1 = feats_m1 * self.channel_attn_m1(feats_m1 + feats_m2)
            feats_m2 = feats_m2 * self.channel_attn_m2(feats_m1 + feats_m2)

        weighted_feats_m1 = weights[..., 0].unsqueeze(-1) * feats_m1
        weighted_feats_m2 = weights[..., 1].unsqueeze(-1) * feats_m2

        fused_feats = self.fusion_leaner(
            torch.cat([weighted_feats_m1, weighted_feats_m2], dim=2)
        )

        return fused_feats


class MixDimensionAttentionFusion(nn.Module):
    def __init__(self, c_in, n_heads=1):
        super().__init__()
        self.channel_attn_m1 = ChannelAttention(c_in)
        self.channel_attn_m2 = ChannelAttention(c_in)
        self.ca_fusion = nn.MultiheadAttention(c_in, n_heads, batch_first=True)

        self.spatial_attn_m1 = SpatialAttention()
        self.spatial_attn_m2 = SpatialAttention()
        self.sa_fusion = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid(),
        )

        self.feat_fusion = nn.Sequential(
            nn.Linear(c_in * 2, c_in),
            nn.ReLU(True),
            nn.Linear(c_in, c_in),
        )

    def forward(self, feats_m1, feats_m2):
        """
        Args:
            feats_m1: list of [N, C] torch.Tensor
            feats_m2: list of [N, C] torch.Tensor
        Returns:
            fused_feats: list of [N, C] torch.Tensor
        """
        assert len(feats_m1) == len(feats_m2), \
            "Batch size of img_feats and pcd_feats must be the same"

        fused_feats = []
        for f_m1, f_m2 in zip(feats_m1, feats_m2):
            # unsqueeze, out: [1, N, C]
            f_m1 = f_m1.unsqueeze(0)
            f_m2 = f_m2.unsqueeze(0)

            # channel attention, out: [1, 1, C]
            ca_f_m1 = self.channel_attn_m1(f_m1)
            ca_f_m2 = self.channel_attn_m2(f_m2)

            # spatial attention, out: [1, N, 1]
            sa_f_m1 = self.spatial_attn_m1(f_m1)
            sa_f_m2 = self.spatial_attn_m2(f_m2)

            # fuse
            ca_fused_m1 = self.ca_fusion(
                query=ca_f_m1,
                key=ca_f_m2,
                value=ca_f_m2,
            )[0]
            ca_fused_m2 = self.ca_fusion(
                query=ca_f_m2,
                key=ca_f_m1,
                value=ca_f_m1,
            )[0]
            f_m1 = ca_fused_m1 * f_m1
            f_m2 = ca_fused_m2 * f_m2
            fused_feat = self.feat_fusion(torch.cat([f_m1, f_m2], dim=2))

            sa_fused = self.sa_fusion(
                torch.cat([sa_f_m1, sa_f_m2], dim=2)
            )
            fused_feat = sa_fused * fused_feat

            fused_feats.append(fused_feat.squeeze(0))

        return fused_feats


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, output_size=1, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(output_size)
        self.max_pool = nn.AdaptiveMaxPool1d(output_size)

        self.fc1 = nn.Conv1d(in_channels, in_channels // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv1d(in_channels // ratio, in_channels, 1, bias=False)

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
