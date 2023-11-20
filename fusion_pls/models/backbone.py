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
        self.img_enc = ResNetEncoderDecoder(cfg.IMG)
        # img_enc_cfg = cfg.PCD
        # img_enc_cfg.INPUT_DIM = 3
        # img_enc_cfg.FREEZE = False
        # self.img_enc = MinkEncoderDecoder(img_enc_cfg)

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
                # AutoWeightedFeatureFusion(
                #     c_in_m1=cfg.PCD.CHANNELS[level - self.n_levels],
                #     c_in_m2=cfg.PCD.CHANNELS[level - self.n_levels],
                #     c_out=self.out_dim[level],
                # )
                MixDimensionAttentionFusion(
                    pcd_c_in=cfg.PCD.CHANNELS[level - self.n_levels],
                    img_c_in=cfg.IMG.HIDDEN_DIM,
                    cla_out_size=1,
                    sa_ks=7,
                    csa_nhead=1,
                )
            )

        # sem_head_in_dim = self.out_dim[-1]
        sem_head_in_dim = cfg.PCD.CHANNELS[-1]
        self.sem_head = nn.Linear(sem_head_in_dim, data_cfg.NUM_CLASSES)

    def forward(self, x):
        # get pcd feats
        pcd_feats = [torch.from_numpy(f).float().cuda() for f in x["feats"]]
        pcd_feats, coords = self.pcd_enc(pcd_feats, x["pt_coord"])
        # todo: use voxel feats

        # get img feats
        image = [torch.from_numpy(i).float().cuda() for i in x["image"]]

        # use minkowski encoder
        # img_feats = self.proj_img2pcd(x['map_img2pcd'], image)
        # img_feats, _ = self.img_enc(img_feats, x["pt_coord"])

        # use resnet encoder
        img_sizes = [tuple(i.shape[-2:]) for i in x['image']]
        img_feats = self.img_enc(image, img_sizes)
        # # project img_feats to pcd
        # img_feats = [
        #     self.proj_img2pcd(x['map_img2pcd'], level)
        #     for level in img_feats
        # ]

        # # pad batch
        # pcd_feats, batched_coords, pad_masks = self.pad_batch(coords, pcd_feats)
        # img_feats, _, _ = self.pad_batch(coords, img_feats)
        #
        # assert pcd_feats[0].shape[0] == img_feats[0].shape[0], \
        #     "pcd_feats and img_feats must have the same number of points"
        #
        # # cross modal feature fusion
        # fused_feats = []
        # for l in range(self.n_levels):
        #     fused_feats.append(self.fusion[l](img_feats[l], pcd_feats[l]))

        # cross modal feature fusion
        fused_feats = []
        for l in range(self.n_levels):
            fused_feats.append(self.fusion[l](img_feats[l], pcd_feats[l], x['map_img2pcd']))
        fused_feats, batched_coords, pad_masks = self.pad_batch(coords, fused_feats)

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
            img2pcd_feats.append(
                img_feats[b].permute(1, 2, 0).contiguous()[m[:, 1], m[:, 0]]
            )

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


class MixDimensionAttentionFusion(nn.Module):
    def __init__(self, pcd_c_in, img_c_in, cla_out_size=1, sa_ks=7, csa_nhead=1):
        super().__init__()
        self.img_proj = nn.Linear(img_c_in, pcd_c_in)

        self.channel_attn = ChannelAttention(pcd_c_in, cla_out_size)
        # self.cross_attn = nn.MultiheadAttention(pcd_c_in, csa_nhead, batch_first=True)
        self.channel_leaner = blocks.MLP(pcd_c_in, pcd_c_in * 2, pcd_c_in, 3)

        self.spatial_attn = SpatialAttention(sa_ks)

    def forward(self, img_feats, pcd_feats, map_i2p):
        """
        Args:
            img_feats: list of [C, H, W] torch.Tensor
            pcd_feats: list of [N, C] torch.Tensor
        Returns:
            fused_feats: list of [N, C] torch.Tensor
        """
        assert len(img_feats) == len(pcd_feats), \
            "Batch size of img_feats and pcd_feats must be the same"
        assert len(img_feats) == len(map_i2p), \
            "Batch size of img_feats and map_img2pcd must be the same"

        fused_feats = []
        for b in range(len(img_feats)):
            # img projection
            i_feats = img_feats[b].permute(1, 2, 0).contiguous()  # [H, W, C]
            i_feats = self.img_proj(i_feats)  # [H, W, C]
            i_feats = i_feats.permute(2, 0, 1).contiguous()  # [C, H, W]

            # channel attention
            ca_img_feats = self.channel_attn(i_feats.unsqueeze(0))  # [1, C, 1]
            ca_img_feats = ca_img_feats.flatten(2).permute(0, 2, 1).contiguous()
            # p_feats, _ = self.cross_attn(pcd_feats[b].unsqueeze(0), ca_img_feats, ca_img_feats)
            p_feats = (pcd_feats[b].unsqueeze(0) * ca_img_feats).squeeze(0)  # [N, C]
            # p_feats = self.channel_leaner(p_feats)

            # spatial attention
            sa_img_feats = self.spatial_attn(img_feats[b].unsqueeze(0))  # [1, 1, H, W]
            # map feats to pcd
            sa_img_feats = sa_img_feats.squeeze(0).permute(1, 2, 0).contiguous()  # [H, W, 1]
            m = map_i2p[b]  # [N, 2]
            sa_img_feats = sa_img_feats[m[:, 1], m[:, 0]]  # [N, 1]
            p_feats = p_feats * sa_img_feats

            # fuse
            fused_feats.append(p_feats)

        return fused_feats


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, output_size, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size)
        self.max_pool = nn.AdaptiveMaxPool2d(output_size)

        self.fc1 = nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
