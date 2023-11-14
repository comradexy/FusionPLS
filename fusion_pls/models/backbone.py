import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
import torchvision.models as models

from typing import Optional
from fusion_pls.models.mink import MinkEncoderDecoder
from fusion_pls.models.resnet import ResNetEncoderDecoder
from fusion_pls.models.pos_enc import PositionEmbeddingSine3D, PositionEmbeddingLearned2D
import fusion_pls.models.blocks as blocks


# todo: add pcd encoder
class PCDEncoder(nn.Module):
    pass


class FusionEncoder(nn.Module):
    """
    Fuse pcd and img features
    """

    def __init__(self, cfg: object, data_cfg: object):
        super().__init__()

        # init raw pts encoder
        self.pcd_enc = MinkEncoderDecoder(cfg.PCD)

        # init img_to_pcd pts encoder
        self.img_enc = ResNetEncoderDecoder(cfg.IMG)

        # init fusion encoder
        self.n_levels = cfg.FUSION.N_LEVELS
        self.out_dim = cfg.FUSION.OUT_DIM
        if isinstance(self.out_dim, int):
            self.out_dim = [cfg.FUSION.OUT_DIM] * self.n_levels

        self.pcd_feats_proj = nn.ModuleList()
        self.img_feats_proj = nn.ModuleList()
        self.pos_embed_3d = nn.ModuleList()
        # todo: init 2d position embedding
        self.fusion = nn.ModuleList()

        for level in range(self.n_levels):
            self.pcd_feats_proj.append(
                nn.Linear(
                    cfg.PCD.CHANNELS[level - self.n_levels],
                    self.out_dim[level],
                )
            )

            self.img_feats_proj.append(
                nn.Sequential(
                    Conv2DModule(
                        cfg.IMG.HIDDEN_DIM,
                        self.out_dim[level],
                        ks=7,
                        s=4,
                        p=3,
                    ),
                    Conv2DModule(
                        self.out_dim[level],
                        self.out_dim[level],
                        ks=3,
                        s=2,
                        p=1,
                    ),
                    Conv2DModule(
                        self.out_dim[level],
                        self.out_dim[level],
                        ks=3,
                        s=2,
                        p=1,
                    ),
                    # nn.Linear(
                    #     cfg.IMG.HIDDEN_DIM,
                    #     self.out_dim[level],
                    # ),
                )
            )

            cfg.POS_EMB_3D.FEAT_SIZE = self.out_dim[level]
            self.pos_embed_3d.append(
                PositionEmbeddingSine3D(cfg.POS_EMB_3D)
            )

            self.fusion.append(
                FeatureFusionModule(
                    c_in=self.out_dim[level],
                    hidden_dim=cfg.FUSION.HIDDEN_DIM,
                    n_head=cfg.FUSION.N_HEAD,
                    dropout=cfg.FUSION.DROPOUT,
                )
            )

        sem_head_in_dim = self.out_dim[-1]
        self.sem_head = nn.Linear(sem_head_in_dim, data_cfg.NUM_CLASSES)

    def forward(self, x):
        # get pcd feats
        pcd = [f[ind] for f, ind in zip(x['feats'], x['indices'])]
        pcd_feats, coords = self.pcd_enc(pcd)
        # pad batch
        pcd_feats, coords, pad_masks = self.pad_batch(coords, pcd_feats)

        # get img feats
        img_sizes = [tuple(i.shape[-2:]) for i in x['image']]
        img_feats = self.img_enc(x['image'])  # list of [B, C, H, W] torch.Tensor
        # img_feats = [
        #     lf.flatten(-2).permute(0, 2, 1)  # [B, H*W, C]
        #     for lf in img_feats
        # ]

        # # project img_feats to pcd
        # img_feats = [
        #     self.proj_img2pcd(x['map_img2pcd'], level)
        #     for level in img_feats
        # ]

        # # project feats to out_dim
        # pcd_feats = [
        #     [
        #         self.pcd_feats_proj[l](batch)
        #         for batch in pcd_feats[l]
        #     ]
        #     for l in range(self.n_levels)
        # ]
        # img_feats = [
        #     [
        #         self.img_feats_proj[l](batch)
        #         for batch in img_feats[l]
        #     ]
        #     for l in range(self.n_levels)
        # ]

        # # pad batch
        # pcd_feats, batched_coords, pad_masks = self.pad_batch(coords, pcd_feats)
        # img_feats, _, _ = self.pad_batch(coords, img_feats)

        # assert pcd_feats[0].shape[0] == img_feats[0].shape[0], \
        #     "pcd_feats and img_feats must have the same number of points"

        # feature fusion
        fused_feats = []
        for l in range(self.n_levels):
            pcd_feat = self.pcd_feats_proj[l](pcd_feats[l])
            img_feat = self.img_feats_proj[l](img_feats[l]).flatten(-2).permute(0, 2, 1)
            fused_feats.append(
                self.fusion[l](
                    feats_m1=pcd_feat,
                    feats_m2=img_feat,
                    pos_embed_m1=self.pos_embed_3d[l](coords[l]),
                    pos_embed_m2=None,
                )
            )

        bb_logits = self.sem_head(fused_feats[-1])

        del pcd_feats, img_feats

        return fused_feats, coords, pad_masks, bb_logits

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
            map_img2pcd: list of [Ni, 2] np.ndarray
            img_feats: [B, C, H, W] torch.Tensor
        Returns:
            img2pcd_feats: list of [Ni, C] torch.Tensor
        """
        assert len(map_img2pcd) == len(img_feats), \
            "Batch size of map_img2pcd and img_feats must be the same"

        img2pcd_feats = []
        for b, m in enumerate(map_img2pcd):
            img2pcd_feats.append(img_feats[b].permute(1, 2, 0)[m[:, 1], m[:, 0]])

        return img2pcd_feats


class FeatureFusionModule(nn.Module):
    def __init__(self, c_in, hidden_dim=256, n_head=8, dropout=0.0):
        super().__init__()
        self.cross_attn = blocks.CrossAttentionLayer(c_in, n_head, dropout)
        self.self_attn = blocks.SelfAttentionLayer(c_in, n_head, dropout)
        self.ffn = blocks.FFNLayer(c_in, hidden_dim, dropout)

        # self.weights_generator = blocks.MLP(c_in, c_in, 2, 3)
        # self.activation = torch.sigmoid
        # self.encoder = blocks.MLP(c_in, c_in, c_out, 3)

    def forward(
            self,
            feats_m1,
            feats_m2,
            pos_embed_m1=None,
            pos_embed_m2=None,
    ):
        """
        Args:
            feats_m1: [B, N, C_in_m1]
            feats_m2: [B, N, C_in_m2]
            pos_embed_m1: [B, N, C_in_m1]
            pos_embed_m2: [B, N, C_in_m2]
        Returns:
            fused_feats: [B, N, C_out]
        """
        assert feats_m1.shape[0] == feats_m2.shape[0], \
            "feats_m1 and feats_m2 must have the same batch size, but got {} and {}".format(
                feats_m1.shape[0], feats_m2.shape[0]
            )
        assert feats_m1.shape[2] == feats_m2.shape[2], \
            "feats_m1 and feats_m2 must have the same channel size, but got {} and {}".format(
                feats_m1.shape[2], feats_m2.shape[2]
            )
        if pos_embed_m1 is not None:
            assert feats_m1.shape == pos_embed_m1.shape, \
                "feats_m1 and pos_embed_m1 must have the same shape, but got {} and {}".format(
                    feats_m1.shape, pos_embed_m1.shape
                )
        if pos_embed_m2 is not None:
            assert feats_m2.shape == pos_embed_m2.shape, \
                "feats_m2 and pos_embed_m2 must have the same shape, but got {} and {}".format(
                    feats_m2.shape, pos_embed_m2.shape
                )

        # # MLP
        # weights = self.weights_generator(torch.cat([feats_m1, feats_m2], dim=2))
        # # sigmoid
        # weights = self.activation(weights)
        #
        # weighted_feats_m1 = weights[..., 0].unsqueeze(-1) * feats_m1
        # weighted_feats_m2 = weights[..., 1].unsqueeze(-1) * feats_m2
        # fused_feats = torch.cat([weighted_feats_m1, weighted_feats_m2], dim=2)
        # # MLP
        # fused_feats = self.encoder(fused_feats)
        # # # residual
        # # fused_feats = fused_feats + feats_m1

        # cross attention
        fused_feats = self.cross_attn(
            q_embed=feats_m1,
            bb_feat=feats_m2,
            query_pos=pos_embed_m1,
            pos=pos_embed_m2,
        )
        # self attention
        fused_feats = self.self_attn(
            q_embed=fused_feats,
            query_pos=pos_embed_m1,
        )
        # ffn
        fused_feats = self.ffn(fused_feats)
        # residual
        fused_feats = fused_feats + feats_m1

        return fused_feats


class Conv2DModule(nn.Module):
    def __init__(self, c_in, c_out, ks=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                c_in,
                c_out,
                kernel_size=ks,
                stride=s,
                padding=p,
                bias=False,
            ),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)
