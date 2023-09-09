import torch
from torch import Tensor, nn
from torch.nn import functional as F

from typing import Optional
from fusion_pls.models.color_encoder import ColorPointEncoder
from fusion_pls.models.mink import MinkEncoderDecoder
from fusion_pls.models.positional_encoder import PositionalEncoder
from fusion_pls.models.blocks import (SelfAttentionLayer,
                                      CrossModalAttentionLayer,
                                      FFNLayer,
                                      MLP)


class FusionEncoder(nn.Module):
    """
    Fuse mono and early-fused pts features
    """

    def __init__(self, cfg: object, data_cfg: object) -> object:
        super().__init__()

        # init mono pts encoder
        cfg.MINK.CHANNELS = cfg.CHANNELS
        cfg.MINK.RESOLUTION = cfg.RESOLUTION
        cfg.MINK.KNN_UP = cfg.KNN_UP
        self.mink = MinkEncoderDecoder(cfg.MINK, data_cfg)

        # init early-fused pts encoder
        cfg.CPE.CHANNELS = cfg.CHANNELS
        cfg.CPE.RESOLUTION = cfg.RESOLUTION
        cfg.CPE.KNN_UP = cfg.KNN_UP
        self.cpe = MinkEncoderDecoder(cfg.CPE, data_cfg)

        # init fusion encoder
        self.n_levels = cfg.FUSION.N_LEVELS
        self.pos_enc = nn.ModuleList()
        self.fusion = nn.ModuleList()
        for level in range(self.n_levels):
            cfg.FUSION.POS_ENC.FEAT_SIZE = cfg.FUSION.OUT_DIM[level]
            self.pos_enc.append(
                PositionalEncoder(cfg.FUSION.POS_ENC),
            )
            self.fusion.append(
                AutoWeightedFeatureFusion(
                    c_in=cfg.FUSION.OUT_DIM[level],
                    c_out=cfg.FUSION.OUT_DIM[level],
                    num_heads=cfg.FUSION.N_HEADS,
                    ffn_hidden_dim=cfg.FUSION.DIM_FFN,
                    dropout=cfg.FUSION.DROPOUT,
                )
            )

        sem_head_in_dim = cfg.FUSION.OUT_DIM[-1]
        self.sem_head = nn.Linear(sem_head_in_dim, 20)

    def forward(self, x):
        mn_feats, mn_coords = self.mink(x)
        ef_feats, ef_coords = self.cpe(x)
        mn_feats, mn_coords, mn_pad_masks = self.pad_batch(mn_coords, mn_feats)
        ef_feats, ef_coords, ef_pad_masks = self.pad_batch(ef_coords, ef_feats)

        # # auto-weighted feature fusion: ef(q) -> mn(kv)
        # feats = []
        # coords = mn_coords
        # for level in range(self.n_levels):
        #     mn_pos = self.pos_enc[level](mn_coords[level])
        #     ef_pos = self.pos_enc[level](ef_coords[level])
        #     feats.append(
        #         self.fusion[level](
        #             ef_feats[level],
        #             mn_feats[level],
        #             pad_masks_m2=mn_pad_masks[level],
        #             q_pos_m1=ef_pos,
        #             kv_pos_m2=mn_pos,
        #         )
        #     )

        feats = mn_feats
        coords = mn_coords

        logits = self.sem_head(feats[-1])
        return feats, coords, mn_pad_masks, logits

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
    def __init__(self, c_in, c_out, num_heads, ffn_hidden_dim=2048, dropout=0.0):
        super().__init__()

        self.weights_generator = nn.ModuleList([
            # CrossModalAttentionLayer(c_in, num_heads, dropout=dropout),
            # SelfAttentionLayer(c_in, num_heads, dropout=dropout),
            FFNLayer(c_in, dim_feedforward=ffn_hidden_dim, dropout=dropout),
            MLP(c_in, c_in, 2, 3),
        ])

        encoder_c_in = c_in * 2
        self.encoder = nn.ModuleList([
            MLP(encoder_c_in, encoder_c_in, c_out, 3),
            # SelfAttentionLayer(c_out, num_heads, dropout=dropout),
            FFNLayer(c_out, dim_feedforward=ffn_hidden_dim, dropout=dropout),
        ])

        # todo: queries_generator

    def forward(
            self,
            q_feats_m1,
            kv_feats_m2,
            pad_masks_m2: Optional[Tensor] = None,
            q_pos_m1: Optional[Tensor] = None,
            kv_pos_m2: Optional[Tensor] = None,
    ):
        assert q_feats_m1.shape == kv_feats_m2.shape, \
            "feats_m1 and feats_m2 must have the same shape"

        # # cross-modal-attention
        # weights = self.weights_generator[0](
        #     q_feats_m1,
        #     kv_feats_m2,
        #     padding_mask=pad_masks_m2,
        #     q_pos=q_pos_m1,
        #     kv_pos=kv_pos_m2,
        # )
        # # self-attention
        # weights = self.weights_generator[1](
        #     weights,
        #     query_pos=q_pos_m1,
        # )
        # FFN
        weights = self.weights_generator[0](torch.cat([q_feats_m1, kv_feats_m2], dim=2))
        # MLP
        weights = self.weights_generator[1](weights)

        weighted_feats_m1 = weights[..., 0].unsqueeze(-1) * q_feats_m1
        weighted_feats_m2 = weights[..., 1].unsqueeze(-1) * kv_feats_m2
        fused_feats = torch.cat([weighted_feats_m1, weighted_feats_m2], dim=2)
        # MLP
        fused_feats = self.encoder[0](fused_feats)
        # # self-attention
        # fused_feats = self.encoder[1](
        #     fused_feats,
        #     query_pos=q_pos_m1,
        # )
        # FFN
        fused_feats = self.encoder[1](fused_feats)

        return fused_feats
