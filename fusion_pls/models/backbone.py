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
        cfg.MINK.KNN_UP = cfg.KNN_UP
        self.mink = MinkEncoderDecoder(cfg.MINK, data_cfg)

        # init early-fused pts encoder
        cfg.CPE.CHANNELS = cfg.CHANNELS
        cfg.CPE.KNN_UP = cfg.KNN_UP
        self.cpe = MinkEncoderDecoder(cfg.CPE, data_cfg)

        # init fusion encoder
        self.n_levels = cfg.FUSION.N_LEVELS
        # self.pos_enc = nn.ModuleList()
        self.fusion = nn.ModuleList()
        for level in range(self.n_levels):
            # cfg.FUSION.POS_ENC.FEAT_SIZE = cfg.FUSION.OUT_DIM[level]
            # self.pos_enc.append(
            #     PositionalEncoder(cfg.FUSION.POS_ENC),
            # )
            self.fusion.append(
                AutoWeightedFeatureFusion(
                    c_in_m1=cfg.CHANNELS[level - self.n_levels],
                    c_in_m2=cfg.CHANNELS[level - self.n_levels],
                    c_out=cfg.FUSION.OUT_DIM[level],
                    ffn_hidden_dim=cfg.FUSION.DIM_FFN,
                    dropout=cfg.FUSION.DROPOUT,
                )
            )

        sem_head_in_dim = cfg.FUSION.OUT_DIM[-1]
        self.sem_head = nn.Linear(sem_head_in_dim, 20)

    def forward(self, x):
        pcd_feats, pcd_coords = self.mink(x)
        img_feats, img_coords = self.cpe(x)
        pcd_feats, pcd_coords, pcd_pad_masks = self.pad_batch(pcd_coords, pcd_feats)
        img_feats, img_coords, img_pad_masks = self.pad_batch(img_coords, img_feats)

        # auto-weighted feature fusion
        feats = []
        coords = pcd_coords
        pad_masks = pcd_pad_masks
        for level in range(self.n_levels):
            # mn_pos = self.pos_enc[level](mn_coords[level])
            # ef_pos = self.pos_enc[level](ef_coords[level])
            feats.append(
                self.fusion[level](
                    img_feats[level],
                    pcd_feats[level],
                )
            )


        logits = self.sem_head(feats[-1])
        return feats, coords, pad_masks, logits

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
    def __init__(self, c_in_m1, c_in_m2, c_out, ffn_hidden_dim=2048, dropout=0.0):
        super().__init__()
        c_in = c_in_m1 + c_in_m2
        self.weights_generator = nn.Sequential(
            # FFNLayer(c_in, dim_feedforward=ffn_hidden_dim, dropout=dropout),
            MLP(c_in, c_in, 2, 3),
        )
        self.activation = torch.sigmoid
        self.encoder = nn.Sequential(
            MLP(c_in, c_in, c_out, 3),
            # FFNLayer(c_out, dim_feedforward=ffn_hidden_dim, dropout=dropout),
        )

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

        return fused_feats


# todo: implement queries_generator
class FusionQueriesGenerator(nn.Module):
    def __init__(self, num_queries, hidden_dim, num_heads, ffn_hidden_dim=2048, dropout=0.0):
        super().__init__()
        self.num_queries = num_queries

        self.query_feat_m1 = nn.Embedding(num_queries, hidden_dim)
        self.query_embed_m1 = nn.Embedding(num_queries, hidden_dim)

        self.query_feat_m2 = nn.Embedding(num_queries, hidden_dim)
        self.query_embed_m2 = nn.Embedding(num_queries, hidden_dim)

        self.query_generator = MLP(2 * num_queries, 2 * num_queries, num_queries, 3)

    def forward(
            self,
            feats_m1, coors_m1, pad_masks_m1,
            feats_m2, coors_m2, pad_masks_m2,
    ):
        pass

