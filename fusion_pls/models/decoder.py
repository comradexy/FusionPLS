# Modified by Rodrigo Marcuzzi from https://github.com/facebookresearch/Mask2Former
import fusion_pls.models.blocks as blocks
import torch
from fusion_pls.models.positional_encoder import PositionalEncoder
from torch import nn


class MaskedTransformerDecoder(nn.Module):
    def __init__(self, in_channels, cfg, bb_cfg, data_cfg):
        super().__init__()
        self.hidden_dim = cfg.HIDDEN_DIM

        # cfg.POS_ENC.FEAT_SIZE = cfg.HIDDEN_DIM
        # self.pe_layer = PositionalEncoder(cfg.POS_ENC)

        self.num_layers = cfg.FEATURE_LEVELS * cfg.DEC_BLOCKS
        self.nheads = cfg.NHEADS

        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                blocks.SelfAttentionLayer(
                    d_model=self.hidden_dim, nhead=self.nheads, dropout=0.0
                )
            )
            self.transformer_cross_attention_layers.append(
                blocks.CrossAttentionLayer(
                    d_model=self.hidden_dim, nhead=self.nheads, dropout=0.0
                )
            )
            self.transformer_ffn_layers.append(
                blocks.FFNLayer(
                    d_model=self.hidden_dim, dim_feedforward=cfg.DIM_FFN, dropout=0.0
                )
            )

        self.decoder_norm = nn.LayerNorm(self.hidden_dim)

        self.num_queries = cfg.NUM_QUERIES
        self.num_feature_levels = cfg.FEATURE_LEVELS

        self.query_feat = nn.Embedding(cfg.NUM_QUERIES, self.hidden_dim)
        self.query_embed = nn.Embedding(cfg.NUM_QUERIES, self.hidden_dim)
        # self.level_embed = nn.Embedding(self.num_feature_levels, self.hidden_dim)

        self.mask_feat_proj = nn.Sequential()

        assert isinstance(in_channels, list), "in_channels must be a list"
        if in_channels[-1] != self.hidden_dim:
            self.mask_feat_proj = nn.Linear(in_channels[-1], self.hidden_dim)

        in_channels = in_channels[:-1][-self.num_feature_levels:]
        self.input_proj = nn.ModuleList()
        for ch in in_channels:
            if ch != self.hidden_dim:  # linear projection to hidden_dim
                self.input_proj.append(nn.Linear(ch, self.hidden_dim))
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        self.class_embed = nn.Linear(self.hidden_dim, data_cfg.NUM_CLASSES + 1)
        self.mask_embed = blocks.MLP(self.hidden_dim, self.hidden_dim, cfg.HIDDEN_DIM, 3)

    def forward(self, feats, coors, pad_masks, query=None):
        mask_features = self.mask_feat_proj(feats[-1]) + coors[-1]
        last_pad = pad_masks[-1]
        # last_coors = coors.copy().pop()
        # mask_features = self.mask_feat_proj(feats.copy().pop()) + self.pe_layer(last_coors)
        # last_pad = pad_masks.copy().pop()
        src = []
        # pos = []
        size_list = []

        for i in range(self.num_feature_levels):
            size_list.append(feats[i].shape[1])
            # pos.append(self.pe_layer(coors[i]))
            feat = self.input_proj[i](feats[i])
            src.append(feat)

        bs = src[0].shape[0]
        if query is not None:
            assert query.shape == (bs, self.num_queries, self.hidden_dim), \
                f"Expect query shape: {(bs, self.num_queries, self.hidden_dim)}, " \
                f"but get query shape: {query.shape}"
            output = query
        else:
            output = self.query_feat.weight.unsqueeze(0).repeat(bs, 1, 1)
        query_pos = self.query_embed.weight.unsqueeze(0).repeat(bs, 1, 1)

        predictions_class = []
        predictions_mask = []

        # predictions on learnable query features, first attn_mask
        outputs_class, outputs_mask, attn_mask = self.pred_heads(
            output,
            mask_features,
            pad_mask=last_pad,
        )

        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)
        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            if attn_mask is not None:
                attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output,
                src[level_index],
                attn_mask=attn_mask,
                padding_mask=pad_masks[level_index],
                # pos=pos[level_index],
                pos=coors[level_index],
                query_pos=query_pos,
            )
            # self-attention
            output = self.transformer_self_attention_layers[i](
                output,
                attn_mask=None,
                padding_mask=None,
                query_pos=query_pos,
            )
            # FFN
            output = self.transformer_ffn_layers[i](output)

            # get predictions and attn mask for next feature level
            outputs_class, outputs_mask, attn_mask = self.pred_heads(
                output,
                mask_features,
                pad_mask=last_pad,
            )
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        assert len(predictions_class) == self.num_layers + 1

        out = {"pred_logits": predictions_class[-1], "pred_masks": predictions_mask[-1]}
        out["aux_outputs"] = self.set_aux(predictions_class, predictions_mask)
        last_queries = output

        return out, last_pad, last_queries

    def pred_heads(
            self,
            output,
            mask_features,
            pad_mask=None,
    ):
        decoder_output = self.decoder_norm(output)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bpc->bpq", mask_embed, mask_features)

        attn_mask = (outputs_mask.sigmoid() < 0.5).detach().bool()
        attn_mask[pad_mask] = True
        attn_mask = (
            attn_mask.unsqueeze(1)
            .repeat(1, self.nheads, 1, 1)
            .flatten(0, 1)
            .permute(0, 2, 1)
        )

        return outputs_class, outputs_mask, attn_mask

    @torch.jit.unused
    def set_aux(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_masks": b}
            for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
        ]


class QueryFusionModule(nn.Module):
    def __init__(self, cfg, dec_cfg, aux_dec_cfg):
        """
        Args:
            num_queries_m1: int, number of queries from modality 1
            num_queries_m2: int, number of queries from modality 2
            num_queries_out: int, number of queries to output
            hidden_dim: int, hidden dimension of the MLP
        """
        super().__init__()
        self.num_queries_m1 = aux_dec_cfg.NUM_QUERIES
        self.num_queries_m2 = aux_dec_cfg.NUM_QUERIES
        self.num_queries_out = dec_cfg.NUM_QUERIES
        self.hidden_dim = dec_cfg.HIDDEN_DIM
        self.num_blocks = cfg.BLOCKS

        # self.query_generator = blocks.MLP(
        #     self.num_queries_m1 + self.num_queries_m2,
        #     self.num_queries_m1 + self.num_queries_m2,
        #     self.num_queries_out,
        #     3,
        # )

        self.ffn_layers = nn.ModuleList()
        self.self_attention_layers = nn.ModuleList()
        # self.cross_attention_layers_m1_to_m2 = nn.ModuleList()
        # self.cross_attention_layers_m2_to_m1 = nn.ModuleList()
        self.mlp_layers = nn.ModuleList()
        for _ in range(self.num_blocks):
            self.self_attention_layers.append(
                blocks.SelfAttentionLayer(
                    d_model=self.hidden_dim, nhead=cfg.NHEADS, dropout=cfg.DROPOUT
                )
            )

            # self.cross_attention_layers_m1_to_m2.append(
            #     blocks.CrossAttentionLayer(
            #         d_model=self.hidden_dim, nhead=cfg.NHEADS, dropout=cfg.DROPOUT
            #     )
            # )
            # self.cross_attention_layers_m2_to_m1.append(
            #     blocks.CrossAttentionLayer(
            #         d_model=self.hidden_dim, nhead=cfg.NHEADS, dropout=cfg.DROPOUT
            #     )
            # )
            self.mlp_layers.append(
                blocks.MLP(
                    self.num_queries_m1 + self.num_queries_m2,
                    self.num_queries_m1 + self.num_queries_m2,
                    self.num_queries_out,
                    3,
                )
            )

    def forward(
            self,
            queries_m1,
            queries_m2,
    ):
        """
        Args:
            queries_m1: Tensor of shape [batch_size, num_queries_m1, hidden_dim]
            queries_m2: Tensor of shape [batch_size, num_queries_m2, hidden_dim]
        """
        assert queries_m1.shape[2] == self.hidden_dim and queries_m2.shape[2] == self.hidden_dim, \
            f"Expect hidden_dim: {self.hidden_dim}," \
            f"but get queries_m1: {queries_m1.shape[2]}, queries_m2: {queries_m2.shape[2]}"

        # # concatenate queries_m1 and queries_m2 along the num_queries dimension
        # queries = torch.cat([queries_m1, queries_m2], dim=1)
        # # permute to [batch_size, hidden_dim, num_queries]
        # queries = queries.permute(0, 2, 1)
        # # fuse queries_m1 and queries_m2
        # queries = self.query_generator(queries)
        # # permute to [batch_size, num_queries, hidden_dim]
        # queries = queries.permute(0, 2, 1)

        queries = torch.cat([queries_m1, queries_m2], dim=1)
        for i in range(self.num_blocks):
            queries = self.self_attention_layers[i](q_embed=queries)

            # # cross-attention
            # queries_m1_to_m2 = self.cross_attention_layers_m1_to_m2[i](
            #     q_embed=queries_m1,
            #     bb_feat=queries_m2,
            # )
            # queries_m2_to_m1 = self.cross_attention_layers_m2_to_m1[i](
            #     q_embed=queries_m2,
            #     bb_feat=queries_m1,
            # )

            # MLP
            # # concatenate queries along the num_queries dimension
            # queries = torch.cat([queries_m1_to_m2, queries_m2_to_m1], dim=1)
            # permute to [batch_size, hidden_dim, num_queries]
            queries = queries.permute(0, 2, 1)
            # fuse queries
            queries = self.mlp_layers[i](queries)
            # permute to [batch_size, num_queries, hidden_dim]
            queries = queries.permute(0, 2, 1)

        return queries
