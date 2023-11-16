# Modified by Rodrigo Marcuzzi from https://github.com/facebookresearch/Mask2Former
import fusion_pls.models.blocks as blocks
import torch
from fusion_pls.models.pos_enc import PositionEmbeddingSine3D
from torch import nn


class PanopticDecoder(nn.Module):
    def __init__(self, in_channels, cfg, data_cfg):
        super().__init__()

        self.enable_detector = cfg.DETECTOR.ENABLE

        self.num_feat_levels = cfg.FEATURE_LEVELS
        self.num_decoders = cfg.DEC_BLOCKS
        self.num_layers = self.num_feat_levels * self.num_decoders

        self.num_queries = cfg.NUM_QUERIES
        self.num_queries_things = cfg.DETECTOR.NUM_QUERIES

        self.hidden_dim = cfg.HIDDEN_DIM
        self.nheads = cfg.NHEADS

        cfg.POS_ENC.FEAT_SIZE = cfg.HIDDEN_DIM
        self.pe_layer = PositionEmbeddingSine3D(cfg.POS_ENC)

        self.mask_decoder = nn.ModuleList()
        self.det_decoder = nn.ModuleList()
        for _ in range(self.num_layers):
            self.mask_decoder.append(PanopticTransformer(cfg))
            self.det_decoder.append(InstanceTransformer(cfg))

        self.decoder_norm = nn.LayerNorm(self.hidden_dim)

        self.query_feat_things = nn.Embedding(self.num_queries_things, self.hidden_dim)
        self.query_embed_things = nn.Embedding(self.num_queries_things, self.hidden_dim)
        self.query_feat = nn.Embedding(self.num_queries, self.hidden_dim)
        self.query_embed = nn.Embedding(self.num_queries, self.hidden_dim)

        self.mask_feat_proj = nn.Sequential()

        assert isinstance(in_channels, list), "in_channels must be a list"
        if in_channels[-1] != self.hidden_dim:
            self.mask_feat_proj = nn.Linear(in_channels[-1], self.hidden_dim)

        in_channels = in_channels[:-1][-self.num_feat_levels:]
        self.input_proj = nn.ModuleList()
        for ch in in_channels:
            if ch != self.hidden_dim:  # linear projection to hidden_dim
                self.input_proj.append(nn.Linear(ch, self.hidden_dim))
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        self.cls_pred_pan = nn.Linear(
            self.hidden_dim,
            data_cfg.NUM_CLASSES + 1,
        )
        self.mask_embed = blocks.MLP(
            self.hidden_dim,
            self.hidden_dim,
            self.hidden_dim,
            3,
        )
        if self.enable_detector:
            self.cls_pred_inst = nn.Linear(
                self.hidden_dim,
                data_cfg.NUM_THING_CLASSES + 1,
            )
            self.off_x_embed = blocks.MLP(
                self.hidden_dim,
                self.hidden_dim,
                self.hidden_dim,
                3,
            )
            self.off_y_embed = blocks.MLP(
                self.hidden_dim,
                self.hidden_dim,
                self.hidden_dim,
                3,
            )
            self.off_z_embed = blocks.MLP(
                self.hidden_dim,
                self.hidden_dim,
                self.hidden_dim,
                3,
            )

    def forward(self, feats, coords, pad_masks):
        last_coords = coords.copy().pop()
        mask_feats = self.mask_feat_proj(feats.copy().pop()) + self.pe_layer(last_coords)
        last_pad = pad_masks.copy().pop()
        src = []
        pos = []
        size_list = []

        for i in range(self.num_feat_levels):
            size_list.append(feats[i].shape[1])
            pos.append(self.pe_layer(coords[i]))
            feat = self.input_proj[i](feats[i])
            src.append(feat)

        bs = src[0].shape[0]
        query_things = self.query_feat_things.weight.unsqueeze(0).repeat(bs, 1, 1)
        query_pos_things = self.query_embed_things.weight.unsqueeze(0).repeat(bs, 1, 1)
        query = self.query_feat.weight.unsqueeze(0).repeat(bs, 1, 1)
        query_pos = self.query_embed.weight.unsqueeze(0).repeat(bs, 1, 1)

        pred_class_panoptic = []
        pred_mask_panoptic = []
        pred_class_things = []
        pred_off_x_things = []
        pred_off_y_things = []
        pred_off_z_things = []
        for i in range(self.num_layers):
            level_index = i % self.num_feat_levels

            # instance center heatmap prediction
            if self.enable_detector:
                outputs_class_things, outputs_off_x, outputs_off_y, outputs_off_z = self.inst_pred_heads(
                    query_things,
                    mask_feats,
                )
                query_things = self.det_decoder[i](
                    query_things,
                    src[level_index],
                    pos=pos[level_index],
                    pad_mask=pad_masks[level_index],
                    query_pos=query_pos_things,
                )
                # embed things query to panoptic query
                query[:, :self.num_queries_things] += query_things
                # save detection outputs
                pred_class_things.append(outputs_class_things)
                pred_off_x_things.append(outputs_off_x)
                pred_off_y_things.append(outputs_off_y)
                pred_off_z_things.append(outputs_off_z)

            # segmentation
            outputs_class, outputs_mask, attn_mask = self.mask_pred_heads(
                query,
                mask_feats,
                pad_mask=last_pad,
            )
            if attn_mask is not None:
                attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            query = self.mask_decoder[i](
                query,
                src[level_index],
                attn_mask=attn_mask,
                pad_mask=pad_masks[level_index],
                pos=pos[level_index],
                query_pos=query_pos,
            )
            # save segmentation outputs
            pred_class_panoptic.append(outputs_class)
            pred_mask_panoptic.append(outputs_mask)

        # final prediction
        out = {}
        if self.enable_detector:
            outputs_class_things, outputs_off_x, outputs_off_y, outputs_off_z = self.inst_pred_heads(
                query_things,
                mask_feats,
            )
            pred_class_things.append(outputs_class_things)
            pred_off_x_things.append(outputs_off_x)
            pred_off_y_things.append(outputs_off_y)
            pred_off_z_things.append(outputs_off_z)

            out["inst_outputs"] = {
                "pred_logits": pred_class_things[-1],
                "pred_off_x": pred_off_x_things[-1],
                "pred_off_y": pred_off_y_things[-1],
                "pred_off_z": pred_off_z_things[-1],
                "aux_outputs": self.set_aux_inst(pred_class_things,
                                                 pred_off_x_things,
                                                 pred_off_y_things,
                                                 pred_off_z_things)
            }

        outputs_class, outputs_mask, attn_mask = self.mask_pred_heads(
            query,
            mask_feats,
            pad_mask=last_pad,
        )
        pred_class_panoptic.append(outputs_class)
        pred_mask_panoptic.append(outputs_mask)

        assert len(pred_class_panoptic) == self.num_layers + 1
        out["pan_outputs"] = {
            "pred_logits": pred_class_panoptic[-1],
            "pred_masks": pred_mask_panoptic[-1],
            "aux_outputs": self.set_aux_pan(pred_class_panoptic,
                                            pred_mask_panoptic)
        }

        return out, last_pad

    def mask_pred_heads(
            self,
            output,
            mask_features,
            pad_mask=None,
    ):
        decoder_output = self.decoder_norm(output)
        outputs_class = self.cls_pred_pan(decoder_output)
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

    def inst_pred_heads(self, output, features):
        decoder_output = self.decoder_norm(output)
        outputs_class = self.cls_pred_inst(decoder_output)
        off_x_embed = self.off_x_embed(decoder_output)
        off_y_embed = self.off_y_embed(decoder_output)
        off_z_embed = self.off_z_embed(decoder_output)
        outputs_off_x = torch.einsum("bqc,bpc->bpq", off_x_embed, features)
        outputs_off_y = torch.einsum("bqc,bpc->bpq", off_y_embed, features)
        outputs_off_z = torch.einsum("bqc,bpc->bpq", off_z_embed, features)

        return outputs_class, outputs_off_x, outputs_off_y, outputs_off_z

    @torch.jit.unused
    def set_aux_pan(self, outputs_class, outputs_mask):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_masks": b}
            for a, b in zip(outputs_class[:-1], outputs_mask[:-1])
        ]

    @torch.jit.unused
    def set_aux_inst(self, outputs_class, outputs_off_x, outputs_off_y, outputs_off_z):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_off_x": b, "pred_off_y": c, "pred_off_z": d}
            for a, b, c, d in zip(outputs_class[:-1], outputs_off_x[:-1], outputs_off_y[:-1], outputs_off_z[:-1])
        ]


class InstanceTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_queries = cfg.DETECTOR.NUM_QUERIES
        self.hidden_dim = cfg.HIDDEN_DIM
        self.nheads = cfg.DETECTOR.NHEADS

        self.cross_attention_layer = blocks.CrossAttentionLayer(
            d_model=self.hidden_dim, nhead=self.nheads, dropout=0.0
        )
        self.self_attention_layer = blocks.SelfAttentionLayer(
            d_model=self.hidden_dim, nhead=self.nheads, dropout=0.0
        )
        self.ffn_layer = blocks.FFNLayer(
            d_model=self.hidden_dim, dim_feedforward=cfg.DETECTOR.DIM_FFN, dropout=0.0
        )

    def forward(
            self,
            queries,
            bb_feat,
            pos,
            pad_mask,
            query_pos=None,
    ):
        # cross-attention
        queries = self.cross_attention_layer(
            queries,
            bb_feat,
            padding_mask=pad_mask,
            pos=pos,
            query_pos=query_pos,
        )
        # self-attention
        queries = self.self_attention_layer(
            queries,
            query_pos=query_pos,
        )
        # FFN
        queries = self.ffn_layer(queries)

        return queries


class PanopticTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_queries = cfg.NUM_QUERIES
        self.hidden_dim = cfg.HIDDEN_DIM
        self.nheads = cfg.NHEADS

        self.cross_attention_layer = blocks.CrossAttentionLayer(
            d_model=self.hidden_dim, nhead=self.nheads, dropout=0.0
        )
        self.self_attention_layer = blocks.SelfAttentionLayer(
            d_model=self.hidden_dim, nhead=self.nheads, dropout=0.0
        )
        self.ffn_layer = blocks.FFNLayer(
            d_model=self.hidden_dim, dim_feedforward=cfg.DIM_FFN, dropout=0.0
        )

    def forward(
            self,
            queries,
            feats,
            attn_mask,
            pad_mask,
            pos,
            query_pos,
    ):
        # cross-attention first
        queries = self.cross_attention_layer(
            queries,
            feats,
            attn_mask=attn_mask,
            padding_mask=pad_mask,
            pos=pos,
            query_pos=query_pos,
        )
        # self-attention
        queries = self.self_attention_layer(
            queries,
            attn_mask=None,
            padding_mask=None,
            query_pos=query_pos,
        )
        # FFN
        queries = self.ffn_layer(queries)

        return queries
