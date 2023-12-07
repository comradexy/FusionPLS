# Modified by Rodrigo Marcuzzi from https://github.com/facebookresearch/Mask2Former
import fusion_pls.models.blocks as blocks
import torch
from fusion_pls.models.pos_enc import MixPositionEmbedding, PositionEncoding3D
from torch import nn
from easydict import EasyDict as edict


class PanopticMaskDecoder(nn.Module):
    def __init__(self, in_channels, cfg, data_cfg):
        super().__init__()
        self.num_queries = cfg.NUM_QUERIES
        self.d_model = cfg.D_MODEL
        self.num_feat_levels = cfg.FEATURE_LEVELS

        # query embeddings
        self.query_feat = nn.Embedding(self.num_queries, self.d_model)
        self.query_embed = nn.Embedding(self.num_queries, self.d_model)
        # self.query_feat_ths = nn.Embedding(self.num_queries, self.d_model)
        # self.query_embed_ths = nn.Embedding(self.num_queries, self.d_model)

        self.cfg_pan = edict(cfg.PANOPTIC)
        self.cfg_pan.NUM_CLASSES = data_cfg.NUM_CLASSES
        self.cfg_pan.NUM_QUERIES = cfg.NUM_QUERIES
        self.cfg_pan.D_MODEL = cfg.D_MODEL
        self.cfg_pan.FEATURE_LEVELS = cfg.FEATURE_LEVELS

        self.cfg_inst = edict(self.cfg_pan.copy())
        self.cfg_inst.update(cfg.INSTANCE)
        self.cfg_inst.NUM_THING_CLASSES = data_cfg.NUM_THING_CLASSES

        self.cfg_sem = edict(self.cfg_pan.copy())
        self.cfg_sem.update(cfg.SEMANTIC)

        self.cfg_pe = edict(cfg.POS_ENC)
        self.cfg_pe.FEAT_SIZE = self.cfg_pan.D_MODEL

        if self.cfg_inst.ENABLE:
            assert self.cfg_inst.NUM_QUERIES == self.cfg_pan.NUM_QUERIES, \
                "NUM_QUERIES in cfg.INSTANCE must be equal to NUM_QUERIES in cfg.PANOPTIC."
        if self.cfg_sem.ENABLE:
            assert self.cfg_sem.NUM_QUERIES == self.cfg_pan.NUM_QUERIES, \
                "NUM_QUERIES in cfg.SEMANTIC must be equal to NUM_QUERIES in cfg.PANOPTIC."

        # decoder blocks
        self.sem_decoder = SemanticSegmentor(in_channels, self.cfg_sem)
        self.inst_decoder = MaskSegmentor(self.cfg_inst, mode='instance')
        self.pan_decoder = MaskSegmentor(self.cfg_pan, mode='panoptic')
        # self.query_fusion = QueryFusionModule(d_model=self.d_model, nhead=8)
        # position embedding
        self.pe_layer = PositionEncoding3D(self.cfg_pe)

        self.mask_feat_proj = nn.Sequential()
        assert isinstance(in_channels, list), "in_channels must be a list"
        if in_channels[-1] != self.d_model:
            self.mask_feat_proj = nn.Linear(in_channels[-1], self.d_model)

        in_channels = in_channels[:-1][-self.num_feat_levels:]
        self.input_proj = nn.ModuleList()
        for ch in in_channels:
            if ch != self.d_model:  # linear projection to hidden_dim
                self.input_proj.append(nn.Linear(ch, self.d_model))
            else:
                self.input_proj.append(nn.Sequential())

    def forward(self, feats, coords, pad_masks):
        last_coords = coords.copy().pop()
        last_pad = pad_masks.copy().pop()
        out = {}

        if self.cfg_sem.ENABLE:
            mask_feats, src, pred_sem = self.sem_decoder(feats)

            # position embedding
            mask_feats = mask_feats + self.pe_layer(last_coords)
            pos = []
            for i in range(self.num_feat_levels):
                pos.append(self.pe_layer(coords[i]))

            out["sem_outputs"] = {
                "pred_sem": pred_sem[-1],
                "aux_outputs": self.set_aux_sem(pred_sem)
            }
        else:
            mask_feats = self.mask_feat_proj(feats.copy().pop()) + self.pe_layer(last_coords)
            src = []
            pos = []

            for i in range(self.num_feat_levels):
                pos.append(self.pe_layer(coords[i]))
                feat = self.input_proj[i](feats[i])
                src.append(feat)

        bs = src[0].shape[0]
        query = self.query_feat.weight.unsqueeze(0).repeat(bs, 1, 1)
        query_pos = self.query_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
        # query_ths = self.query_feat_ths.weight.unsqueeze(0).repeat(bs, 1, 1)
        # query_pos_ths = self.query_embed_ths.weight.unsqueeze(0).repeat(bs, 1, 1)

        if self.cfg_inst.ENABLE:
            pred_cls_inst, pred_mask_inst, pred_off_inst, query = self.inst_decoder(
                query, query_pos, src, pos, mask_feats, last_pad, pad_masks
            )
            # query = self.query_fusion(
            #     query_1=query,
            #     query_pos_1=query_pos,
            #     query_2=query_ths,
            #     query_pos_2=query_pos_ths,
            # )
            out["inst_outputs"] = {
                "pred_logits": pred_cls_inst[-1],
                "pred_masks": pred_mask_inst[-1],
                "pred_off_x": pred_off_inst[0][-1],
                "pred_off_y": pred_off_inst[1][-1],
                "pred_off_z": pred_off_inst[2][-1],
                "aux_outputs": self.set_aux_inst(pred_cls_inst,
                                                 pred_mask_inst,
                                                 pred_off_inst)
            }

        pred_cls_pan, pred_mask_pan, query = self.pan_decoder(
            query, query_pos, src, pos, mask_feats, last_pad, pad_masks
        )
        out["pan_outputs"] = {
            "pred_logits": pred_cls_pan[-1],
            "pred_masks": pred_mask_pan[-1],
            "aux_outputs": self.set_aux_pan(pred_cls_pan,
                                            pred_mask_pan)
        }

        return out, last_pad

    @torch.jit.unused
    def set_aux_pan(self, outputs_class, outputs_mask):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": cls, "pred_masks": mask}
            for cls, mask in
            zip(outputs_class[:-1], outputs_mask[:-1])
        ]

    @torch.jit.unused
    def set_aux_sem(self, outputs_sem):
        return [
            {"pred_sem": sem}
            for sem in outputs_sem[:-1]
        ]

    @torch.jit.unused
    def set_aux_inst(self, outputs_class, outputs_mask, outputs_off):
        return [
            {"pred_logits": cls, "pred_masks": mask,
             "pred_off_x": off_x, "pred_off_y": off_y, "pred_off_z": off_z}
            for cls, mask, off_x, off_y, off_z in
            zip(outputs_class[:-1], outputs_mask[:-1],
                outputs_off[0][:-1], outputs_off[1][:-1], outputs_off[2][:-1])
        ]


class MaskSegmentor(nn.Module):
    def __init__(self, cfg, mode='panoptic'):
        super().__init__()
        self.mode = mode

        self.d_model = cfg.D_MODEL
        self.num_heads = cfg.NUM_HEADS
        self.d_ffn = cfg.D_FFN
        self.num_feat_levels = cfg.FEATURE_LEVELS
        self.num_decoders = cfg.DEC_BLOCKS
        self.num_queries = cfg.NUM_QUERIES
        self.dropout = cfg.DROPOUT
        self.num_classes = cfg.NUM_CLASSES
        # if mode == 'instance':
        #     self.num_classes_things = cfg.NUM_THING_CLASSES

        # decoder layers
        self.decoder = nn.ModuleList()
        self.num_layers = self.num_feat_levels * self.num_decoders
        for _ in range(self.num_layers):
            self.decoder.append(
                TransformerLayer(
                    d_model=self.d_model,
                    d_ffn=self.d_ffn,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                )
            )
        self.decoder_norm = nn.LayerNorm(self.d_model)

        # prediction heads
        if mode == 'panoptic':
            self.cls_pred = nn.Linear(
                self.d_model,
                self.num_classes + 1,
            )
            self.mask_embed = blocks.MLP(
                self.d_model,
                self.d_model,
                self.d_model,
                3,
            )
        elif mode == 'semantic':
            self.sem_pred = nn.Linear(
                self.d_model,
                self.num_classes,
            )
        elif mode == 'instance':
            self.cls_pred = nn.Linear(
                self.d_model,
                # self.num_classes_things + 1,
                self.num_classes + 1,
            )
            self.mask_embed = blocks.MLP(
                self.d_model,
                self.d_model,
                self.d_model,
                3,
            )
            self.off_x_embed = blocks.MLP(
                self.d_model,
                self.d_model,
                self.d_model,
                3,
            )
            self.off_y_embed = blocks.MLP(
                self.d_model,
                self.d_model,
                self.d_model,
                3,
            )
            self.off_z_embed = blocks.MLP(
                self.d_model,
                self.d_model,
                self.d_model,
                3,
            )
        else:
            raise NotImplementedError

    def forward(self, query, query_pos, src, pos, mask_feats, last_pad, pad_masks):
        pred_cls = []
        pred_mask = []
        pred_sem = []
        pred_off = [[] for _ in range(3)]

        if self.mode == 'panoptic':
            for i in range(self.num_layers):
                level_index = i % self.num_feat_levels

                outputs_class, outputs_mask, attn_mask = self.mask_pred_heads(
                    query,
                    mask_feats,
                    pad_mask=last_pad,
                )
                if attn_mask is not None:
                    attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

                query = self.decoder[i](
                    query,
                    src[level_index],
                    attn_mask=attn_mask,
                    pad_mask=pad_masks[level_index],
                    pos=pos[level_index],
                    query_pos=query_pos,
                )

                pred_cls.append(outputs_class)
                pred_mask.append(outputs_mask)

            outputs_class, outputs_mask, attn_mask = self.mask_pred_heads(
                query,
                mask_feats,
                pad_mask=last_pad,
            )

            pred_cls.append(outputs_class)
            pred_mask.append(outputs_mask)

            assert len(pred_cls) == self.num_layers + 1

            return pred_cls, pred_mask, query

        elif self.mode == 'instance':
            for i in range(self.num_layers):
                level_index = i % self.num_feat_levels

                outputs_class, outputs_mask, attn_mask, outputs_off = self.inst_pred_heads(
                    query,
                    mask_feats,
                    pad_mask=last_pad,
                )
                if attn_mask is not None:
                    attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

                query = self.decoder[i](
                    query,
                    src[level_index],
                    attn_mask=attn_mask,
                    pos=pos[level_index],
                    pad_mask=pad_masks[level_index],
                    query_pos=query_pos,
                )

                pred_cls.append(outputs_class)
                pred_mask.append(outputs_mask)
                pred_off[0].append(outputs_off[0])
                pred_off[1].append(outputs_off[1])
                pred_off[2].append(outputs_off[2])

            outputs_class, outputs_mask, attn_mask, outputs_off = self.inst_pred_heads(
                query,
                mask_feats,
                pad_mask=last_pad,
            )

            pred_cls.append(outputs_class)
            pred_mask.append(outputs_mask)
            pred_off[0].append(outputs_off[0])
            pred_off[1].append(outputs_off[1])
            pred_off[2].append(outputs_off[2])

            assert len(pred_cls) == self.num_layers + 1

            return pred_cls, pred_mask, pred_off, query

        elif self.mode == 'semantic':
            for i in range(self.num_layers):
                level_index = i % self.num_feat_levels
                outputs_sem = self.sem_pred_heads(src[level_index])
                pred_sem.append(outputs_sem)
            outputs_sem = self.sem_pred_heads(mask_feats)
            pred_sem.append(outputs_sem)

            return pred_sem

        else:
            raise NotImplementedError

    def mask_pred_heads(
            self,
            output,
            mask_feats,
            pad_mask=None,
    ):
        decoder_output = self.decoder_norm(output)
        outputs_class = self.cls_pred(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bpc->bpq", mask_embed, mask_feats)

        attn_mask = (outputs_mask.sigmoid() < 0.5).detach().bool()
        attn_mask[pad_mask] = True
        attn_mask = (
            attn_mask.unsqueeze(1)
            .repeat(1, self.num_heads, 1, 1)
            .flatten(0, 1)
            .permute(0, 2, 1)
        )

        return outputs_class, outputs_mask, attn_mask

    def inst_pred_heads(
            self,
            output,
            mask_feats,
            pad_mask=None,
    ):
        decoder_output = self.decoder_norm(output)
        outputs_class = self.cls_pred(decoder_output)

        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bpc->bpq", mask_embed, mask_feats)
        attn_mask = (outputs_mask.sigmoid() < 0.5).detach().bool()
        attn_mask[pad_mask] = True
        attn_mask = (
            attn_mask.unsqueeze(1)
            .repeat(1, self.num_heads, 1, 1)
            .flatten(0, 1)
            .permute(0, 2, 1)
        )

        off_x_embed = self.off_x_embed(decoder_output)
        off_y_embed = self.off_y_embed(decoder_output)
        off_z_embed = self.off_z_embed(decoder_output)
        outputs_off_x = torch.einsum("bqc,bpc->bpq", off_x_embed, mask_feats)
        outputs_off_y = torch.einsum("bqc,bpc->bpq", off_y_embed, mask_feats)
        outputs_off_z = torch.einsum("bqc,bpc->bpq", off_z_embed, mask_feats)

        return outputs_class, outputs_mask, attn_mask, [outputs_off_x, outputs_off_y, outputs_off_z]

    def sem_pred_heads(
            self,
            mask_feats,
    ):
        outputs_sem = self.sem_pred(mask_feats)
        return outputs_sem


class SemanticSegmentor(nn.Module):
    def __init__(self, in_channels, cfg):
        super().__init__()
        self.d_model = cfg.D_MODEL
        self.num_feat_levels = cfg.FEATURE_LEVELS
        self.num_classes = cfg.NUM_CLASSES

        assert isinstance(in_channels, list), \
            "in_channels must be a list"

        self.mask_feat_proj = nn.Sequential()
        self._mask_feat_proj = nn.Sequential()
        self.mask_feat_proj = nn.Linear(in_channels[-1], self.d_model)
        self._mask_feat_proj = nn.Linear(in_channels[-1], self.d_model)

        in_channels = in_channels[:-1][-self.num_feat_levels:]
        self.input_proj = nn.ModuleList()
        self._input_proj = nn.ModuleList()
        for ch in in_channels:
            self.input_proj.append(nn.Linear(ch, self.d_model))
            self._input_proj.append(nn.Linear(ch, self.d_model))

        self.sem_pred = nn.Linear(
            self.d_model,
            self.num_classes,
        )
        self.layer_norm = nn.LayerNorm(self.d_model)

    def forward(self, feats):
        assert isinstance(feats, list), "feats must be a list"
        assert len(feats) == self.num_feat_levels + 1, \
            f"feats must have {self.num_feat_levels + 1} levels, got {len(feats)}"

        mask_feats = feats.copy().pop()
        mask_feats_p = self.mask_feat_proj(mask_feats)
        mask_feats_s = self._mask_feat_proj(mask_feats)
        src_p = []
        src_s = []
        for i in range(self.num_feat_levels):
            src_p.append(self.input_proj[i](feats[i]))
            src_s.append(self._input_proj[i](feats[i]))

        outputs_sem = []
        for i in range(self.num_feat_levels):
            outputs_sem.append(self.sem_pred(src_s[i]))
        outputs_sem.append(self.sem_pred(mask_feats_s))

        mask_feats = self.layer_norm(mask_feats_p + mask_feats_s)
        src = [
            self.layer_norm(src_p[i] + src_s[i])
            for i in range(self.num_feat_levels)
        ]

        return mask_feats, src, outputs_sem


class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_ffn, num_heads, dropout):
        super().__init__()
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.num_heads = num_heads
        self.dropout = dropout

        self.cross_attention_layer = blocks.CrossAttentionLayer(
            d_model=self.d_model, nhead=self.num_heads, dropout=self.dropout
        )
        self.self_attention_layer = blocks.SelfAttentionLayer(
            d_model=self.d_model, nhead=self.num_heads, dropout=self.dropout
        )
        self.ffn_layer = blocks.FFNLayer(
            d_model=self.d_model, dim_feedforward=self.d_ffn, dropout=self.dropout
        )

    def forward(
            self,
            queries,
            feats,
            attn_mask=None,
            pad_mask=None,
            pos=None,
            query_pos=None,
    ):
        assert queries.shape[-1] == self.d_model and feats.shape[-1] == self.d_model, \
            "queries and feats must have the same dimensionality," \
            f"got {queries.shape[-1]} and {feats.shape[-1]} respectively"

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


class QueryFusionModule(nn.Module):
    def __init__(self, d_model, nhead=8, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.attn = blocks.CrossAttentionLayer(d_model=self.d_model, nhead=nhead, dropout=dropout)
        self.ffn = blocks.FFNLayer(d_model=self.d_model, dim_feedforward=4 * self.d_model, dropout=dropout)

    def forward(self, query_1, query_pos_1, query_2, query_pos_2):
        query_fused = self.ffn(
            self.attn(
                q_embed=query_1,
                bb_feat=query_2,
                pos=query_pos_2,
                query_pos=query_pos_1,
            )
        )
        query_fused = query_fused + query_1
        return query_fused
