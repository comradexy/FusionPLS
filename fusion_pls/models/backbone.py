import torch
import torch.nn as nn
import torch.nn.functional as F

from fusion_pls.models.color_encoder import ColorPointEncoder
from fusion_pls.models.mink import MinkEncoderDecoder


class FusionEncoder(nn.Module):
    """
    Fuse color and geometry features
    """

    def __init__(self, cfg: object, data_cfg: object) -> object:
        super().__init__()
        self.mink = MinkEncoderDecoder(cfg.MINK, data_cfg)
        self.cpe = ColorPointEncoder(cfg.CPE, data_cfg)
        sem_head_in_dim = cfg.MINK.CHANNELS[-1] + cfg.CPE.CHANNELS[-1]
        self.sem_head = nn.Linear(sem_head_in_dim, 20)

        if cfg.MINK.PRETRAINED is None and cfg.CPE.PRETRAINED is None:
            if cfg.PRETRAINED is not None:
                print("Loading pretrained weights from {}".format(cfg.PRETRAINED))
                self.init_weights(cfg.PRETRAINED)

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            print("load pretrained model from {}".format(pretrained))
            pretrained_dict = torch.load(pretrained)
            model_dict = self.state_dict()
            pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items()
                if k in model_dict and model_dict[k].size() == pretrained_dict[k].size()
            }
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)

    def forward(self, x):
        geo_feats, geo_coors = self.mink(x)
        color_feats, color_coors = self.cpe(x)
        feats = [
            [
                torch.cat([g, c], dim=1)
                for g, c in zip(gl, cl)
            ]
            for gl, cl in zip(geo_feats, color_feats)
        ]
        coors = geo_coors
        feats, coors, pad_masks = self.pad_batch(coors, feats)
        logits = self.sem_head(feats[-1])
        return feats, coors, pad_masks, logits

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
