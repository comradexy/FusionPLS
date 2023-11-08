import MinkowskiEngine as ME
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fusion_pls.utils.interpolate import knn_up


class MinkEncoderDecoder(nn.Module):
    """
    ResNet-like architecture using sparse convolutions
    """

    def __init__(self, cfg: object) -> object:
        super().__init__()

        self.input_dim = cfg.INPUT_DIM
        self.modality = cfg.MODALITY
        self.res = cfg.RESOLUTION
        self.knn_up = knn_up(cfg.KNN_UP)

        cs = cfg.CHANNELS
        self.stem = nn.Sequential(
            ME.MinkowskiConvolution(self.input_dim, cs[0], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(cs[0]),

            ME.MinkowskiReLU(True),
            ME.MinkowskiConvolution(cs[0], cs[0], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(cs[0]),
            ME.MinkowskiReLU(inplace=True),
        )

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(cs[0], cs[0], ks=2, stride=2),
            ResidualBlock(cs[0], cs[1], ks=3),
            ResidualBlock(cs[1], cs[1], ks=3),
        )

        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(cs[1], cs[1], ks=2, stride=2),
            ResidualBlock(cs[1], cs[2], ks=3),
            ResidualBlock(cs[2], cs[2], ks=3),
        )

        self.stage3 = nn.Sequential(
            BasicConvolutionBlock(cs[2], cs[2], ks=2, stride=2),
            ResidualBlock(cs[2], cs[3], ks=3),
            ResidualBlock(cs[3], cs[3], ks=3),
        )

        self.stage4 = nn.Sequential(
            BasicConvolutionBlock(cs[3], cs[3], ks=2, stride=2),
            ResidualBlock(cs[3], cs[4], ks=3),
            ResidualBlock(cs[4], cs[4], ks=3),
        )

        self.up1 = nn.ModuleList(
            [
                BasicDeconvolutionBlock(cs[4], cs[5], ks=2, stride=2),
                nn.Sequential(
                    ResidualBlock(cs[5] + cs[3], cs[5], ks=3),
                    ResidualBlock(cs[5], cs[5], ks=3),
                ),
            ]
        )

        self.up2 = nn.ModuleList(
            [
                BasicDeconvolutionBlock(cs[5], cs[6], ks=2, stride=2),
                nn.Sequential(
                    ResidualBlock(cs[6] + cs[2], cs[6], ks=3),
                    ResidualBlock(cs[6], cs[6], ks=3),
                ),
            ]
        )

        self.up3 = nn.ModuleList(
            [
                BasicDeconvolutionBlock(cs[6], cs[7], ks=2, stride=2),
                nn.Sequential(
                    ResidualBlock(cs[7] + cs[1], cs[7], ks=3),
                    ResidualBlock(cs[7], cs[7], ks=3),
                ),
            ]
        )

        self.up4 = nn.ModuleList(
            [
                BasicDeconvolutionBlock(cs[7], cs[8], ks=2, stride=2),
                nn.Sequential(
                    ResidualBlock(cs[8] + cs[0], cs[8], ks=3),
                    ResidualBlock(cs[8], cs[8], ks=3),
                ),
            ]
        )

        levels = [cs[-i] for i in range(4, 0, -1)]
        self.out_bnorm = nn.ModuleList([nn.BatchNorm1d(l) for l in levels])

        self.init_weights(cfg.PRETRAINED)
        if cfg.FREEZE:
            for param in self.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):
        if pretrained is not None:
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
        in_field = self.TensorField(x)

        x0 = self.stem(in_field.sparse())
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        y1 = self.up1[0](x4)
        y1 = ME.cat(y1, x3)
        y1 = self.up1[1](y1)

        y2 = self.up2[0](y1)
        y2 = ME.cat(y2, x2)
        y2 = self.up2[1](y2)

        y3 = self.up3[0](y2)
        y3 = ME.cat(y3, x1)
        y3 = self.up3[1](y3)

        y4 = self.up4[0](y3)
        y4 = ME.cat(y4, x0)
        y4 = self.up4[1](y4)

        out_feats = [y1, y2, y3, y4]

        # vox2feat and apply batchnorm
        feats, coords = self.voxel_to_point(in_field, out_feats)

        return feats, coords

    def voxel_to_point(self, in_field, out_feats):
        coords = [in_field.decomposed_coordinates for _ in range(len(out_feats))]
        coords = [[c * self.res for c in coords[i]] for i in range(len(coords))]
        bs = in_field.coordinate_manager.number_of_unique_batch_indices()
        vox_coords = [
            [l.coordinates_at(i) * self.res for i in range(bs)] for l in out_feats
        ]
        feats = [
            [
                bn(self.knn_up(vox_c, vox_f, pt_c))
                for vox_c, vox_f, pt_c in zip(vc, vf.decomposed_features, pc)
            ]
            for vc, vf, pc, bn in zip(vox_coords, out_feats, coords, self.out_bnorm)
        ]
        return feats, coords

    def TensorField(self, x):
        """
        Build a tensor field from coordinates and features from the
        input batch
        The coordinates are quantized using the provided resolution
        """
        if self.modality == "xyzi" and self.input_dim == 4:
            feats = x["feats"]
            coords = [f[:, :3] for f in x["feats"]]
        elif self.modality == "uvrgb" and self.input_dim == 5:
            feats = x["uvrgb"]
            coords = [f[ind][:, :3] for f, ind in zip(x["feats"], x["uvrgb_ind"])]
        elif self.modality == "rgb" and self.input_dim == 3:
            feats = [f[:, -3:] for f in x["uvrgb"]]
            coords = [f[ind][:, :3] for f, ind in zip(x["feats"], x["uvrgb_ind"])]
        # elif self.modality == "uvdrgb" and self.input_dim == 6:
        #     depth = [np.linalg.norm(f[:, :2], axis=1, keepdims=True) for f in x["feats"]]
        #     feats = [np.concatenate([f[:, 4:6], d, f[:, -3:]], axis=1) for f, d in zip(x["feats"], depth)]
        else:
            raise Exception(
                f"modality '{self.modality}' with input_dim {self.input_dim} not supported"
            )
        # get batched features
        features = torch.from_numpy(np.concatenate(feats, 0)).float()
        # get batched coordinates
        coordinates = ME.utils.batched_coordinates(
            [i / self.res for i in coords], dtype=torch.float32
        )
        # create tensor field
        feat_tfield = ME.TensorField(
            features=features,
            coordinates=coordinates,
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device="cuda",
        )
        return feat_tfield

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


class BasicConvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1, D=3):
        super().__init__()
        self.net = nn.Sequential(
            ME.MinkowskiConvolution(
                inc, outc, kernel_size=ks, dilation=dilation, stride=stride, dimension=D
            ),
            ME.MinkowskiBatchNorm(outc),
            ME.MinkowskiLeakyReLU(inplace=True),
        )

    def forward(self, x):
        out = self.net(x)
        return out


class BasicDeconvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, D=3):
        super().__init__()
        self.net = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                inc, outc, kernel_size=ks, stride=stride, dimension=D
            ),
            ME.MinkowskiBatchNorm(outc),
            ME.MinkowskiLeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1, D=3):
        super().__init__()
        self.net = nn.Sequential(
            ME.MinkowskiConvolution(
                inc, outc, kernel_size=ks, dilation=dilation, stride=stride, dimension=D
            ),
            ME.MinkowskiBatchNorm(outc),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(
                outc, outc, kernel_size=ks, dilation=dilation, stride=1, dimension=D
            ),
            ME.MinkowskiBatchNorm(outc),
        )

        self.downsample = (
            nn.Sequential()
            if (inc == outc and stride == 1)
            else nn.Sequential(
                ME.MinkowskiConvolution(
                    inc, outc, kernel_size=1, dilation=1, stride=stride, dimension=D
                ),
                ME.MinkowskiBatchNorm(outc),
            )
        )

        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out
