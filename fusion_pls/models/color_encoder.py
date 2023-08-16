import MinkowskiEngine as ME
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fusion_pls.utils.interpolate import knn_up


class ColorPointEncoder(nn.Module):
    """
    ResNet-like architecture using sparse convolutions
    """

    def __init__(self, cfg: object, data_cfg: object) -> object:
        super().__init__()

        n_classes = data_cfg.NUM_CLASSES

        input_dim = cfg.INPUT_DIM
        self.res = cfg.RESOLUTION
        self.knn_up = knn_up(cfg.KNN_UP)

        cs = cfg.CHANNELS
        self.stem = nn.Sequential(
            ME.MinkowskiConvolution(input_dim, cs[0], kernel_size=3, dimension=3),
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

        if cfg.PRETRAINED is not None:
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
        coors = [in_field.decomposed_coordinates for _ in range(len(out_feats))]
        coors = [[c * self.res for c in coors[i]] for i in range(len(coors))]
        bs = in_field.coordinate_manager.number_of_unique_batch_indices()
        vox_coors = [
            [l.coordinates_at(i) * self.res for i in range(bs)] for l in out_feats
        ]
        feats = [
            [
                bn(self.knn_up(vox_c, vox_f, pt_c))
                for vox_c, vox_f, pt_c in zip(vc, vf.decomposed_features, pc)
            ]
            for vc, vf, pc, bn in zip(vox_coors, out_feats, coors, self.out_bnorm)
        ]

        return feats, coors

    def TensorField(self, x):
        """
        Build a tensor field from coordinates and features from the
        input batch
        The coordinates are quantized using the provided resolution
        """
        feat_tfield = ME.TensorField(
            features=torch.from_numpy(np.concatenate(x["rgb"], 0)).float(),
            coordinates=ME.utils.batched_coordinates(
                [i / self.res for i in x["pt_coord"]], dtype=torch.float32
            ),
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


class PointNet(nn.Module):
    def __init__(self, in_channel=9, out_channels=32):
        super(PointNet, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channel, out_channels, 1)
        self.conv2 = torch.nn.Conv1d(out_channels, out_channels, 1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.out_channels = out_channels

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x.transpose(1, 2)

        return x


# class ColorPointEncoder(nn.Module):
#     def __init__(self, out_channels=32):
#         super(ColorPointEncoder, self).__init__()
#         self.rgbi_encoder = PointNet(4, 16)
#         self.xyz_encoder = PointNet(3, 16)
#         self.fusion = PointNet(32, out_channels)
#
#     def forward(self, x):
#         xyz = torch.from_numpy(x["pt_coord"]).float()
#         feats = torch.from_numpy(x["feats"]).float()
#         rgb = torch.from_numpy(x["rgb"]).float()
#         intensity = feats[:, :, -1]
#         rgbi = torch.cat([rgb, intensity], dim=2)
#         rgbi = self.rgbi_encoder(rgbi)
#         xyz = self.xyz_encoder(xyz)
#         out = torch.cat([xyz, rgbi], dim=2)
#         out = self.fusion(out)
#         return out

class CPConvs(nn.Module):
    def __init__(self):
        super(CPConvs, self).__init__()
        self.pointnet1_fea = PointNet(6, 12)
        self.pointnet1_wgt = PointNet(6, 12)
        self.pointnet1_fus = PointNet(108, 12)

        self.pointnet2_fea = PointNet(12, 24)
        self.pointnet2_wgt = PointNet(6, 24)
        self.pointnet2_fus = PointNet(216, 24)

        self.pointnet3_fea = PointNet(24, 48)
        self.pointnet3_wgt = PointNet(6, 48)
        self.pointnet3_fus = PointNet(432, 48)

    def forward(self, points_features, points_neighbor):
        if points_features.shape[0] == 0:
            return points_features

        N, F = points_features.shape
        N, M = points_neighbor.shape
        point_empty = (points_neighbor == 0).nonzero()
        points_neighbor[point_empty[:, 0], point_empty[:, 1]] = point_empty[:, 0]

        pointnet_in_xiyiziuiviri = torch.index_select(points_features[:, [0, 1, 2, 6, 7, 8]], 0,
                                                      points_neighbor.view(-1)).view(N, M, -1)
        pointnet_in_x0y0z0u0v0r0 = points_features[:, [0, 1, 2, 6, 7, 8]].unsqueeze(dim=1).repeat([1, M, 1])
        pointnet_in_xyzuvr = pointnet_in_xiyiziuiviri - pointnet_in_x0y0z0u0v0r0
        points_features[:, 3:6] /= 255.0

        pointnet1_in_fea = points_features[:, :6].view(N, 1, -1)
        pointnet1_out_fea = self.pointnet1_fea(pointnet1_in_fea).view(N, -1)
        pointnet1_out_fea = torch.index_select(pointnet1_out_fea, 0, points_neighbor.view(-1)).view(N, M, -1)
        pointnet1_out_wgt = self.pointnet1_wgt(pointnet_in_xyzuvr)
        pointnet1_feas = pointnet1_out_fea * pointnet1_out_wgt
        pointnet1_feas = self.pointnet1_fus(pointnet1_feas.reshape(N, 1, -1)).view(N, -1)

        pointnet2_in_fea = pointnet1_feas.view(N, 1, -1)
        pointnet2_out_fea = self.pointnet2_fea(pointnet2_in_fea).view(N, -1)
        pointnet2_out_fea = torch.index_select(pointnet2_out_fea, 0, points_neighbor.view(-1)).view(N, M, -1)
        pointnet2_out_wgt = self.pointnet2_wgt(pointnet_in_xyzuvr)
        pointnet2_feas = pointnet2_out_fea * pointnet2_out_wgt
        pointnet2_feas = self.pointnet2_fus(pointnet2_feas.reshape(N, 1, -1)).view(N, -1)

        pointnet3_in_fea = pointnet2_feas.view(N, 1, -1)
        pointnet3_out_fea = self.pointnet3_fea(pointnet3_in_fea).view(N, -1)
        pointnet3_out_fea = torch.index_select(pointnet3_out_fea, 0, points_neighbor.view(-1)).view(N, M, -1)
        pointnet3_out_wgt = self.pointnet3_wgt(pointnet_in_xyzuvr)
        pointnet3_feas = pointnet3_out_fea * pointnet3_out_wgt
        pointnet3_feas = self.pointnet3_fus(pointnet3_feas.reshape(N, 1, -1)).view(N, -1)

        pointnet_feas = torch.cat([pointnet3_feas, pointnet2_feas, pointnet1_feas, points_features[:, :6]], dim=-1)
        return pointnet_feas
