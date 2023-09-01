import MinkowskiEngine as ME
import os
from os.path import join
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from easydict import EasyDict as edict
from fusion_pls.utils.interpolate import knn_up
from fusion_pls.models.mink import MinkEncoderDecoder
from fusion_pls.models.color_encoder import ColorPointEncoder
from fusion_pls.models.backbone import FusionEncoder
from fusion_pls.models.mask_model import MaskPS
from fusion_pls.datasets.semantic_dataset import SemanticDatasetModule


def getDir(obj):
    return os.path.dirname(os.path.abspath(obj))


def pcd_painting(pts, img, Trv2c, P2):
    """Paint points with image.

    Note:
        This function is for KITTI only.

    Args:
        pts (np.ndarray, shape=[N, 3]): Coordinates of points.
        img (np.ndarray, shape=[H, W, 3]): Image.
        Trv2c (np.ndarray, shape=[4, 4]): Matrix to project points in
            camera coordinate to lidar coordinate.
        P2 (p.array, shape=[4, 4]): Intrinsics of Camera2.

    Returns:
        colors: np.ndarray, shape=[N, 3]: RGB colors of points.
    """
    # Convert points from lidar coordinate to camera coordinate
    pts_cam = lidar_to_camera(pts, Trv2c, P2)
    # Convert points from camera coordinate to image coordinate
    pts_img = pts_cam[:, :2] / pts_cam[:, 2:]
    # Convert image coordinate to pixel coordinates
    pts_img = pts_img.astype(np.int32)
    # Get RGB colors from image
    image = Image.fromarray(img)
    colors = []
    for pt in pts_img:
        x, y = pt
        x = min(max(x, 0), img.shape[1] - 1)
        y = min(max(y, 0), img.shape[0] - 1)
        rgb = image.getpixel((x, y))
        colors.append(rgb)
    colors = np.array(colors)
    return colors


def lidar_to_camera(points, velo2cam, P2):
    """Convert points in lidar coordinate to camera coordinate.

    Args:
        points (np.ndarray, shape=[N, 3]): Points in lidar coordinate.
        velo2cam (np.ndarray, shape=[4, 4]): Matrix to project points in
            lidar coordinate to camera coordinate.
        P2 (np.ndarray, shape=[4, 4]): Intrinsics of Camera2.

    Returns:
        np.ndarray, shape=[N, 3]: Points in camera coordinate.
    """
    points_shape = list(points.shape[0:-1])
    if velo2cam.shape != (4, 4):
        velo2cam = np.concatenate(
            [velo2cam, np.array([[0, 0, 0, 1.]], dtype=np.float32)], axis=0)
    if points.shape[-1] == 3:
        points = np.concatenate([points, np.ones(points_shape + [1])], axis=-1)
    cam_points = points @ velo2cam.T @ P2.T
    return cam_points[..., :3]


def parse_calibration(filename):
    calib = {}
    calib_file = open(filename)
    for line in calib_file:
        key, content = line.strip().split(":")
        values = [float(v) for v in content.strip().split()]
        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0
        calib[key] = pose
    calib_file.close()
    return calib


def TensorField(x, res):
    """
    Build a tensor field from coordinates and features from the
    input batch
    The coordinates are quantized using the provided resolution
    """
    # fuse feats and rgb
    feats = torch.from_numpy(np.concatenate(x["feats"], 0)).float()
    rgb = torch.from_numpy(np.concatenate(x["rgb"], 0)).float()
    features = torch.cat([feats, rgb], dim=1)
    # features = torch.from_numpy(np.concatenate(x["feats"], 0)).float()
    coordinates = ME.utils.batched_coordinates(
        [i / res for i in x["pt_coord"]], dtype=torch.float32
    )

    feat_tfield = ME.TensorField(
        features=features,
        coordinates=coordinates,
        quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
        minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
        device="cuda",
    )

    return feat_tfield


if __name__ == '__main__':
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(device)
    model_cfg = edict(
        yaml.safe_load(open(join(getDir(__file__), "./config/model.yaml")))
    )
    backbone_cfg = edict(
        yaml.safe_load(open(join(getDir(__file__), "./config/backbone.yaml")))
    )
    decoder_cfg = edict(
        yaml.safe_load(open(join(getDir(__file__), "./config/decoder.yaml")))
    )
    cfg = edict({**model_cfg, **backbone_cfg, **decoder_cfg})

    # backbone = FusionEncoder(cfg.BACKBONE, cfg[cfg.MODEL.DATASET])
    # backbone = backbone.to(device)

    # cpe = ColorPointEncoder(cfg.BACKBONE.CPE, cfg[cfg.MODEL.DATASET])
    # cpe = cpe.to(device)

    # model = MaskPS(cfg)
    # model.to(device)
    cfg.KITTI.PATH = '/data/dxy/SemanticKITTI_Fov/dataset'
    data = SemanticDatasetModule(cfg)
    data.setup()
    # 获取test DataLoader
    test_loader = data.train_dataloader()
    # 从DataLoader中获取迭代器
    test_iter = iter(test_loader)
    # # 通过迭代器遍历所有样本
    # for i in tqdm(test_iter, desc='Vist dataloader'):
    #     pass

    sample = next(test_iter)

    # xyz = sample['pt_coord']
    # feats = sample['feats']
    # rgb = sample['rgb']
    # intensity = [feats[0][:, -1:]]
    # print(f'xyz: type--{type(xyz[0])}; shape--{xyz[0].shape}')
    # print(f'feats: type--{type(feats[0])}; shape--{feats[0].shape}')
    # print(f'rgb: type--{type(rgb[0])}; shape--{rgb[0].shape}')
    # print(f'intensity: type--{type(intensity[0])}; shape--{intensity[0].shape}')

    # xyz = torch.from_numpy(xyz[0]).to(device).reshape(1, -1, 3)
    # rgb = torch.from_numpy(rgb[0]).to(device).reshape(1, -1, 3)
    # intensity = torch.from_numpy(intensity[0]).to(device).reshape(1, -1, 1)
    # color_feats = color_encoder(rgb, intensity, xyz)
    # print(f'color_feats: type--{type(color_feats)}; shape--{color_feats.shape}')

    # feats, coors = cpe(sample)
    # for i in range(len(feats)):
    #     print(f'level{i}:')
    #     for j in range(len(feats[i])):
    #         print(f'  feats_batch{j}: shape--{feats[i][j].shape}')
    # for i in range(len(coors)):
    #     print(f'level{i}:')
    #     for j in range(len(coors[i])):
    #         print(f'  coors_batch{j}: shape--{coors[i][j].shape}')

    # outputs, padding, bb_logits = model(sample)
    # print(outputs['pred_logits'].shape)
    # print(outputs['pred_masks'].shape)
    # for i in range(len(outputs['aux_outputs'])):
    #     print(f'aux_outputs{i}:')
    #     for k, v in outputs['aux_outputs'][i].items():
    #         print(f'  {k}: shape--{v.shape}')
    # print(f"padding.shape: {padding.shape}")
    # print(f"bb_logits.shape: {bb_logits.shape}")

    print(sample.keys())
    print(sample['pt_coord'][0].shape)
    print(sample['feats'][0].shape)
