import os
from os.path import join, abspath

import click
import torch
import yaml
import numpy as np
from PIL import Image
from easydict import EasyDict as edict
from fusion_pls.datasets.semantic_dataset import SemanticDatasetModule
from fusion_pls.models.mask_model import FusionLPS
from pytorch_lightning import Trainer


@click.command()
@click.option("--save", "-S", type=str, default="../output")
@click.option("--ckpt", "-C", type=str, required=True)
@click.option("--dataset", "-d", type=str, default=None)
@click.option("--sequence", "-s", type=int, default=8)
@click.option("--frame", "-f", type=int, default=0)
def main(save, ckpt, dataset, sequence, frame):
    model_cfg = edict(
        yaml.safe_load(open(join(getDir(__file__), "../config/model.yaml")))
    )
    backbone_cfg = edict(
        yaml.safe_load(open(join(getDir(__file__), "../config/backbone.yaml")))
    )
    decoder_cfg = edict(
        yaml.safe_load(open(join(getDir(__file__), "../config/decoder.yaml")))
    )
    cfg = edict({**model_cfg, **backbone_cfg, **decoder_cfg})

    cfg.BACKBONE.PRETRAINED = None
    cfg.BACKBONE.MINK.PRETRAINED = None
    cfg.BACKBONE.CPE.PRETRAINED = None

    if save is not None:
        results_dir = create_dirs(save)
        print(f"Saving predictions in directory {abspath(results_dir)}")
        cfg.RESULTS_DIR = results_dir

    if dataset is not None:
        cfg.KITTI.PATH = dataset
        cfg.KITTI.CONFIG = "../datasets/semantic-kitti.yaml"

    sample = SampleLoader(dataset, sequence, frame)
    model = FusionLPS(cfg)
    weights = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(weights["state_dict"])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.test_step(sample, 0)
    print("Done!")


def getDir(obj):
    return os.path.dirname(os.path.abspath(obj))


def create_dirs(save):
    results_dir = join(
        getDir(__file__),
        save,
    )
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)
    return results_dir


def SampleLoader(dataset, sequence, frame):
    seq_path = join(dataset, "sequences", str(sequence).zfill(2))
    pts_path = join(seq_path, "velodyne_fov_multi", str(frame).zfill(6) + ".bin")
    img_path = join(seq_path, "image_2", str(frame).zfill(6) + ".png")
    pose_path = join(seq_path, "poses.txt")
    calib_path = join(seq_path, "calib.txt")

    points = np.memmap(
        pts_path,
        dtype=np.float32,
        mode="r",
    ).reshape((-1, 7))
    xyz = points[:, :3]
    intensity = points[:, 3]
    rgb = points[:, 4:7]
    feats = np.concatenate(
        (xyz.reshape(-1, 3),
         intensity.reshape(-1, 1),
         rgb.reshape(-1, 3)),
        axis=1
    )
    image = Image.open(img_path)
    image = np.array(image)
    pose, calib = load_pose_and_calib(pose_path, calib_path, frame)

    sample = {
        "pt_coord": [xyz],
        "feats": [feats],
        "image": [image],
        "sem_label": None,
        "ins_label": None,
        "masks": None,
        "masks_cls": None,
        "masks_ids": None,
        "fname": [pts_path.replace("velodyne_fov_multi", "velodyne")],
        "calib": [calib],
        "pose": [pose],
        "token": [],
    }

    return sample


def load_pose_and_calib(pose_path, calib_pth, frame):
    calib = parse_calibration(calib_pth)
    pose_f64 = parse_pose(pose_path, calib, frame)
    pose = pose_f64.astype(np.float32)
    return pose, calib


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


def parse_pose(filename, calib, frame):
    file = open(filename)
    pose = None
    Tr = calib["Tr"]
    Tr_inv = np.linalg.inv(Tr)
    for i, line in enumerate(file):
        if i == int(frame):
            values = [float(v) for v in line.strip().split()]
            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0
            pose = np.matmul(Tr_inv, np.matmul(pose, Tr))
            break
    return pose


if __name__ == "__main__":
    main()
