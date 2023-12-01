import os

import numpy as np
import torch
import yaml
from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from fusion_pls.utils.img2pcd import img_feat_proj, get_map_img2pcd


class SemanticDatasetModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.things_ids = []
        self.color_map = []
        self.label_names = []
        self.train_mask_set = None
        self.val_mask_set = None
        self.test_mask_set = None

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        train_set = SemanticDataset(
            cfg=self.cfg,
            split="train",
        )
        self.train_mask_set = MaskSemanticDataset(
            dataset=train_set,
            split="train",
            min_pts=self.cfg[self.cfg.MODEL.DATASET].MIN_POINTS,
            space=self.cfg[self.cfg.MODEL.DATASET].SPACE,
            sub_pts=self.cfg[self.cfg.MODEL.DATASET].SUB_NUM_POINTS,
            subsample=self.cfg.TRAIN.SUBSAMPLE,
            aug=self.cfg.TRAIN.AUG,
        )

        val_set = SemanticDataset(
            cfg=self.cfg,
            split="valid",
        )
        self.val_mask_set = MaskSemanticDataset(
            dataset=val_set,
            split="valid",
            min_pts=self.cfg[self.cfg.MODEL.DATASET].MIN_POINTS,
            space=self.cfg[self.cfg.MODEL.DATASET].SPACE,
        )

        test_set = SemanticDataset(
            cfg=self.cfg,
            split="test",
        )
        self.test_mask_set = MaskSemanticDataset(
            dataset=test_set,
            split="test",
            min_pts=self.cfg[self.cfg.MODEL.DATASET].MIN_POINTS,
            space=self.cfg[self.cfg.MODEL.DATASET].SPACE,
        )

        self.things_ids = train_set.things_ids
        self.color_map = train_set.color_map
        self.label_names = train_set.label_names

    def train_dataloader(self):
        batch_size = self.cfg.TRAIN.BATCH_SIZE
        dataset = self.train_mask_set
        collate_fn = BatchCollation(dataset, batch_size)
        self.train_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=True,
            num_workers=self.cfg.TRAIN.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
            timeout=0,
        )
        return self.train_loader

    def val_dataloader(self):
        batch_size = self.cfg.TRAIN.BATCH_SIZE
        dataset = self.val_mask_set
        collate_fn = BatchCollation(dataset, batch_size)
        self.valid_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=False,
            num_workers=self.cfg.TRAIN.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
            timeout=0,
        )
        return self.valid_loader

    def test_dataloader(self):
        batch_size = self.cfg.TRAIN.BATCH_SIZE
        dataset = self.test_mask_set
        collate_fn = BatchCollation(dataset, batch_size)
        self.test_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=False,
            num_workers=self.cfg.TRAIN.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
            timeout=0,
        )
        return self.test_loader


class SemanticDataset(Dataset):
    def __init__(self, cfg, split="train"):
        self.cfg = cfg
        self.dataset = cfg.MODEL.DATASET
        data_path = cfg[self.dataset].PATH + "/sequences/"
        yaml_path = cfg[self.dataset].CONFIG
        self.in_camera_fov = cfg[self.dataset].IN_CAMERA_FOV
        self.img_mean, self.img_std = cfg[self.dataset].IMG_NORM_PARAMS

        with open(yaml_path, "r") as stream:
            semyaml = yaml.safe_load(stream)

        self.things = get_things(self.dataset)
        self.stuff = get_stuff(self.dataset)

        self.label_names = {**self.things, **self.stuff}
        self.things_ids = get_things_ids(self.dataset)

        self.color_map = semyaml["color_map_learning"]
        self.labels = semyaml["labels"]
        self.learning_map = semyaml["learning_map"]
        self.inv_learning_map = semyaml["learning_map_inv"]
        self.split = split
        split = semyaml["split"][self.split]

        self.im_idx = []
        pose_files = []
        calib_files = []
        fill = 2 if self.dataset == "KITTI" else 4
        for i_folder in split:
            self.im_idx += absoluteFilePaths(
                "/".join([data_path, str(i_folder).zfill(fill), "velodyne"])
            )
            pose_files.append(
                absoluteDirPath(
                    "/".join([data_path, str(i_folder).zfill(fill), "poses.txt"])
                )
            )
            calib_files.append(
                absoluteDirPath(
                    "/".join([data_path, str(i_folder).zfill(fill), "calib.txt"])
                )
            )

        self.im_idx.sort()
        poses, calibs = load_poses_and_calibs(pose_files, calib_files)
        self.poses = poses
        self.calibs = calibs

    def __len__(self):
        return len(self.im_idx)

    def __getitem__(self, index):
        fname = self.im_idx[index]
        pose = self.poses[index]
        calib = self.calibs[index]

        # points feats: xyzi
        points = np.memmap(
            self.im_idx[index],
            dtype=np.float32,
            mode="r",
        ).reshape((-1, 4))
        xyz = points[:, :3]
        intensity = points[:, 3]

        image = Image.open(
            self.im_idx[index].replace("velodyne", "image_2")[:-3] + "png"
        )
        image = np.array(image, dtype=np.float32, copy=False) / 255.
        img_mean = np.asarray(self.img_mean, dtype=np.float32)
        img_std = np.asarray(self.img_mean, dtype=np.float32)
        image = (image - img_mean) / img_std  # [H, W, C]
        image = np.transpose(image, (2, 0, 1))  # [H, W, C] -> [C, H, W]

        if len(intensity.shape) == 2:
            intensity = np.squeeze(intensity)
        if self.split == "test":
            annotated_data = np.expand_dims(
                np.zeros_like(points[:, 0], dtype=int), axis=1
            )
            sem_labels = annotated_data
            ins_labels = annotated_data
        else:
            annotated_data = np.memmap(
                self.im_idx[index].replace("velodyne", "labels")[:-3] + "label",
                dtype=np.int32,
                mode="r",
            ).reshape((-1, 1))
            sem_labels = annotated_data & 0xFFFF
            ins_labels = annotated_data >> 16
            sem_labels = np.vectorize(self.learning_map.__getitem__)(sem_labels)

        return (
            xyz,
            intensity,
            image,
            sem_labels,
            ins_labels,
            fname,
            calib,
            pose,
        )


class MaskSemanticDataset(Dataset):
    def __init__(
            self,
            dataset,
            split,
            min_pts,
            space,
            sub_pts=0,
            subsample=False,
            aug=False,
    ):
        self.dataset = dataset
        self.in_camera_fov = dataset.in_camera_fov
        self.sub_pts = sub_pts
        self.split = split
        self.min_points = min_pts
        self.aug = aug
        self.subsample = subsample
        self.th_ids = dataset.things_ids
        self.xlim = space[0]
        self.ylim = space[1]
        self.zlim = space[2]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        empty = True
        while empty:
            data = self.dataset[index]
            xyz, intensity, image, sem_labels, ins_labels, fname, calib, pose = data
            keep = np.argwhere(
                (self.xlim[0] < xyz[:, 0])
                & (xyz[:, 0] < self.xlim[1])
                & (self.ylim[0] < xyz[:, 1])
                & (xyz[:, 1] < self.ylim[1])
                & (self.zlim[0] < xyz[:, 2])
                & (xyz[:, 2] < self.zlim[1])
            )[:, 0]
            xyz = xyz[keep]
            intensity = intensity[keep]
            sem_labels = sem_labels[keep]
            ins_labels = ins_labels[keep]

            # skip scans without instances in train set
            if self.split != "train":
                empty = False
                break

            if len(np.unique(ins_labels)) == 1:
                empty = True
                index = np.random.randint(0, len(self.dataset))
            else:
                empty = False

        feats = np.concatenate(
            (
                xyz.reshape(-1, 3),
                intensity.reshape(-1, 1),
            ),
            axis=1,
        )

        if self.split == "test":
            return (
                xyz,
                feats,
                np.array([]),
                np.array([]),
                np.array([]),
                torch.tensor([]),
                torch.tensor([]),
                torch.tensor([]),
                torch.tensor([]),
                {},
                {},
                fname,
                calib,
                pose,
            )

        # Subsample
        if self.split == "train" and self.subsample and len(xyz) > self.sub_pts:
            idx = np.random.choice(np.arange(len(xyz)), self.sub_pts, replace=False)
            xyz = xyz[idx]
            feats = feats[idx]
            sem_labels = sem_labels[idx]
            ins_labels = ins_labels[idx]

        map_img2pcd, indices = get_map_img2pcd(xyz, tuple(image.shape[1:3]), calib)

        # Augmentations
        # xyz contains original points coordinates
        # feats contains augmented coordinates and other textural features
        if self.split == "train" and self.aug:
            # xyz = pcd_augmentations(xyz)
            feats = pcd_augmentations(feats)
            xyz = feats[:, :3]

        # get decoder labels
        dec_lab = self.get_decoder_labels(
            xyz, sem_labels, ins_labels
        )

        if self.in_camera_fov:
            # get feats and coords in camera fov
            feats = feats[indices]
            xyz = xyz[indices]
            # get sem_labels and ins_labels in camera fov
            sem_labels = sem_labels[indices]
            ins_labels = ins_labels[indices]
            # get decoder labels in camera fov
            dec_lab = self.get_decoder_labels(
                xyz, sem_labels, ins_labels
            )

        # assert dec_lab["things_masks"].shape[0] != 0, \
        #     "things_masks is empty," \
        #     f"file path: {fname}"

        if dec_lab["things_masks"].shape[0] == 0:
            return None

        return (
            xyz,
            feats,
            image,  # normalized image
            map_img2pcd,  # mapping from image to pcd, shape=[N, 2]
            indices,  # indices of points in image
            sem_labels,
            ins_labels,
            dec_lab,
            fname,  # file path of pcd
            calib,
            pose,
        )

    def get_decoder_labels(self, xyz, sem_labels, ins_labels):
        stuff_masks = np.array([]).reshape(0, xyz.shape[0])
        stuff_masks_ids = []
        things_masks = np.array([]).reshape(0, xyz.shape[0])
        things_cls = np.array([], dtype=int)
        things_masks_ids = []

        stuff_labels = np.asarray(
            [0 if s in self.th_ids else s for s in sem_labels[:, 0]]
        )
        stuff_cls, st_cnt = np.unique(stuff_labels, return_counts=True)
        # filter small masks
        keep_st = np.argwhere(st_cnt > self.min_points)[:, 0]
        stuff_cls = stuff_cls[keep_st][1:]
        if len(stuff_cls):
            stuff_masks = np.array(
                [np.where(stuff_labels == i, 1.0, 0.0) for i in stuff_cls]
            )
            stuff_masks_ids = [
                torch.from_numpy(np.where(m == 1)[0]) for m in stuff_masks
            ]
        # things masks
        ins_sems = np.where(ins_labels == 0, 0, sem_labels)
        _ins_labels = ins_sems + ((ins_labels << 16) & 0xFFFF0000).reshape(-1, 1)
        things_ids, th_idx, th_cnt = np.unique(
            _ins_labels[:, 0], return_index=True, return_counts=True
        )
        # filter small instances
        keep_th = np.argwhere(th_cnt > self.min_points)[:, 0]
        things_ids = things_ids[keep_th]
        th_idx = th_idx[keep_th]
        # remove instances with wrong sem class
        keep_th = np.array(
            [i for i, idx in enumerate(th_idx) if sem_labels[idx] in self.th_ids],
            dtype=int,
        )
        things_ids = things_ids[keep_th]
        th_idx = th_idx[keep_th]
        if len(th_idx):
            things_masks = np.array(
                [np.where(_ins_labels[:, 0] == i, 1.0, 0.0) for i in things_ids]
            )
            things_masks_ids = [
                torch.from_numpy(np.where(m == 1)[0]) for m in things_masks
            ]
            things_cls = np.array([sem_labels[i] for i in th_idx]).squeeze(1)

        masks = torch.from_numpy(np.concatenate((stuff_masks, things_masks)))
        masks_cls = torch.from_numpy(np.concatenate((stuff_cls, things_cls)))
        stuff_masks_ids.extend(things_masks_ids)
        masks_ids = stuff_masks_ids

        assert (
                masks.shape[0] == masks_cls.shape[0]
        ), f"not same number masks and classes: masks {masks.shape[0]}, classes {masks_cls.shape[0]} "

        # get things offsets
        things_off = np.zeros((len(things_masks), xyz.shape[0], 3))
        for i, mask in enumerate(things_masks):
            # get instance xyz
            _xyz = xyz[mask.astype(bool)]
            # generate offset to center of instance
            center = np.mean(_xyz, axis=0)
            offset = _xyz - center
            # normalize offset
            max_x, max_y, max_z = np.max(offset, axis=0)
            min_x, min_y, min_z = np.min(offset, axis=0)
            offset = offset / np.array([max_x - min_x, max_y - min_y, max_z - min_z])
            things_off[i, mask.astype(bool)] = offset
        things_cls = torch.from_numpy(things_cls)
        things_off = torch.from_numpy(things_off)
        things_masks = torch.from_numpy(things_masks)

        outputs = {
            "masks": masks,
            "masks_cls": masks_cls,
            "masks_ids": masks_ids,
            "things_off": things_off,
            "things_cls": things_cls,
            "things_masks": things_masks,
            "things_masks_ids": things_masks_ids,
        }

        return outputs


class BatchCollation:
    def __init__(self, dataset, batch_size):
        self.keys = [
            "pt_coord",
            "feats",
            "image",
            "map_img2pcd",
            "indices",
            "sem_label",
            "ins_label",
            "dec_lab",
            "fname",
            "calib",
            "pose",
        ]
        self.dataset = dataset
        self.batch_size = batch_size

    def __call__(self, data):
        # return {self.keys[i]: list(x) for i, x in enumerate(zip(*data))}

        filtered_data = list(filter(lambda x: x is not None, data))
        if len(filtered_data) < self.batch_size:
            # get additional samples from dataset
            num_samples = self.batch_size - len(filtered_data)
            add_samples = []
            for i in range(num_samples):
                s = self.dataset[np.random.randint(len(self.dataset))]
                while s is None:
                    s = self.dataset[np.random.randint(len(self.dataset))]
                add_samples.append(s)
            filtered_data.extend(add_samples)

        return {self.keys[i]: list(x) for i, x in enumerate(zip(*filtered_data))}


def absoluteFilePaths(directory):
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))


def absoluteDirPath(directory):
    return os.path.abspath(directory)


def pcd_augmentations(feats):
    """
    Note:
        Augment point cloud data with random rotation, flip and translation
    Args:
        feats: np.array, shape=[N, C], xyz coordinates and other features
    """
    # get xyz coordinates
    xyz = feats[:, 0:3]
    # rotation
    rotate_rad = np.deg2rad(np.random.random() * 360)
    c, s = np.cos(rotate_rad), np.sin(rotate_rad)
    j = np.matrix([[c, s], [-s, c]])
    xyz[:, :2] = np.dot(xyz[:, :2], j)

    # flip
    flip_type = np.random.choice(4, 1)
    if flip_type == 1:
        xyz[:, 0] = -xyz[:, 0]
    elif flip_type == 2:
        xyz[:, 1] = -xyz[:, 1]
    elif flip_type == 3:
        xyz[:, 0] = -xyz[:, 0]
        xyz[:, 1] = -xyz[:, 1]
    # scale
    noise_scale = np.random.uniform(0.95, 1.05)
    xyz[:, 0] = noise_scale * xyz[:, 0]
    xyz[:, 1] = noise_scale * xyz[:, 1]
    # transform
    trans_std = [0.1, 0.1, 0.1]
    noise_translate = np.array(
        [
            np.random.normal(0, trans_std[0], 1),
            np.random.normal(0, trans_std[1], 1),
            np.random.normal(0, trans_std[2], 1),
        ]
    ).T
    xyz[:, 0:3] += noise_translate
    # replace feats with augmented xyz
    feats[:, 0:3] = xyz

    return feats


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


def parse_poses(filename, calibration):
    file = open(filename)
    poses = []
    Tr = calibration["Tr"]
    Tr_inv = np.linalg.inv(Tr)
    for line in file:
        values = [float(v) for v in line.strip().split()]
        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0
        poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))
    return poses


def load_poses_and_calibs(pose_files, calib_files):
    poses = []
    calibs = []
    for i in range(len(pose_files)):
        calib = parse_calibration(calib_files[i])
        seq_poses_f64 = parse_poses(pose_files[i], calib)
        seq_poses = [pose.astype(np.float32) for pose in seq_poses_f64]
        poses += seq_poses
        calibs += [calib] * len(seq_poses)
    return poses, calibs


def load_calibs(calib_files):
    calibs = []
    for i in range(len(calib_files)):
        calib = parse_calibration(calib_files[i])
        calibs.append(calib)
    return calibs


def getDir(obj):
    return os.path.dirname(os.path.abspath(obj))


def get_things(dataset):
    if dataset == "KITTI":
        things = {
            1: "car",
            2: "bicycle",
            3: "motorcycle",
            4: "truck",
            5: "other-vehicle",
            6: "person",
            7: "bicyclist",
            8: "motorcyclist",
        }
    elif dataset == "NUSCENES":
        things = {
            2: "bycicle",
            3: "bus",
            4: "car",
            5: "construction_vehicle",
            6: "motorcycle",
            7: "pedestrian",
            9: "trailer",
            10: "truck",
        }
    return things


def get_stuff(dataset):
    if dataset == "KITTI":
        stuff = {
            9: "road",
            10: "parking",
            11: "sidewalk",
            12: "other-ground",
            13: "building",
            14: "fence",
            15: "vegetation",
            16: "trunk",
            17: "terrain",
            18: "pole",
            19: "traffic-sign",
        }
    elif dataset == "NUSCENES":
        stuff = {
            1: "barrier",
            8: "traffic_cone",
            11: "driveable_surface",
            12: "other_flat",
            13: "sidewalk",
            14: "terrain",
            15: "manmade",
            16: "vegetation",
        }
    return stuff


def get_things_ids(dataset):
    if dataset == "KITTI":
        return [1, 2, 3, 4, 5, 6, 7, 8]
