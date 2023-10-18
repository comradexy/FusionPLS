from typing import List

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor


def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def pad_stack(tensor_list: List[Tensor]):
    """
    pad each tensor on the input to the max value in shape[1] and
    concatenate them in a single tensor.
    Input:
        list of tensors [Ni,Pi]
    Output:
        tensor [sum(Ni),max(Pi)]
    """
    _max = max([t.shape[1] for t in tensor_list])
    batched = torch.cat([F.pad(t, (0, _max - t.shape[1])) for t in tensor_list])
    return batched


def sample_points(masks, masks_ids, n_pts, n_samples):
    # select n_pts per mask to focus on instances
    # plus random points up to n_samples
    sampled = []
    for ids, mm in zip(masks_ids, masks):
        m_idx = torch.cat(
            [
                id[torch.randperm(n_pts)[:n_pts]] if id.shape[0] > n_pts else id
                for id in ids
            ]
        )
        r_idx = torch.randint(mm.shape[1], [n_samples - m_idx.shape[0]]).to(m_idx)
        idx = torch.cat((m_idx, r_idx))
        sampled.append(idx)
    return sampled


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [cx, cy, cz, w, l, h, rot] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # do an early check
    assert (boxes1[:, 3:] >= 0).all()
    assert (boxes2[:, 3:] >= 0).all()

    # Calculate the 3D IoU
    iou, union = box_iou_3d(boxes1, boxes2)

    # Calculate the minimum and maximum coordinates of the boxes
    min_coords = torch.min(boxes1[:, None, :3] - boxes1[:, None, 3:6] / 2, boxes2[:, :3] - boxes2[:, 3:6] / 2)
    max_coords = torch.max(boxes1[:, None, :3] + boxes1[:, None, 3:6] / 2, boxes2[:, :3] + boxes2[:, 3:6] / 2)

    # Calculate the width, height, and depth of the intersecting box
    whd = (max_coords - min_coords).clamp(min=0)  # [N, M, 3]

    # Calculate the volume of the intersecting box
    volume = whd[:, :, 0] * whd[:, :, 1] * whd[:, :, 2]

    # Calculate the volume of the smallest enclosing box
    volume1 = boxes1[:, 3] * boxes1[:, 4] * boxes1[:, 5]
    volume2 = boxes2[:, 3] * boxes2[:, 4] * boxes2[:, 5]
    enclosing_volume = volume1[:, None] + volume2 - volume

    # Calculate the GIoU
    giou = iou - (enclosing_volume - union) / enclosing_volume

    return giou


def box_iou_3d(boxes1, boxes2):
    volume1 = boxes1[:, 3] * boxes1[:, 4] * boxes1[:, 5]
    volume2 = boxes2[:, 3] * boxes2[:, 4] * boxes2[:, 5]

    min_coords = torch.max(boxes1[:, None, :3] - boxes1[:, None, 3:6] / 2, boxes2[:, :3] - boxes2[:, 3:6] / 2)
    max_coords = torch.min(boxes1[:, None, :3] + boxes1[:, None, 3:6] / 2, boxes2[:, :3] + boxes2[:, 3:6] / 2)

    whd = (max_coords - min_coords).clamp(min=0)  # [N,M,3]
    inter_volume = whd[:, :, 0] * whd[:, :, 1] * whd[:, :, 2]  # [N,M]

    union_volume = volume1[:, None] + volume2 - inter_volume

    iou = inter_volume / union_volume
    return iou, union_volume
