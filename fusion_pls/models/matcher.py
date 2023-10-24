# Modified by Rodrigo Marcuzzi from https://github.com/facebookresearch/Mask2Former
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment as lsa
from torch import nn
from torch.cuda.amp import autocast
from fusion_pls.utils.misc import generalized_box_iou


class InstMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

        For efficiency reasons, the targets don't include the no_object. Because of this, in general,
        there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
        while the others are un-matched (and thus treated as non-objects).
        """

    def __init__(self, cost_cls: float = 1, cost_chm: float = 1, p_ratio: float = 0.4):
        """Creates the matcher

        Params:
            cost_cls: This is the relative weight of the classification error in the matching cost
            cost_chm: This is the relative weight of the L1 error of the center heatmap in the matching cost
        """
        super().__init__()
        self.weight_cls = cost_cls
        self.weight_chm = cost_chm

        assert cost_cls != 0 or cost_chm != 0, "all costs cant be 0"

        self.p_ratio = p_ratio

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_heatmaps": Tensor of dim [batch_size, num_queries, num_pts] with the predicted center heatmap

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "classes": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "heatmaps": Tensor of dim [num_pts, num_target_heatmaps] containing the target center heatmap

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_heatmaps)
        """
        return self.memory_efficient_forward(outputs, targets)

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        """More memory-friendly matching"""
        bs, num_queries, num_classes = outputs["pred_logits"].shape
        indices = []

        # Iterate through batch size
        for b in range(bs):
            out_prob = outputs["pred_logits"][b].softmax(-1)
            tgt_ids = targets["classes"][b].type(torch.int64)

            cost_class = -out_prob[:, tgt_ids]

            out_chm = outputs["pred_heatmaps"][b].permute(1, 0)  # [num_queries, num_pts]
            tgt_chm = targets["heatmaps"][b].to(out_chm)
            n_pts_scan = tgt_chm.shape[1]

            # all heatmaps share the same set of points for efficient matching!
            pt_idx = torch.randint(
                0, n_pts_scan, (int(self.p_ratio * n_pts_scan), 1)
            ).squeeze(1)

            # get gt labels
            tgt_chm = tgt_chm[:, pt_idx]
            out_chm = out_chm[:, pt_idx]

            with autocast(enabled=False):
                out_chm = out_chm.float()  # [num_q,num_pts]
                tgt_chm = tgt_chm.float()  # [n_ins,num_pts]
                cost_chm = batch_smooth_l1_cost(out_chm, tgt_chm)

            # Final cost matrix
            C = (
                    self.weight_chm * cost_chm
                    + self.weight_cls * cost_class
            )
            C = C.reshape(num_queries, -1).cpu()
            indices.append(lsa(C))

        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]


class MaskMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, costs_class: float = 2.0, cost_mask: float = 5.0, cost_dice: float = 5.0, p_ratio: float = 0.4):
        """Creates the matcher

        Params:
            weight_class: This is the relative weight of the classification error in the matching cost
            weight_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            weight_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.weight_class = costs_class
        self.weight_mask = cost_mask
        self.weight_dice = cost_dice

        assert self.weight_class != 0 or self.weight_mask != 0 or self.weight_dice != 0, "all costs cant be 0"

        self.p_ratio = p_ratio

    @torch.no_grad()
    def forward(self, outputs, targets):
        return self.memory_efficient_forward(outputs, targets)

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        """More memory-friendly matching"""
        bs, num_queries, num_classes = outputs["pred_logits"].shape

        indices = []

        # Iterate through batch size
        for b in range(bs):
            out_prob = outputs["pred_logits"][b].softmax(-1)
            tgt_ids = targets["classes"][b].type(torch.int64)

            cost_class = -out_prob[:, tgt_ids]

            out_mask = outputs["pred_masks"][b].permute(1, 0)  # [num_queries, num_pts]
            tgt_mask = targets["masks"][b].to(out_mask)
            n_pts_scan = tgt_mask.shape[1]

            # all masks share the same set of points for efficient matching!
            pt_idx = torch.randint(
                0, n_pts_scan, (int(self.p_ratio * n_pts_scan), 1)
            ).squeeze(1)

            # get gt labels
            tgt_mask = tgt_mask[:, pt_idx]
            out_mask = out_mask[:, pt_idx]

            with autocast(enabled=False):
                out_mask = out_mask.float()  # [num_q,num_pts]
                tgt_mask = tgt_mask.float()  # [n_ins,num_pts]
                cost_mask = batch_sigmoid_ce_cost_jit(out_mask, tgt_mask)
                cost_dice = batch_dice_cost_jit(out_mask, tgt_mask)

            # Final cost matrix
            C = (
                    self.weight_mask * cost_mask
                    + self.weight_class * cost_class
                    + self.weight_dice * cost_dice
            )
            C = C.reshape(num_queries, -1).cpu()
            indices.append(lsa(C))

        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]


def batch_dice_cost(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


batch_dice_cost_jit = torch.jit.script(batch_dice_cost)  # type: torch.jit.ScriptModule


def batch_sigmoid_ce_cost(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    num_pts = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    return loss / num_pts


batch_sigmoid_ce_cost_jit = torch.jit.script(
    batch_sigmoid_ce_cost
)  # type: torch.jit.ScriptModule


def batch_smooth_l1_cost(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs.
                Stores the regression target for each element in inputs.
    Returns:
        Loss tensor
    """
    num_pts = inputs.shape[1]
    inputs = inputs.flatten(1)  # [num_queries, num_pts]
    targets = targets.flatten(1)  # [num_targets, num_pts]
    # compute the L1 loss between each prediction and target,
    # get a [num_queries, num_targets] matrix
    loss = F.smooth_l1_loss(inputs[:, None, :], targets[None, :, :], reduction="none")
    # sum over the targets, and average
    loss = loss.sum(-1) / num_pts

    return loss


batch_smooth_l1_cost_jit = torch.jit.script(
    batch_smooth_l1_cost
)  # type: torch.jit.ScriptModule
