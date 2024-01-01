import fusion_pls.utils.testing as testing
import MinkowskiEngine as ME
import torch
import torch.nn as nn
import torch.nn.functional as F
from fusion_pls.models.decoder import PanopticMaskDecoder
from fusion_pls.models.loss import MaskLoss, OffLoss, SemLoss, DistillLoss
from fusion_pls.models.backbone import FusionEncoder, PcdEncoder
from fusion_pls.datasets.semantic_dataset import get_things_ids
from fusion_pls.utils.evaluate_panoptic import PanopticEvaluator
from pytorch_lightning.core.lightning import LightningModule


class FusionLPS(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(dict(hparams))
        self.cfg = hparams

        self.backbone = nn.ModuleDict()

        self.enable_kd = hparams.MODEL.ENABLE_KD
        # use pcd encoder as student
        hparams.BACKBONE.PCD.FREEZE = (not self.enable_kd) and hparams.BACKBONE.PCD.FREEZE
        print(f"pcd_bb freeze: {hparams.BACKBONE.PCD.FREEZE}.")
        backbone_s = PcdEncoder(hparams.BACKBONE, hparams[hparams.MODEL.DATASET])
        self.backbone.update({"student": backbone_s})
        if self.enable_kd:
            # use fusion encoder as teacher
            hparams.BACKBONE.PCD.FREEZE = self.enable_kd and hparams.BACKBONE.IMG.FREEZE
            print(f"img_bb freeze: {hparams.BACKBONE.PCD.FREEZE}.")
            backbone_t = FusionEncoder(hparams.BACKBONE, hparams[hparams.MODEL.DATASET])
            self.backbone.update({"teacher": backbone_t})

        ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(self.backbone)

        self.enable_inst_dec = hparams.DECODER.INSTANCE.ENABLE
        self.enable_sem_dec = hparams.DECODER.SEMANTIC.ENABLE
        self.decoder = PanopticMaskDecoder(
            self.backbone["student"].out_dim,
            hparams.DECODER,
            hparams[hparams.MODEL.DATASET],
        )

        self.mask_loss = MaskLoss(hparams.LOSS, hparams[hparams.MODEL.DATASET])
        self.off_loss = OffLoss(hparams.LOSS, hparams[hparams.MODEL.DATASET])
        self.sem_loss = SemLoss(hparams.LOSS)
        self.kd_loss = DistillLoss(hparams.LOSS)

        self.evaluator = PanopticEvaluator(
            hparams[hparams.MODEL.DATASET], hparams.MODEL.DATASET
        )

    def forward(self, x):
        feats, coords, pad_masks, feats_s = self.backbone["student"](x, self.enable_kd)
        outputs, padding = self.decoder(feats, coords, pad_masks)
        return outputs, padding, feats_s

    def get_loss(self, x, outputs, padding, feats_t=None, feats_s=None):
        losses = {}

        dec_labels = x["dec_lab"]
        sem_labels = [
            torch.from_numpy(i).type(torch.LongTensor).cuda()
            for i in x["sem_label"]
        ]
        sem_labels = torch.cat([s.squeeze(1) for s in sem_labels], dim=0)

        masks = [b["masks"] for b in dec_labels]
        masks_cls = [b["masks_cls"] for b in dec_labels]
        masks_ids = [b["masks_ids"] for b in dec_labels]
        # things_cls = [b["things_cls"] for b in dec_labels]
        things_off = [b["things_off"] for b in dec_labels]
        # things_masks = [b["things_masks"] for b in dec_labels]
        things_masks_ids = [b["things_masks_ids"] for b in dec_labels]
        mask_targets = {"classes": masks_cls, "masks": masks}

        # calculate semantic decoder loss
        if self.enable_sem_dec:
            # loss_sem = self.mask_loss(outputs["sem_outputs"], mask_targets, masks_ids)
            loss_sem = self.sem_loss.get_dec_loss(outputs["sem_outputs"], sem_labels, padding)
            loss_sem = {
                f"sem_{k}": v for k, v in loss_sem.items()
            }
            losses.update(loss_sem)

        # calculate instance decoder loss
        if self.enable_inst_dec:
            inst_off_targets = {"offsets": things_off}
            # inst_mask_targets = {"classes": things_cls, "masks": things_masks}
            loss_inst = self.off_loss(outputs["inst_outputs"], inst_off_targets, things_masks_ids)
            loss_inst.update(self.mask_loss(outputs["inst_outputs"], mask_targets, masks_ids))
            loss_inst = {
                f"inst_{k}": v for k, v in loss_inst.items()
            }
            losses.update(loss_inst)

        # calculate panoptic decoder loss
        loss_pan = self.mask_loss(outputs["pan_outputs"], mask_targets, masks_ids)
        loss_pan = {
            f"pan_{k}": v for k, v in loss_pan.items()
        }
        losses.update(loss_pan)

        # calculate KD loss
        if feats_t is not None and feats_s is not None:
            loss_kd = self.kd_loss(feats_t, feats_s)
            losses.update(loss_kd)

        return losses

    def training_step(self, x: dict, idx):
        outputs, padding, feats_s = self.forward(x)

        if self.enable_kd:
            feats_t, _, _ = self.backbone["teacher"](x)
        else:
            feats_t = None

        loss_dict = self.get_loss(x, outputs, padding, feats_t, feats_s)
        for k, v in loss_dict.items():
            self.log(f"train/{k}", v, batch_size=self.cfg.TRAIN.BATCH_SIZE)
        total_loss = sum(loss_dict.values())
        self.log("train_loss", total_loss, batch_size=self.cfg.TRAIN.BATCH_SIZE)
        torch.cuda.empty_cache()

        return total_loss

    def validation_step(self, x: dict, idx):
        if "EVALUATE" in self.cfg:
            self.evaluation_step(x, idx)
            return

        outputs, padding, feats_s = self.forward(x)

        if self.enable_kd:
            feats_t, _, _ = self.backbone["teacher"](x)
        else:
            feats_t = None

        loss_dict = self.get_loss(x, outputs, padding, feats_t, feats_s)
        for k, v in loss_dict.items():
            self.log(f"val/{k}", v, batch_size=self.cfg.TRAIN.BATCH_SIZE)
        total_loss = sum(loss_dict.values())
        self.log("val_loss", total_loss, batch_size=self.cfg.TRAIN.BATCH_SIZE)

        sem_pred, ins_pred = self.panoptic_inference(outputs["pan_outputs"], padding)

        self.evaluator.update(sem_pred, ins_pred, x)

        torch.cuda.empty_cache()
        return total_loss

    def validation_epoch_end(self, outputs):
        bs = self.cfg.TRAIN.BATCH_SIZE
        self.log("metrics/pq", self.evaluator.get_mean_pq(), batch_size=bs)
        self.log("metrics/iou", self.evaluator.get_mean_iou(), batch_size=bs)
        self.log("metrics/rq", self.evaluator.get_mean_rq(), batch_size=bs)
        self.log("metrics/pq_dagger", self.evaluator.get_mean_pq_dagger(), batch_size=bs)
        self.log("metrics/sq", self.evaluator.get_mean_sq(), batch_size=bs)
        self.log("metrics/pq_stuff", self.evaluator.get_mean_pq_stuff(), batch_size=bs)
        self.log("metrics/rq_stuff", self.evaluator.get_mean_rq_stuff(), batch_size=bs)
        self.log("metrics/sq_stuff", self.evaluator.get_mean_sq_stuff(), batch_size=bs)
        self.log("metrics/pq_things", self.evaluator.get_mean_pq_things(), batch_size=bs)
        self.log("metrics/rq_things", self.evaluator.get_mean_rq_things(), batch_size=bs)
        self.log("metrics/sq_things", self.evaluator.get_mean_sq_things(), batch_size=bs)
        if not ("EVALUATE" in self.cfg):
            self.evaluator.reset()

    def evaluation_step(self, x: dict, idx):
        outputs, padding, _ = self.forward(x)
        sem_pred, ins_pred = self.panoptic_inference(outputs["pan_outputs"], padding)

        if "RESULTS_DIR" in self.cfg:
            results_dir = self.cfg.RESULTS_DIR
            class_inv_lut = self.evaluator.get_class_inv_lut()
            testing.save_results(
                sem_pred, ins_pred, results_dir, x, class_inv_lut,
            )

        self.evaluator.update(sem_pred, ins_pred, x)
        torch.cuda.empty_cache()

    def test_step(self, x: dict, idx):
        outputs, padding, _ = self.forward(x)
        sem_pred, ins_pred = self.panoptic_inference(outputs["pan_outputs"], padding)

        if "RESULTS_DIR" in self.cfg:
            results_dir = self.cfg.RESULTS_DIR
            class_inv_lut = self.evaluator.get_class_inv_lut()
            testing.save_results(
                sem_pred, ins_pred, results_dir, x, class_inv_lut,
            )
        torch.cuda.empty_cache()

    def configure_optimizers(self):
        # pcd_enc_params = list(self.backbone.pcd_enc.parameters())
        # other_params = list(self.backbone.img_enc.parameters()) + \
        #                list(self.backbone.fusion.parameters()) + \
        #                list(self.decoder.parameters())
        # optimizer = torch.optim.AdamW([
        #     {'params': pcd_enc_params, 'lr': 0.01 * self.cfg.TRAIN.LR},
        #     {'params': other_params, 'lr': self.cfg.TRAIN.LR}
        # ])
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.TRAIN.LR
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.cfg.TRAIN.STEP, gamma=self.cfg.TRAIN.DECAY
        )
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(
        #     optimizer, gamma=self.cfg.TRAIN.DECAY
        # )
        return [optimizer], [scheduler]

    def semantic_inference(self, outputs, padding):
        mask_cls = outputs["pred_logits"]
        mask_pred = outputs["pred_masks"]
        semseg = []
        for mask_cls, mask_pred, pad in zip(mask_cls, mask_pred, padding):
            mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
            mask_pred = mask_pred[~pad].sigmoid()  # throw padding points
            pred = torch.einsum("qc,pq->pc", mask_cls, mask_pred)
            semseg.append(torch.argmax(pred, dim=1))
        return semseg

    def panoptic_inference(self, outputs, padding):
        mask_cls = outputs["pred_logits"]
        mask_pred = outputs["pred_masks"]
        # things_ids = self.trainer.datamodule.things_ids
        things_ids = get_things_ids(self.cfg.MODEL.DATASET)
        num_classes = self.cfg[self.cfg.MODEL.DATASET].NUM_CLASSES
        sem_pred = []
        ins_pred = []
        panoptic_output = []
        info = []
        for mask_cls, mask_pred, pad in zip(mask_cls, mask_pred, padding):
            scores, labels = mask_cls.max(-1)
            mask_pred = mask_pred[~pad].sigmoid()
            keep = labels.ne(num_classes)

            cur_scores = scores[keep]
            cur_classes = labels[keep]
            cur_masks = mask_pred[:, keep]
            cur_mask_cls = mask_cls[keep]
            cur_mask_cls = cur_mask_cls[:, :-1]

            # prob to belong to each of the `keep` masks for each point
            cur_prob_masks = cur_scores.unsqueeze(0) * cur_masks

            panoptic_seg = torch.zeros(
                (cur_masks.shape[0]), dtype=torch.int32, device=cur_masks.device
            )
            sem = torch.zeros_like(panoptic_seg)
            ins = torch.zeros_like(panoptic_seg)
            segments_info = []
            masks = []
            segment_id = 0
            if cur_masks.shape[1] == 0:  # no masks detected
                panoptic_output.append(panoptic_seg)
                info.append(segments_info)
                sem_pred.append(sem.cpu().numpy())
                ins_pred.append(ins.cpu().numpy())
            else:
                # mask index for each point: between 0 and (`keep` - 1)
                cur_mask_ids = cur_prob_masks.argmax(1)
                stuff_memory_list = {}
                for k in range(cur_classes.shape[0]):
                    pred_class = cur_classes[k].item()  # current class
                    isthing = pred_class in things_ids
                    mask_area = (cur_mask_ids == k).sum().item()  # points in mask k
                    original_area = (cur_masks[:, k] >= 0.5).sum().item()  # binary mas
                    mask = (cur_mask_ids == k) & (cur_masks[:, k] >= 0.5)

                    if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                        if mask_area / original_area < self.cfg.MODEL.OVERLAP_THRESHOLD:
                            continue  # binary mask occluded 80%
                        if not isthing:  # merge stuff regions
                            if int(pred_class) in stuff_memory_list.keys():
                                # in the list, asign id stored on the list for that class
                                panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                                continue
                            else:
                                # not in the list, class = cur_id + 1
                                stuff_memory_list[int(pred_class)] = segment_id + 1
                        segment_id += 1
                        panoptic_seg[mask] = segment_id
                        masks.append(mask)
                        # indice which class each segment id has
                        segments_info.append(
                            {
                                "id": segment_id,
                                "isthing": bool(isthing),
                                "sem_class": int(pred_class),
                            }
                        )
                panoptic_output.append(panoptic_seg)
                info.append(segments_info)
                for mask, inf in zip(masks, segments_info):
                    sem[mask] = inf["sem_class"]
                    if inf["isthing"]:
                        ins[mask] = inf["id"]
                    else:
                        ins[mask] = 0
                sem_pred.append(sem.cpu().numpy())
                ins_pred.append(ins.cpu().numpy())

        return sem_pred, ins_pred
