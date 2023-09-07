import os
# import subprocess
from os.path import join

import click
import torch
import yaml
from easydict import EasyDict as edict
from fusion_pls.datasets.semantic_dataset import SemanticDatasetModule
from fusion_pls.models.mask_model import MaskPS
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint


@click.command()
@click.option("--name", "-n", type=str, default="default", required=False)
@click.option("--version", "-v", type=str, default=None, required=False)
@click.option("--ckpt", "-c", type=str, default=None, required=False)
@click.option("--weights", "-w", type=str, default=None, required=False)
@click.option("--data_path", "-d", type=str, default=None)
@click.option("--nuscenes", is_flag=False)
def main(name, version, ckpt, weights, data_path, nuscenes):
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

    if nuscenes:
        cfg.MODEL.DATASET = "NUSCENES"
        if data_path:
            cfg.NUSCENES.PATH = data_path
    else:
        if data_path:
            cfg.KITTI.PATH = data_path

    data = SemanticDatasetModule(cfg)
    model = MaskPS(cfg)

    if weights is not None:
        weights = torch.load(weights, map_location="cpu")
        model.load_state_dict(weights["state_dict"])

    # for param in model.backbone.parameters():
    #     param.requires_grad = False
    # for param in model.backbone.mink.parameters():
    #     param.requires_grad = False

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir="experiments/" + cfg.EXPERIMENT.ID,
        name=name,
        version=version,
        default_hp_metric=False,
    )

    # Callbacks
    lr_monitor = LearningRateMonitor(logging_interval="step")
    iou_ckpt = ModelCheckpoint(
        monitor="metrics/iou",
        filename=cfg.EXPERIMENT.ID + "_epoch{epoch:02d}_iou{metrics/iou:.2f}",
        auto_insert_metric_name=False,
        mode="max",
        save_last=True,
    )
    pq_ckpt = ModelCheckpoint(
        monitor="metrics/pq",
        filename=cfg.EXPERIMENT.ID + "_epoch{epoch:02d}_pq{metrics/pq:.2f}",
        auto_insert_metric_name=False,
        mode="max",
        save_last=True,
    )

    trainer = Trainer(
        gpus=cfg.TRAIN.N_GPUS,
        accelerator="ddp",
        logger=tb_logger,
        max_epochs=cfg.TRAIN.MAX_EPOCH,
        callbacks=[lr_monitor, pq_ckpt, iou_ckpt],
        log_every_n_steps=1,
        gradient_clip_val=0.5,
        accumulate_grad_batches=cfg.TRAIN.BATCH_ACC,
        resume_from_checkpoint=ckpt,
    )

    trainer.fit(model, data)


class PrintMetricsCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        # get current metrics
        metrics = trainer.callback_metrics
        # print metrics/pq, metrics/rq and metrics/iou
        print(f"Metrics: {{"
              f"pq: {metrics['metrics/pq']:.4f}; "
              f"rq: {metrics['metrics/rq']:.4f}; "
              f"iou: {metrics['metrics/iou']:.4f}"
              f"}}")


def getDir(obj):
    return os.path.dirname(os.path.abspath(obj))


if __name__ == "__main__":
    main()
