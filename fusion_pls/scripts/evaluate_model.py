import os
from os.path import join

import click
import torch
import yaml
from easydict import EasyDict as edict
from fusion_pls.datasets.semantic_dataset import SemanticDatasetModule
from fusion_pls.models.mask_model import FusionLPS
from pytorch_lightning import Trainer


def getDir(obj):
    return os.path.dirname(os.path.abspath(obj))


@click.command()
@click.option("--name", "-n", type=str, default="default", required=False)
@click.option("--split", "-sp", type=str, default="valid", required=False)
@click.option("--save", is_flag=False)
@click.option("--weights", "-w", type=str, required=True)
@click.option("--data_path", "-dp", type=str, default=None)
@click.option("--batch_size", "-bs", type=int, default=1)
def main(name, split, weights, save, data_path, batch_size):
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

    cfg.EVALUATE = True
    cfg.MODEL.ENABLE_KD = False
    cfg.TRAIN.BATCH_SIZE = batch_size
    if save:
        results_dir = create_dirs(name, split)
        print(f"Saving {split} set predictions in directory {results_dir}")
        cfg.RESULTS_DIR = results_dir

    if data_path:
        cfg.KITTI.PATH = data_path

    data = SemanticDatasetModule(cfg)
    model = FusionLPS(cfg)
    w = torch.load(weights, map_location="cpu")
    model.load_state_dict(w["state_dict"], strict=False)

    trainer = Trainer(
        gpus=cfg.TRAIN.N_GPUS,
        accelerator="ddp",
        logger=False)

    if split == "test":
        trainer.test(model, data)
    elif split == "valid":
        trainer.validate(model, data)
    else:
        raise ValueError(f"Invalid split type: {split}")
    model.evaluator.print_results()


def create_dirs(name="test", split="test"):
    results_dir = join(getDir(__file__), "..", "output", name, "sequences")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)
    if split == "test":
        for i in range(11, 22):
            sub_dir = os.path.join(results_dir, str(i).zfill(2), "predictions")
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir, exist_ok=True)
    elif split == "valid":
        for i in range(8, 9):
            sub_dir = os.path.join(results_dir, str(i).zfill(2), "predictions")
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir, exist_ok=True)
    else:
        raise ValueError("Invalid split type")
    return results_dir


if __name__ == "__main__":
    main()
