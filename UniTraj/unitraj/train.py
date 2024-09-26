import sys
import os
print("You should modify the path in unitraj/train.py. Hack implementation, by sys.path.insert.")
sys.path.append("./models/mtr/ops/")
sys.path.append("./")

import pytorch_lightning as pl
import torch

torch.set_float32_matmul_precision('medium')
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from models import build_model
from datasets import build_dataset
from utils.utils import set_seed
from pytorch_lightning.callbacks import ModelCheckpoint  # Import ModelCheckpoint
import hydra
from omegaconf import OmegaConf



@hydra.main(version_base=None, config_path="configs", config_name="challenge_config")
def train(cfg):
    set_seed(cfg.seed)
    OmegaConf.set_struct(cfg, False)  # Open the struct
    cfg = OmegaConf.merge(cfg, cfg.method)

    model = build_model(cfg)

    train_set = build_dataset(cfg)
    val_set = build_dataset(cfg, val=True)

    train_batch_size = max(cfg.method['train_batch_size'] // len(cfg.devices) // train_set.data_chunk_size, 1)
    eval_batch_size = max(cfg.method['eval_batch_size'] // len(cfg.devices) // val_set.data_chunk_size, 1)
    print("train_batch_size ", train_batch_size)
    print("eval_batch_size ", eval_batch_size)
    # train_batch_size = 8
    # eval_batch_size = 8
    call_backs = []

    checkpoint_callback = ModelCheckpoint(
        monitor='val/brier_fde',  # Replace with your validation metric
        filename='{epoch}-{val/brier_fde:.2f}',
        save_top_k=1,
        mode='min',  # 'min' for loss/error, 'max' for accuracy
    )

    call_backs.append(checkpoint_callback)

    train_loader = DataLoader(
        train_set, batch_size=train_batch_size, num_workers=cfg.load_num_workers, drop_last=False,
        collate_fn=train_set.collate_fn)

    val_loader = DataLoader(
        val_set, batch_size=eval_batch_size, num_workers=cfg.load_num_workers, shuffle=False, drop_last=False,
        collate_fn=train_set.collate_fn)

    strategy = "auto"
    if not cfg.debug and len(cfg.devices) > 1:
        strategy = "ddp"

    print("len(cfg.devices)", len(cfg.devices))
    print("strategy ", strategy)
    trainer = pl.Trainer(
        max_epochs=cfg.method.max_epochs,
        logger=None, #WandbLogger(project="unitraj", name=cfg.exp_name), # if cfg.debug else WandbLogger(project="unitraj", name=cfg.exp_name),
        devices=cfg.devices if cfg.debug else cfg.devices,
        gradient_clip_val=cfg.method.grad_clip_norm,
        accelerator="cpu" if cfg.debug else "gpu",
        profiler="simple",
        strategy=strategy,
        callbacks=call_backs
    )
    
    # load model weights only, other from scratch
    if cfg.pretrained is not None:
        print(f"Loading pretrained weights from {cfg.pretrained}...")
        pretrained_weights = torch.load(cfg.pretrained)["state_dict"]
        model.load_state_dict(pretrained_weights)

    if cfg.ckpt_path is not None:
        trainer.validate(model=model, dataloaders=val_loader, ckpt_path=cfg.ckpt_path)
        # trainer.validate(model=model, dataloaders=val_loader)

    if cfg.ckpt_path is not None:
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=cfg.ckpt_path)
    else:
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == '__main__':
    train()
