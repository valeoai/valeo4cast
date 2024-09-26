import sys
import os
print("You should modify the path in predict.py. Hack implementation, by sys.path.insert.")
sys.path.append("./models/mtr/ops/")
sys.path.append("./")
import torch

import pytorch_lightning as pl

torch.set_float32_matmul_precision('medium')

from torch.utils.data import DataLoader

import hydra

from models import build_model
from datasets import build_dataset
from omegaconf import OmegaConf
from utils.utils import set_seed

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import BasePredictionWriter


class CustomWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir
        print("!!!!!!!!!!!!!!!!!!!i!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", self.output_dir)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        prediction_dict, pred_trajs_world, pred_dict_list = prediction
        for pred in pred_dict_list:
            filename = pred["scenario_id"] + '___' + pred["object_id"] + '___' + str(pred["current_time_index"]) + "___.pt"
            torch.save(pred, os.path.join(self.output_dir, filename))

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        #torch.save(predictions, os.path.join(self.output_dir, "predictions.pt"))
        return


@hydra.main(version_base=None, config_path="configs", config_name="challenge_config")
def prediction(cfg):
    set_seed(cfg.seed)
    OmegaConf.set_struct(cfg, False)  # Open the struct
    cfg = OmegaConf.merge(cfg, cfg.method)
    cfg['eval'] = True

    model = build_model(cfg)

    val_set = build_dataset(cfg, val=True)

    eval_batch_size = max(cfg.method['eval_batch_size'] // len(cfg.devices) // val_set.data_chunk_size, 1)

    val_loader = DataLoader(
        val_set, batch_size=eval_batch_size, num_workers=cfg.load_num_workers, shuffle=False, drop_last=False,
        collate_fn=val_set.collate_fn)

    data_path = cfg["val_data_path"]
    assert len(data_path) == 1 # below won't work if there are multiple files
    data_path = data_path[0]
    dataset_name = data_path.split('/')[-1]
    output_path = os.path.join(data_path, f'output_{cfg.method.model_name}')
    output_path = output_path.replace("/mnt/proj1/dd-24-41/", "/mnt/proj2/dd-23-131/")
    print("!!!!!!!!!!!!!!!!!!!i!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", output_path)
    if os.path.exists(output_path) and len(os.listdir(output_path)) >0:
        print(f"You should remove first old results in {output_path}")
        assert False

    pred_writer = CustomWriter(output_dir=output_path, write_interval="batch")



    trainer = pl.Trainer(
        inference_mode=True,
        logger=None, # if cfg.debug else WandbLogger(project="unitraj", name=cfg.exp_name),
        devices=cfg.devices,
        accelerator="cpu" if cfg.debug else "gpu",
        profiler="simple",
        strategy='ddp',
        callbacks=[pred_writer],
    )

    predictions = trainer.predict(model=model, dataloaders=val_loader, ckpt_path=cfg.ckpt_path)

if __name__ == '__main__':
    prediction()

