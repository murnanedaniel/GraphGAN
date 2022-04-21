import sys
import argparse
import yaml
import time

import torch
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer

sys.path.append("../")
from lightning_modules.Point_GAN.Models.gan_gnn import GanGNN

import wandb

def main():
    print("Running main")
    print(time.ctime())
    
    default_config_path = "default_config.yaml"
    
    with open(default_config_path) as file:
        default_configs = yaml.load(file, Loader=yaml.FullLoader)

    wandb.finish()
    wandb.init(config = default_configs, project=default_configs["project"])
    config = wandb.config
    
    print("Initialising model")
    print(time.ctime())
    model = GanGNN(dict(config))
    logger = WandbLogger()
    logger.watch(model, log="all")
    
    trainer = Trainer(gpus=1, max_epochs=config["max_epochs"], logger=logger)
    trainer.fit(model)
    
    
if __name__ == "__main__":
    
    main()