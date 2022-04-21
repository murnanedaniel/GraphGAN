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

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser('train.py')
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='default_config.yaml')
    return parser.parse_args()


def main():
    print("Running main")
    print(time.ctime())
    
    args = parse_args()
        
    with open(args.config) as file:
        default_configs = yaml.load(file, Loader=yaml.FullLoader)
    
    print("Initialising model")
    print(time.ctime())
    model = GanGNN(default_configs)
    
    wandb.finish()
    logger = WandbLogger(project=default_configs["project"], save_dir=default_configs["artifacts"])
    
    trainer = Trainer(gpus=1, max_epochs=default_configs["max_epochs"], logger=logger)
    trainer.fit(model)
    
    
if __name__ == "__main__":
    
    main()