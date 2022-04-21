import sys, os
import logging

import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from datetime import timedelta
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch.nn import Linear
import torch
import numpy as np

from .utils import load_dataset
from sklearn.metrics import roc_auc_score


class GNNBase(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        """
        Initialise the Lightning Module that can scan over different GNN training regimes
        """
                
        # Assign hyperparameters
        self.save_hyperparameters(hparams)
        self.trainset, self.valset, self.testset = None, None, None

        
    def setup(self, stage):
        # Handle any subset of [train, val, test] data split, assuming that ordering
                
        if self.trainset is None:
            print("Setting up dataset")
            
            self.trainset, self.valset, self.testset = load_dataset(self.hparams["datasplit"])
        
        if (self.trainer) and ("logger" in self.trainer.__dict__.keys()) and ("_experiment" in self.logger.__dict__.keys()):
            self.logger.experiment.define_metric("val_loss" , summary="min")
            self.logger.experiment.define_metric("auc" , summary="max")

    def train_dataloader(self):
        if self.trainset is not None:
            return DataLoader(self.trainset, batch_size=self.hparams["train_batch_size"], num_workers=0)#, pin_memory=True, persistent_workers=True)
        else:
            return None

    def val_dataloader(self):
        if self.valset is not None:
            return DataLoader(self.valset, batch_size=self.hparams["val_batch_size"], num_workers=0)#, pin_memory=True, persistent_workers=True)
        else:
            return None

    def test_dataloader(self):
        if self.testset is not None:
            return DataLoader(self.testset, batch_size=self.hparams["val_batch_size"], num_workers=0)#, pin_memory=True, persistent_workers=True)
        else:
            return None

    def configure_optimizers(self):
        optimizer = [
            torch.optim.AdamW(
                self.parameters(),
                lr=(self.hparams["lr"]),
                betas=(0.9, 0.999),
                eps=1e-08,
                amsgrad=True,
            )
        ]
        scheduler = [
            {
                "scheduler": torch.optim.lr_scheduler.StepLR(
                    optimizer[0],
                    step_size=self.hparams["patience"],
                    gamma=self.hparams["factor"],
                ),
                "interval": "epoch",
                "frequency": 1,
            }
        ]
        return optimizer, scheduler
    
    def handle_directed(self, batch):
        
        edge_sample = torch.cat([batch.edge_index, batch.edge_index.flip(0)], dim=-1)
        
        if ("directed" in self.hparams.keys()) and self.hparams["directed"]:
            direction_mask = batch.x[edge_sample[0], 0] < batch.x[edge_sample[1], 0]
            edge_sample = edge_sample[:, direction_mask]
            truth_sample = truth_sample[direction_mask]
        
        return edge_sample
    
    def training_step(self, batch, batch_idx):

        truth = batch.y.float()
        
        edge_sample = self.handle_directed(batch)
        output = self(batch.x.float(), edge_sample, batch.batch).squeeze()
            
        loss = F.poisson_nll_loss(output, truth.float(), log_input=False)

        self.log("train_loss", loss, on_step=False, on_epoch=True)
                
        return loss

    def get_metrics(self, truth, output):
        
        predictions = torch.round(output)
        print(predictions, predictions.shape)
        print(output, output.shape)
        print(truth, truth.shape)
        correct = predictions == truth
        acc = correct.sum() / correct.shape[0]
        
        return predictions, acc
    
    def shared_evaluation(self, batch, batch_idx, log=True):

        truth = batch.y.float()
        
        edge_sample = self.handle_directed(batch)
        output = self(batch.x.float(), edge_sample, batch.batch).squeeze()
        loss = F.poisson_nll_loss(output, truth.float().squeeze(), log_input=False)

        predictions, acc = self.get_metrics(truth, output)

        if log:
            current_lr = self.optimizers().param_groups[0]["lr"]
            self.log_dict(
                {"val_loss": loss, "acc": acc, "current_lr": current_lr}, sync_dist=True
            )

        return {
            "loss": loss,
            "preds": predictions,
            "output": output,
            "truth": truth,
        }

    
    def validation_step(self, batch, batch_idx):

        outputs = self.shared_evaluation(batch, batch_idx)

        return outputs["loss"]

    def test_step(self, batch, batch_idx):

        outputs = self.shared_evaluation(batch, batch_idx, log=False)

        return outputs

    def test_step_end(self, output_results):

        print("Step:", output_results)

    def test_epoch_end(self, outputs):

        print("Epoch:", outputs)

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure=None,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        # warm up lr
        if (self.hparams["warmup"] is not None) and (
            self.trainer.global_step < self.hparams["warmup"]
        ):
            lr_scale = min(
                1.0, float(self.trainer.global_step + 1) / self.hparams["warmup"]
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams["lr"]

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()
