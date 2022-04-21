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
import wandb

import matplotlib.pyplot as plt

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
        lr = self.hparams["lr"]
        
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
    
        schedulers = [
            {
                "scheduler": torch.optim.lr_scheduler.StepLR(
                    opt_g,
                    step_size=self.hparams["patience"],
                    gamma=self.hparams["factor"],
                ),
                "interval": "epoch",
                "frequency": 1,
            },
            {
                "scheduler": torch.optim.lr_scheduler.StepLR(
                    opt_d,
                    step_size=self.hparams["patience"],
                    gamma=self.hparams["factor"],
                ),
                "interval": "epoch",
                "frequency": 1,
            }
        ]
        return [opt_g, opt_d], schedulers
    
    def handle_directed(self, batch):
        
        edge_sample = torch.cat([batch.edge_index, batch.edge_index.flip(0)], dim=-1)
        
        if ("directed" in self.hparams.keys()) and self.hparams["directed"]:
            direction_mask = batch.x[edge_sample[0], 0] < batch.x[edge_sample[1], 0]
            edge_sample = edge_sample[:, direction_mask]
            truth_sample = truth_sample[direction_mask]
        
        return edge_sample
    
    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy_with_logits(y_hat, y)
    
    def training_step(self, batch, batch_idx, optimizer_idx):

        # Generate edges
        input_data = batch.x.float()
        x, generated_edges = self(input_data, batch.batch)
            
        # Train Generator
        if optimizer_idx == 0:
            
            # Create truth
            truth = torch.ones(batch.batch.max() + 1, 1).squeeze(1)
            truth = truth.type_as(input_data)
            
            # Adversarial loss
            discriminator_output = self.discriminator(x, generated_edges, batch.batch)
            # print("T", discriminator_output, truth)
            g_loss = self.adversarial_loss(discriminator_output, truth)

            if self.hparams["l2_loss"]:
                g_loss += F.mse_loss(x, input_data)
                
            self.log("generator_train_loss", g_loss, on_step=False, on_epoch=True)

            return g_loss
        
        if optimizer_idx == 1:
            
            # Create true graphs
            input_edges = self.handle_directed(batch)
            truth = torch.ones(batch.batch.max() + 1, 1).squeeze(1) * self.hparams["smoothing"]
            truth = truth.type_as(input_data)
            
            real_loss = self.adversarial_loss(self.discriminator(input_data, input_edges, batch.batch), truth)
            
            # Create generated graphs
            truth = torch.zeros(batch.batch.max() + 1, 1).squeeze(1)
            truth = truth.type_as(input_data)
            
            fake_loss = self.adversarial_loss(self.discriminator(x, generated_edges.detach(), batch.batch), truth)
            
            d_loss = (real_loss + fake_loss) / 2
            self.log("discriminator_train_loss", d_loss, on_step=False, on_epoch=True)
            
            return d_loss
            
    def get_metrics(self, truth, output):
        
        predictions = torch.round(output)
        print(predictions, predictions.shape)
        print(output, output.shape)
        print(truth, truth.shape)
        correct = predictions == truth
        acc = correct.sum() / correct.shape[0]
        
        return predictions, acc
    
    def plot_polygon(self, nodes, edges):
        
        # print(nodes.shape, edges.shape)
        nodes_cpu, edges_cpu = nodes.detach().cpu(), edges.detach().cpu()
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.scatter(nodes_cpu.T[0], nodes_cpu.T[1])
        ax.plot(nodes_cpu[:, 0][edges_cpu], nodes_cpu[:, 1][edges_cpu]); 

        # If we haven't already shown or saved the plot, then we need to
        # draw the figure first...
        fig.canvas.draw()

        # Now we can save it to a numpy array.
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        return data       
        
    def shared_evaluation(self, batch, batch_idx, log=True):

        # Generate edges
        input_data = batch.x.float()
        x, generated_edges = self(input_data, batch.batch)
            
        # Create truth
        truth = torch.ones(batch.batch.max().item() + 1, 1).squeeze(1)
        truth = truth.type_as(input_data)

        # Adversarial loss
        discriminator_output = self.discriminator(x, generated_edges, batch.batch)
        # print("V", discriminator_output, truth)
        g_loss = self.adversarial_loss(discriminator_output, truth)

        self.logger.experiment.log({
            "val/examples": wandb.Image(self.plot_polygon(x, generated_edges))
        })
            
        # Create true graphs
        input_edges = self.handle_directed(batch)
        # print(batch.batch.max())
        truth = torch.ones(batch.batch.max() + 1, 1).squeeze(1)
        truth = truth.type_as(input_data)

        real_loss = self.adversarial_loss(self.discriminator(input_data, input_edges, batch.batch), truth)

        # Create generated graphs
        truth = torch.zeros(batch.batch.max().item() + 1, 1).squeeze(1)
        truth = truth.type_as(input_data)

        fake_loss = self.adversarial_loss(self.discriminator(x, generated_edges.detach(), batch.batch), truth)

        d_loss = (real_loss + fake_loss) / 2
        
        # predictions, acc = self.get_metrics(truth, output)

        if log:
            current_lr = self.optimizers()[0].param_groups[0]["lr"]
            self.log_dict(
                {"generator_train_loss": g_loss, "discriminator_train_loss": d_loss, "current_lr": current_lr}, sync_dist=True
            )

        return {
            "loss": g_loss,
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
