import sys, os
import logging

import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from datetime import timedelta
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch.nn import Linear
import torch
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import numpy as np
import wandb

import matplotlib.pyplot as plt
import matplotlib.cm as cm

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
            
            self.trainset, self.valset, self.testset = load_dataset(self.hparams["datasplit"], self.hparams["length"])
        
        if (self.trainer) and ("logger" in self.trainer.__dict__.keys()) and ("_experiment" in self.logger.__dict__.keys()):
            self.logger.experiment.define_metric("discriminator_train_loss" , summary="max")

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
        
        opt_g = torch.optim.RMSprop(self.generator.parameters(), lr=self.generator_hparams["lr"])
        opt_d = torch.optim.RMSprop(self.discriminator.parameters(), lr=self.discriminator_hparams["lr"])
    
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
        # return F.binary_cross_entropy(y_hat, y)
        return F.mse_loss(y_hat, y)
    
    def training_step(self, batch, batch_idx, optimizer_idx):

        # Generate points
        true_data = batch.x.float()
        ptr_repeater = batch.ptr[1:] - batch.ptr[:-1]
        input_edges = self.handle_directed(batch)
        random_node_data = torch.normal(0.0, 1.0, (true_data.shape[0], self.generator_hparams["input_node_channels"])).to(self.device)   
        random_graph_data = torch.normal(0.0, 1.0, (batch.batch.max()+1, self.generator_hparams["input_graph_channels"])).to(self.device).repeat_interleave(ptr_repeater, dim=0) 
        predicted_data = self(random_node_data, random_graph_data, input_edges, batch.batch, ptr_repeater)
            
        # Train Generator
        if optimizer_idx == 0:
            
            # Create truth
            truth = torch.ones(batch.batch.max() + 1, 1).squeeze(1)
            truth = truth.type_as(predicted_data)
            
            # Adversarial loss
            discriminator_output = self.discriminator(predicted_data, input_edges, batch.batch)
            g_loss = self.adversarial_loss(discriminator_output, truth)
                
            self.log("generator_train_loss", g_loss, on_step=False, on_epoch=True)

            return g_loss
        
        # Train Discriminator
        if optimizer_idx == 1:
            
            # Create true graphs
            truth = torch.ones(batch.batch.max() + 1, 1).squeeze(1) * self.hparams["smoothing"]
            truth = truth.type_as(true_data)
            
            real_loss = self.adversarial_loss(self.discriminator(true_data, input_edges, batch.batch), truth)
            
            # Create generated graphs
            truth = torch.zeros(batch.batch.max() + 1, 1).squeeze(1)
            truth = truth.type_as(predicted_data)
            
            fake_loss = self.adversarial_loss(self.discriminator(predicted_data.detach(), input_edges, batch.batch), truth)
            
            d_loss = (real_loss + fake_loss) / 2
            
            if self.discriminator_hparams["grad_penalty"]:
                gp = self.gradient_penalty(self.discriminator_hparams["grad_penalty"], true_data, predicted_data.detach(), input_edges, batch.batch, batch.batch.max() + 1)
                gpitem = gp.item()
                d_loss += gp
                
            self.log("discriminator_train_loss", d_loss, on_step=False, on_epoch=True)
            
            return d_loss
            
    def get_metrics(self, truth, output):
        
        predictions = torch.round(output)
        # print(predictions, predictions.shape)
        # print(output, output.shape)
        # print(truth, truth.shape)
        correct = predictions == truth
        acc = correct.sum() / correct.shape[0]
        # print(truth.cpu(), output.cpu())
        auc = roc_auc_score(truth.cpu(), output.cpu())
        
        # print("AUC", auc)
        
        return predictions, acc, auc
    
    def plot_polygon(self, nodes, edges):
        
        # print(nodes.shape, edges.shape)
        nodes_cpu, edges_cpu = nodes.detach().cpu(), edges.detach().cpu()
        fig, ax = plt.subplots(figsize=(3, 3))        
        
        plt.plot(nodes_cpu[:, 0][edges], nodes_cpu[:, 1][edges], c="b"); 
        ax.scatter(nodes_cpu.T[0], nodes_cpu.T[1], c="r")

        # If we haven't already shown or saved the plot, then we need to
        # draw the figure first...
        fig.canvas.draw()

        # Now we can save it to a numpy array.
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        return data       
        
    def shared_evaluation(self, batch, batch_idx, log=True):

        # Generate points
        true_data = batch.x.float()
        ptr_repeater = batch.ptr[1:] - batch.ptr[:-1]
        input_edges = self.handle_directed(batch)
        random_node_data = torch.normal(0.0, 1.0, (true_data.shape[0], self.generator_hparams["input_node_channels"])).to(self.device)   
        random_graph_data = torch.normal(0.0, 1.0, (batch.batch.max()+1, self.generator_hparams["input_graph_channels"])).to(self.device).repeat_interleave(ptr_repeater, dim=0) 
        predicted_data = self(random_node_data, random_graph_data, input_edges, batch.batch, ptr_repeater)
                    
        # Create truth
        truth = torch.ones(batch.batch.max().item() + 1, 1).squeeze(1)
        truth = truth.type_as(predicted_data)

        # Adversarial loss
        discriminator_output = self.discriminator(predicted_data, input_edges, batch.batch)
        # print("V", discriminator_output, truth)
        g_loss = self.adversarial_loss(discriminator_output, truth)

        # self.logger.experiment.log({
        #     "val/examples": wandb.Image(self.plot_polygon(predicted_data, input_edges))
        # })
            
        # print(batch.batch.max())
        real_truth = torch.ones(batch.batch.max() + 1, 1).squeeze(1)
        real_truth = real_truth.type_as(true_data)

        real_predictions = self.discriminator(true_data, input_edges, batch.batch)
        real_loss = self.adversarial_loss(real_predictions, real_truth)

        # Create generated graphs
        fake_truth = torch.zeros(batch.batch.max().item() + 1, 1).squeeze(1)
        fake_truth = fake_truth.type_as(predicted_data)

        fake_predictions = self.discriminator(predicted_data, input_edges, batch.batch)
        fake_loss = self.adversarial_loss(fake_predictions, fake_truth)

        d_loss = (real_loss + fake_loss) / 2

        eps = 1e-12
        length_preds = torch.sqrt(torch.sum((predicted_data[0] - predicted_data[1])**2, dim=-1) + eps)
        length_preds_error = (length_preds - self.hparams["length"]).abs().sum()
        
        disc_predictions, disc_acc, disc_auc = self.get_metrics(torch.cat([real_truth, fake_truth], axis=-1), torch.cat([real_predictions, fake_predictions], axis=-1))
        
        # print(input_edges)
        # print(predicted_data)
        # print(predicted_data[input_edges[0]], predicted_data[input_edges[1]])
        # print(length_preds, length_preds_error)


        if log:
            current_g_lr = self.optimizers()[0].param_groups[0]["lr"]
            current_d_lr = self.optimizers()[1].param_groups[0]["lr"]
            self.log_dict(
                {"generator_val_loss": g_loss, "discriminator_val_loss": d_loss, "current_g_lr": current_g_lr, "current_d_lr": current_d_lr, "disc_acc": disc_acc, "length_error": length_preds_error}, sync_dist=True
            )

        return {
            "loss": g_loss,
            "scores": torch.cat([real_predictions, fake_predictions], axis=-1),
            "truth": torch.cat([real_truth, fake_truth], axis=-1)                                    
        }

    
    def validation_step(self, batch, batch_idx):

        outputs = self.shared_evaluation(batch, batch_idx)

        return outputs

    def validation_epoch_end(self, validation_step_outputs):
        # print(validation_step_outputs)
        
        # try:
        all_truth = torch.cat([result["truth"] for result in validation_step_outputs])
        all_scores = torch.cat([result["scores"] for result in validation_step_outputs])
        print(all_truth)
        print(all_scores)
        disc_auc = roc_auc_score(all_truth.cpu(), all_scores.cpu())
        print(disc_auc)
        self.log_dict(
            {"disc_auc": disc_auc}, sync_dist=True
        )
            
        # except:
        #     pass           
        
        
        
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
        lrs = ["generator_hparams", "discriminator_hparams"]
        
        # warm up lr
        if (getattr(self, lrs[optimizer_idx])["warmup"] is not None) and (
            self.current_epoch < getattr(self, lrs[optimizer_idx])["warmup"]
        ):
            
            lr_scale = min(
                1.0, float(self.current_epoch + 1) / getattr(self, lrs[optimizer_idx])["warmup"]
            )
            
            
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * getattr(self, lrs[optimizer_idx])["lr"]

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

        
    # from https://github.com/EmilienDupont/wgan-gp
    def gradient_penalty(self, gp_lambda, real_data, generated_data, edge_inputs, batch, batch_size):
        # Calculate interpolation
        alpha = torch.rand(real_data.shape[0], 1).to(self.device)      
        interpolated = alpha * real_data + (1 - alpha) * generated_data
        interpolated = Variable(interpolated, requires_grad=True).to(self.device)

        del alpha
        torch.cuda.empty_cache()

        # Calculate probability of interpolated examples
        prob_interpolated = self.discriminator(interpolated, edge_inputs, batch)
        
        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(
            outputs=prob_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones(prob_interpolated.size()).to(self.device),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0].to(self.device)
        gradients = gradients.contiguous()

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        gp = gp_lambda * ((gradients_norm - 1) ** 2).mean()
        return gp