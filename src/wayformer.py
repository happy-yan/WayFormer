import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
import pytorch_lightning as pl

from typing import List, Dict, Tuple
from einops import rearrange, reduce, repeat
from math import pi, log

from lib import EarlyFusion, Decoder
from utils import Args, Metrics, get_from_mapping


class Wayformer(nn.Module):

    def __init__(self, args: Args) -> None:
        super(Wayformer, self).__init__()
        self.args = args
        self.encoder = EarlyFusion(args)
        self.decoder = Decoder(args)
    
    def forward(self, mappings: List, device) -> Tensor:
        agents = get_from_mapping(mappings, 'agents')
        matrix = get_from_mapping(mappings, 'matrix')

        labels = get_from_mapping(mappings, 'labels')
        batch_size = len(labels)
        labels = rearrange(labels, 'b t d -> b t d')
        labels = torch.tensor(labels, device=device)

        memory, memory_mask = self.encoder(agents, matrix, device)
        weights, trajs = self.decoder(memory, memory_mask)

        indices = self.get_closest_traj(trajs, labels)
        cls_loss = F.cross_entropy(weights, indices, reduction='mean')
        ind = torch.arange(batch_size, device=device)
        traj_selected = trajs[ind, indices]  # (batch_size, t, d)
        means, logvars = traj_selected.chunk(2, dim=-1)
        negative_log_prob = - self.log_prob_gaussian(means, logvars, labels)
        loss = cls_loss + negative_log_prob.mean()

        return loss, means

    def log_prob_gaussian(self, means: Tensor, logvars: Tensor, labels: Tensor) -> Tensor:
        # print(means.max().item())
        var = torch.exp(logvars)
        log_prob = -0.5 * (log(2 * pi) + logvars + ((labels - means) ** 2) / var)
        return reduce(log_prob, 'b t d -> b', 'sum')

    @torch.no_grad()
    def get_closest_traj(self, trajs: Tensor, labels: Tensor) -> Tensor:
        k = trajs.shape[1]
        xy = trajs[..., :2]
        labels = repeat(labels, 'b t d -> b k t d', k=k)
        ADEs = reduce((xy - labels)**2, 'b k t d -> b k', 'sum')
        indices = torch.min(ADEs, dim=1)[1]
        return indices
    
    @torch.no_grad()
    def predict(self, mappings: List, device, num_query: int=-1) -> Tensor:
        agents = get_from_mapping(mappings, 'agents')
        matrix = get_from_mapping(mappings, 'matrix')
        memory, memory_mask = self.encoder(agents, matrix, device)
        scores, trajs = self.decoder(memory, memory_mask)
        # scores: (batch_size, k)
        # trajs: (batch_size, k, t, d)
        k = scores.shape[1]
        if num_query > 0 and num_query < k:
            _, indices = torch.topk(scores, k=num_query, dim=1)
            trajs = trajs[torch.arange(len(trajs))[:, None], indices]
        
        return trajs


class WayformerPL(pl.LightningModule):
    
    def __init__(self, args: Args) -> None:
        super(WayformerPL, self).__init__()
        self.args = args
        self.model = Wayformer(args)
        self.save_hyperparameters()

    def forward(self, mappings: List) -> Tensor:
        return self.model(mappings, self.device)

    def training_step(self, batch, batch_idx):
        loss, _ = self(batch)
        self.log('train_loss', loss, prog_bar=True, batch_size=self.args.batch_size)
        # assert isinstance(loss, Tensor)
        # assert not torch.isnan(loss)
        # assert loss.shape == ()
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _ = self(batch)
        self.log('val_loss', loss, prog_bar=True, batch_size=self.args.batch_size)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 1.0, 0.0, self.args.max_epochs)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "epoch",
            }
        }
    
    @torch.no_grad()
    def evaluate_model(self, mappings: List, k: int=6, threshold: float=2.0) -> Metrics:
        trajs = self.model.predict(mappings, self.device, num_query=k)
        labels = get_from_mapping(mappings, 'labels')
        labels = rearrange(labels, 'b t d -> b t d')
        labels = torch.tensor(labels, device=self.device)
        ADE = self.get_minADE_k(trajs, labels)
        FDE, missed = self.get_minFDE_k(trajs, labels, threshold)

        return Metrics(trajs.shape[1], ADE, FDE, missed)

    @torch.no_grad()
    def get_minADE_k(self, trajs: Tensor, labels: Tensor) -> float:
        # trajs: (batch_size, k, t, d)
        # labels: (batch_size, t, d)
        k = trajs.shape[1]
        xy = trajs[..., :2]
        labels = repeat(labels, 'b t d -> b k t d', k=k)
        ADEs = reduce((xy - labels)**2, 'b k t d -> b k t', 'sum')
        ADEs = reduce(torch.sqrt(ADEs), 'b k t -> b k', 'mean')
        minADEs = reduce(ADEs, 'b k -> b', 'min')
        return minADEs.mean().item()
    
    @torch.no_grad()
    def get_minFDE_k(self, trajs: Tensor, labels: Tensor, threshold: float) -> Tuple[float, int]:
        # get minFDE and missed predictions
        # trajs: (batch_size, k, t, d)
        # labels: (batch_size, t, d)
        k = trajs.shape[1]
        xy_final = trajs[..., -1, :2]
        labels = repeat(labels, 'b t d -> b k t d', k=k)
        labels = labels[..., -1, :2]
        FDEs = reduce((xy_final - labels)**2, 'b k d -> b k', 'sum')
        FDEs = (torch.sqrt(FDEs))
        minFDEs = reduce(FDEs, 'b k -> b', 'min')
        return minFDEs.mean().item(), (minFDEs > threshold).sum().item()