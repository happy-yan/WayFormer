import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
import pytorch_lightning as pl

from typing import List
from einops import rearrange, reduce, repeat
from math import pi, log
import pandas as pd
# from modeling.lib import EarlyFusion, Decoder
# from utils import Args, get_from_mapping
from lib import EarlyFusion, Decoder
from utils import Args, get_from_mapping

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
        cls_loss = F.cross_entropy(weights, indices, reduction='mean')  ### 根据trajs和labels得到标签，进行交叉熵处理
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


class WayformerPL(pl.LightningModule):
    
    def __init__(self, args: Args) -> None:
        super(WayformerPL, self).__init__()
        self.args = args
        self.model = Wayformer(args)
        self.save_hyperparameters()

    def forward(self, mappings: List) -> Tensor:
        return self.model(mappings, self.device)

    def training_step(self, batch, batch_idx):
        train_loss_list = []
        loss, _ = self(batch)
        train_loss_list.append(loss)
        data = pd.DataFrame(train_loss_list)
        writer = pd.ExcelWriter('train_loss_data.xlsx')  # 写入Excel文件
        data.to_excel(writer, 'page_1', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
        writer.save()
        writer.close()
        self.log('train_loss', loss, prog_bar=True, batch_size=self.args.batch_size)
        # assert isinstance(loss, Tensor)
        # assert not torch.isnan(loss)
        # assert loss.shape == ()
        return loss

    def validation_step(self, batch, batch_idx):
        val_loss_list = []
        loss, _ = self(batch)
        val_loss_list.append(loss)
        data = pd.DataFrame(val_loss_list)
        writer = pd.ExcelWriter('val_loss_data.xlsx')  # 写入Excel文件
        data.to_excel(writer, 'page_1', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
        writer.save()
        writer.close()
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