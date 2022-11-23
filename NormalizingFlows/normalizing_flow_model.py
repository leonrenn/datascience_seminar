import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.distributions import Uniform
from torch.distributions.normal import Normal
from torch.optim import Adam


class Flow1d(pl.LightningModule):
    def __init__(self,
                 n_components):
        super(Flow1d, self).__init__()

        # class variable
        self.n_components = n_components

        # target distributions that is changed during learning 
        # procedure (simples distribution)
        self.target_distribution = Uniform(0.0, 1.0)

        # infered paramters mu, sigma
        self.mus = nn.Parameter(torch.randn(
            self.n_components), requires_grad=True)
        self.log_sigmas = nn.Parameter(
            torch.zeros(self.n_components), requires_grad=True)

        # weights
        self.weight_logits = nn.Parameter(
            torch.ones(self.n_components), requires_grad=True)
        
        # loss
        self.total_loss = []
        self.epoch_end = False # only on epoch end

    def forward(self, x):
        
        # reshaping the input
        x = x.view(-1, 1)

        # transforming the distribution
        weights = self.weight_logits.softmax(dim=0).view(1, -1)
        distribution = Normal(self.mus, self.log_sigmas.exp())
        z = (distribution.cdf(x) * weights).sum(dim=1)
        dz_by_dx = (distribution.log_prob(x).exp() * weights).sum(dim=1)
        return z, dz_by_dx

    def loss_function(self, target_distribution, z, dz_by_dx):
        log_likelihood = target_distribution.log_prob(z) + dz_by_dx.log()
        return -log_likelihood.mean()

    def configure_optimizers(self):
        optim = Adam(self.parameters())
        return optim

    def training_step(self, batch, batch_idx):
        # unpacking the batch
        x = batch

        # evaluating the flow
        z, dz_by_dx = self(x)

        # evaluating the loss
        loss = self.loss_function(self.target_distribution, z, dz_by_dx)
        if self.epoch_end == True:
            self.total_loss.append(loss.item())
            self.epoch_end = False
        return loss

    def on_epoch_end(self):
        self.epoch_end = True
        return