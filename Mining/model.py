import os
from typing import Any

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.optim import Adam


class Model(pl.LightningModule):
    def __init__(self,
                 generate_gif=False) -> None:
        super().__init__()

        # generate gif
        self.generate_gif = generate_gif

        # total loss on epoch end    
        self.total_loss = []
        self.epoch_end = False

        self.net = nn.Sequential(
            nn.Linear(3, 10),
            nn.PReLU(),
            nn.Linear(10, 10),
            nn.PReLU(),
            nn.Linear(10, 10),
            nn.PReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, x) -> Any:
        print(x.shape)
        output = self.net(x)
        print(output.shape)
        return output

    def configure_optimizers(self):
        optim = Adam(self.parameters())
        return optim

    def training_step(self, train_batch, batch_idx):
        X, Y = train_batch

        # output of the model
        ratio_hat = self.net(X).view(-1).exp()

        # unpack Y
        targets, ratio = Y[:, :, 0].view(-1), Y[:, :, 1].view(-1).exp()

        # loss function
        loss = 1/ratio.shape[0] * torch.sum(targets * (ratio - ratio_hat)**2 + targets * (1/ratio - 1/ratio_hat)**2)
        self.log("train_loss", loss)

        if self.epoch_end is True:
            self.total_loss.append(loss.detach().numpy())
            self.epoch_end = False

        return loss

    def on_epoch_end(self):
        # toggle epoch end for easy loss display
        self.epoch_end = True

        # generating training gif
        if self.generate_gif is True:
            x_min, x_max = -3, 4
            y_min, y_max = -3, 4
            size = 20
            X = torch.empty((size, 3))
            X[:, 0] = torch.linspace(x_min, x_max, size)
            X[:, 1] = torch.zeros(size)
            X[:, 2] = torch.ones(size)

            output = self.net(X).detach().numpy()

            true_log_ratio = Normal(loc=0, scale=1).log_prob(X[:, 0]) - Normal(loc=1, scale=1).log_prob(X[:, 0])

            plt.plot(X[:, 0].detach().numpy(), output, "ro",label="Model Output")
            plt.plot(X[:, 0].detach().numpy(), true_log_ratio.detach().numpy(), "b-",label="True Log Ratio")
            plt.legend()
            plt.xlabel("X")
            plt.ylabel("Log Likelihood Ratio")
            plt.title(f"Epoch {len(self.total_loss)}")
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            plt.savefig(f"frames/train_epoch_{len(self.total_loss)}.png")
            plt.clf()
        return 

    def on_train_end(self) -> None:
        if self.generate_gif is True:
            frames = np.stack([iio.imread(f'frames/train_epoch_{epoch}.png') for epoch in range(len(self.total_loss))], axis=0)
            iio.imwrite('animations/ratio_training.gif', frames)
            os.system("cd frames && rm -rf && cd ..")
        print("GIF DONE")
        return 

    