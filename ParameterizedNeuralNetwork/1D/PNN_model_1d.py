import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class ParameterizedNeuralNet(pl.LightningModule):
    def __init__(self, llr_analysis=False): 
        """Neural Network parameterized with hypothesis of signal and background
        means.

        Args:
            llr_analysis (bool, optional): For likelihood ratio analysis the 
            output should be without the Sigmoid (Do not use when training). Defaults to False.
        """
        super().__init__()

        # losses for training process visualization
        self.losses = []

        # likelihood ratio analysis
        self.llr_analysis = llr_analysis

        # network structure
        self.net = nn.Sequential(nn.Linear(in_features=3, out_features=20),
                                 nn.PReLU(),
                                 nn.Linear(in_features=20, out_features=20),
                                 nn.PReLU(),
                                 nn.Linear(in_features=20, out_features=20),
                                 nn.PReLU(),
                                 nn.Linear(in_features=20, out_features=20),
                                 nn.PReLU(),
                                 nn.Linear(in_features=20, out_features=1))
                                 
        # deciding if the output will come from the sigmoid layer or from last neuron
        if self.llr_analysis is False:
            self.net.append(nn.Sigmoid())
                                 
    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        optim  = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optim

    def training_step(self, train_batch, batch_idx):
        X, Y = train_batch
        Y_hat = self.net(X)
        loss = F.binary_cross_entropy(torch.squeeze(Y_hat, -1), Y)
        self.losses.append(loss.detach().numpy())
        self.log("train_loss", loss)
        return loss