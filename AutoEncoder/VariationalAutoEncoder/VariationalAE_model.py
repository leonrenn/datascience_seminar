import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import distributions
from torch.optim import Adam


class VariationalAutoEncoder(pl.LightningModule):
    def __init__(self, 
                 latent_dimension,
                 variable_space,
                 steps):
        super().__init__()

        # class variables
        self.latent_dimension = latent_dimension
        self.variable_space = variable_space
        self.steps = steps

        # discrete steps on logarithmic scale
        factor = (self.latent_dimension/self.variable_space)**(1/self.steps)
        self.discrete_steps = [int(self.variable_space * (factor)**(k)) for k in range(self.steps)]
        # enforce the network to have the right latent dimensions
        self.discrete_steps[-1] = self.latent_dimension

        # build network
        encoder_list = []
        decoder_list = []
        for index, step in enumerate(self.discrete_steps):
            if index != (self.steps - 1):
                # encoder layers
                encoder_list.append(nn.Linear(in_features=step,
                                                out_features=self.discrete_steps[index+1]))
                encoder_list.append(nn.PReLU())
                # decoder layers
                decoder_list.append(nn.Linear(in_features=self.discrete_steps[steps-1-index],
                                                out_features=self.discrete_steps[steps - 2 - index]))
                decoder_list.append(nn.PReLU())

        # variational encoder structure (remove the last two layers in contrast two
        # vanilla encoder as we need them for reparameterization)
        self.encoder = nn.Sequential(*encoder_list[:-2])

        # define the addtional layers (mu and log_var) for reparameterization
        self.mu_layer = nn.Linear(in_features=self.discrete_steps[-2],
                                  out_features=self.discrete_steps[-1])
        self.log_var_layer = nn.Linear(in_features=self.discrete_steps[-2],
                                  out_features=self.discrete_steps[-1])

        # decoder structure
        # cut last activation function
        decoder_list = decoder_list[:-1]
        self.decoder = nn.Sequential(*decoder_list)
        
        # loss
        # total loss
        self.total_loss = []
        # loss criterion
        self.mse_criterion = nn.MSELoss(reduction="mean")
        # kullback leibler
        self.kullback_leibler_loss = 0
    
    def variational_encoding(self, x):
        # function for calling variational encoding 
        # forward pass
        encoded = self.encoder(x)
        return self.reparameterization(encoded=encoded)

    def reparameterization(self, encoded):
        # reparameterization like: z ~ (mu + std * epsilon)
        mu = self.mu_layer(encoded)
        log_var = self.log_var_layer(encoded)

        # noise from normal distributions
        epsilon = distributions.Normal(0,1).sample(sample_shape=mu.shape)
        
        # compute kullback-leibler-divergence (loss)
        self.kullback_leibler_loss = -0.5 * (1 + log_var - mu**2 - torch.exp(log_var)).sum()

        return mu + torch.exp(log_var) * epsilon

    def forward(self, x):
        # encoding and reparameterization
        variational_encoded = self.variational_encoding(x)

        # decoding
        decoded = self.decoder(variational_encoded)
        return decoded

    def configure_optimizers(self):
        optim = Adam(self.parameters())
        return optim

    def training_step(self, batch, batch_idx):

        # unpacking the batch
        x = batch

        # encode and decode via the model
        z = self.decoder(self.variational_encoding(x))

        # compare via mse and kullback-leibe
        self.loss = self.mse_criterion(z, x) + self.kullback_leibler_loss

        return self.loss

    def on_epoch_end(self):
        self.total_loss.append(self.loss.detach().numpy())
        return 