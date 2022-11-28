import pytorch_lightning as pl
import torch.nn as nn
from torch.optim import Adam


class VanillaAutoEncoder(pl.LightningModule):
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

        # encoder structure
        self.encoder = nn.Sequential(*encoder_list)

        # decoder structure
        # cut last activation function
        decoder_list = decoder_list[:-1]
        self.decoder = nn.Sequential(*decoder_list)

        # total loss
        self.total_loss = []

        # criterion
        self.criterion = nn.MSELoss(reduction="mean")

    def forward(self, x):
        # encoding
        encoded = self.encoder(x)

        # decoding
        decoded = self.decoder(encoded)
        return decoded

    def configure_optimizers(self):
        optim = Adam(self.parameters())
        return optim

    def training_step(self, batch, batch_idx):

        # unpacking the batch
        x = batch

        # encode and decode via the model
        z = self.decoder(self.encoder(x))

        # compare via mse
        self.loss = self.criterion(z, x)

        return self.loss

    def on_epoch_end(self):
        self.total_loss.append(self.loss.detach().numpy())
        return 