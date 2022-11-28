import random
from typing import Union

import torch
from torch.distributions import Normal
from torch.utils.data import Dataset


class Dataset_1d(Dataset):
    def __init__(self,
                 variable_space: int,
                 size: int,
                 generations: int,
                 random_scaling: Union[bool, float] = True):
        super().__init__()

        # class variable
        self.variable_space = variable_space
        self.size = size
        self.generations = generations

        # dataset
        self.dataset = {}

        # variable scale -> this should be true for training and of for 
        # visualization of the same data
        if random_scaling is True:
            scales = torch.linspace(0.1, 1, 10) # scaling from 0.1 to 1
        else:
            scales = [random_scaling]

        for gen_idx in range(self.generations):
            # select random selection
            scale = random.choice(scales)
            # sample from gaussian
            distr = Normal(loc=0., scale=scale).rsample((self.size,))
            data, _ = torch.histogram(distr, bins=self.variable_space, density=True)
            self.dataset[gen_idx] = data

    
    def __len__(self):
        return self.generations

    def __getitem__(self, index):
        return self.dataset[index]
