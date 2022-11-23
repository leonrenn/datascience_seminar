import torch
from torch.distributions import Normal
from torch.utils.data import Dataset


class Dataset_1d(Dataset):
    def __init__(self,
                 variable_space,
                 size,
                 generations):
        super().__init__()

        # class variable
        self.variable_space = variable_space
        self.size = size
        self.generations = generations

        # dataset
        self.dataset = {}

        for gen_idx in range(self.generations):
            # sample from gaussian
            distr = Normal(loc=0., scale=1.0).rsample((self.size,))
            data, _ = torch.histogram(distr, bins=self.variable_space, density=True)
            self.dataset[gen_idx] = data

    
    def __len__(self):
        return self.generations

    def __getitem__(self, index):
        return self.dataset[index]
