import numpy as np
from torch.utils.data import Dataset


class ExampleDataset(Dataset):
    def __init__(self,
                 number_of_points):
        super().__init__()

        # class variable
        self.number_of_points = number_of_points

        # data generation from two different gaussians
        n = int(self.number_of_points / 2)
        gaussian1 = np.random.normal(loc=-1, scale=0.25, size=(n,))
        gaussian2 = np.random.normal(
            loc=0.5, scale=0.5, size=(self.number_of_points-n,))
        # concatenation equal parts of the distributions
        self.array = np.concatenate([gaussian1, gaussian2])
        
    def __len__(self):
        return len(self.array)

    def __getitem__(self, index):
        return self.array[index]
