import numpy as np
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, x_range_lower: float = 0.0,
                        x_range_upper: float = 1.0,
                        trained_params: int = 10,
                       data_points: int = 1000,
                       total_generations: int = 10000):
        """Generates data for NN training. Output is the parameterized data
        X and the labels Y per dataset index. 

        Args:
            x_range_lower (float, optional): Lower border of data training. Defaults to 0.0.
            x_range_upper (float, optional): Upper border of data training. Defaults to 1.0.
            trained_params (int, optional): Devides range from lower to upper bound into equal segments to 
            improve scaling performance on the training range. Defaults to 10.
            data_points (int, optional): Amount of data points generated per each event. Defaults to 1000.
            total_generations (int, optional): Amount of events that the NN is trained on. Defaults to 10000.
        """
        # seeding for ensuring to always generate the same data
        np.random.seed(42)
        
        # default class variables
        self.x_range_lower = x_range_lower
        self.x_range_upper = x_range_upper
        self.trained_params = trained_params
        self.data_points = data_points
        self.total_generations = total_generations

        # hypothesis
        hypothesis = np.linspace(start=self.x_range_lower,
                                 stop=self.x_range_upper,
                                 num=self.trained_params)


        # empty dataset holding the generations of events
        self.dataset = {}

        # vector that contains the targets for binary classification (0,...,0,1,...,1) and transform to tensor
        Y = np.concatenate([np.zeros(self.data_points), np.ones(self.data_points)])
        Y = torch.tensor(Y, requires_grad=True).float()

        # iterating over amount of generations and storing
        # converted tensors in dataset
        for gen_idx in range(self.total_generations):

            # empty vector that holds the data from the events from one generation (x, H_bg, H_signal)
            X = np.zeros(shape=(self.data_points * 2, 3))

            # random selection of hypothesis
            background, signal = np.random.choice(hypothesis,2)

            # event generation for hypothesis
            X[:self.data_points, 0] = np.random.normal(loc = background, scale = 0.02, size = self.data_points)
            X[self.data_points:, 0] = np.random.normal(loc = signal, scale = 0.02, size = self.data_points)

            # hypothesis for parameterizing the NN
            X[:, 1] = np.ones(self.data_points * 2) * background
            X[:, 2] = np.ones(self.data_points * 2) * signal


            # make tensors of data and parameters
            X = torch.tensor(X, requires_grad=True).float()
    

            self.dataset[gen_idx] = (X, Y)

    def __len__(self):
        return self.total_generations # len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index][0], self.dataset[index][1]