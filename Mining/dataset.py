import torch
from torch.distributions import Normal
from torch.utils.data import Dataset


class GoldGenerator(Dataset):
    def __init__(self) -> None:
        super().__init__()
        # seed
        torch.random.seed()

        # dataset
        self.dataset = {}

        # num of points and generations
        self.data_points = 100
        self.generations = 100

        # theta hypothesis
        param_size = 1
        theta_0 = 0.0
        theta_1 = torch.linspace(1, 1, param_size)

        # std for data generation
        x_std = 0.1
        z_std = 1.0

        for gen_idx in range(self.generations):
            # select random theta_1
            theta_1_param = theta_1[torch.randint(0, param_size, size=(1,))[0]]

            # distributions
            dist_0 = Normal(loc=theta_0, scale=z_std)
            dist_1 = Normal(loc=theta_1_param, scale=z_std)

            # target labels and parameter thetas
            theta_0_tensor = torch.zeros(self.data_points)
            theta_1_tensor = torch.ones(self.data_points)
            y = torch.vstack([theta_0_tensor[:int(self.data_points/2)],
                              theta_1_tensor[:int(self.data_points/2)]])

            # empty tensors
            X = torch.empty((self.data_points, 3))
            Y = torch.empty((self.data_points, 2))

            # from theta_0
            z0 = dist_0.rsample((int(self.data_points/2),))
            # from theta_1
            z1 = dist_1.rsample((int(self.data_points/2),))
            Z = torch.vstack([z0, z1])

            # from z0
            x0 = Normal(loc=z0, scale=x_std).rsample((1,))
            # from z1
            x1 = Normal(loc=z1, scale=x_std).rsample((1,))
            x = torch.hstack([x0, x1]).reshape((self.data_points,))
            X[:, 0] = x
            X[:, 1] = theta_0_tensor
            X[:, 2] = theta_1_tensor * theta_1_param  # for paramterization

            # ratio calculation
            r_xz = dist_0.log_prob(Z) - dist_1.log_prob(Z)

            # targets
            Y[:, 0] = y.reshape((self.data_points,))
            Y[:, 1] = r_xz.reshape((self.data_points,))

            self.dataset[gen_idx] = (X, Y)

    def __len__(self):
        return self.generations

    def __getitem__(self, index):
        return self.dataset[index][0], self.dataset[index][1]
