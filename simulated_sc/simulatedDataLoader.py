import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import copy
import scipy.io
from scipy.sparse import csc_matrix, coo_matrix, csr_matrix


class simulatedData(Dataset):
    def __init__(self, dataset_path, time_points = 5):
        self.data = torch.from_numpy(np.float32(np.load(dataset_path)))
        # self.batch_size = batch_size
        self.data_shape = self.data.size()
        self.time_points = time_points

    def __len__(self):
        return self.data.size(1)

    def __getitem__(self, i):
        N = self.data.size(1)
        # ot = np.random.randint(N - self.batch_size) if N > self.batch_size else 0
        # x = self.data[:self.time_points, ot:(ot + self.batch_size), :]
        x = self.data[:self.time_points, i, :]
        return x

    def __str__(self):
        return "| DataLoader : simulated | Data Shape: {} time points, {} cells, {} genes |".format(
            self.data_shape[0], self.data_shape[1], self.data_shape[2])


if __name__ == "__main__":
    loader = simulatedData("../data/sc_simulated_data.npy", time_points = 5)
    print(loader)
    print(loader[0].shape)

