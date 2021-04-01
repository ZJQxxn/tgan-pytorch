import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import copy
import scipy.io
from scipy.sparse import csc_matrix, coo_matrix, csr_matrix


class SCData(Dataset):
    def __init__(self, dataset_path):
        self.data = torch.from_numpy(np.float32(np.load(dataset_path)))
        # self.batch_size = batch_size
        self.data_shape = self.data.size()
        self.time_points = self.data_shape[0]

    def __len__(self):
        return self.data.size(1)

    def __getitem__(self, i):
        N = self.data.size(1)
        # ot = np.random.randint(N - self.batch_size) if N > self.batch_size else 0
        # x = self.data[:self.time_points, ot:(ot + self.batch_size), :]
        x = self.data[:self.time_points, i, :]
        return x

    def __str__(self):
        return "| DataLoader : mouse cell | Data Shape: {} time points, {} cells, {} genes |".format(
            self.data_shape[0], self.data_shape[1], self.data_shape[2])


if __name__ == "__main__":
    loader = SCData("../mouse_cell/sampling_cell_data.npy")
    print(loader)
    print(loader[0].shape)


