import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class SCData(Dataset): #TODO: change this class
    def __init__(self, dataset_path, batch_size = 32):
        self.data = torch.from_numpy(np.float32(np.load(dataset_path)))
        self.batch_size = batch_size
        self.data_shape = self.data.size()

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, i):
        N = self.data.size(1)
        ot = np.random.randint(N - self.batch_size) if N > self.batch_size else 0
        x = self.data[i, ot:(ot + self.batch_size)] #TODO: 怎么选择不同的time point？
        return x

    def __str__(self):
        return "| DataLoader : scRNA-seq | Data Shape: {} time points, {} cells, {} genes |".format(
            self.data_shape[0], self.data_shape[1], self.data_shape[2])


if __name__ == "__main__":
    dset = SCData("../data/sc_simulated_data.npy", batch_size=12)
    print(dset)
    zero = dset[0]
    print(type(zero))
    print(zero.size())
