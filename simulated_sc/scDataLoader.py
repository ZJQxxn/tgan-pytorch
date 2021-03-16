import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import copy
import scipy.io
from scipy.sparse import csc_matrix, coo_matrix, csr_matrix


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
    # dset = SCData("../data/sc_simulated_data.npy", batch_size=12)
    # print(dset)
    # zero = dset[0]
    # print(type(zero))
    # print(zero.size())

    #
    all_data = []
    for f in [
        "../mouse_cell/gene_exp_mat_time_9_5_10k.mtx",
        "../mouse_cell/gene_exp_mat_time_10_5_10k.mtx",
        "../mouse_cell/gene_exp_mat_time_11_5_10k.mtx",
        "../mouse_cell/gene_exp_mat_time_12_5_10k.mtx",
        "../mouse_cell/gene_exp_mat_time_13_5_10k.mtx"
    ]:
        print("-" * 50)
        print("Filename : {}".format(f))
        sparse_mat = scipy.io.mmread(f)  # coo format
        sparse_mat = coo_matrix.transpose(sparse_mat)  # rows now correspond to samples
        sparse_mat = coo_matrix.tocsr(sparse_mat)  # easier to index into
        num_cells, num_genes = np.shape(sparse_mat)
        print("Num cells {} | Num genes {}".format(num_cells, num_genes))
        dense_mat = sparse_mat.toarray()
        all_data.append(dense_mat)
    print()
    all_data = np.concatenate(all_data)
    print(all_data.shape)

