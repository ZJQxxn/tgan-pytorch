import numpy as np
import copy
import scipy.io
from scipy.sparse import csc_matrix, coo_matrix, csr_matrix

cell_num = []
all_data = []
for f in [
    "../mouse_cell/gene_exp_mat_time_9_5_100k.mtx",
    "../mouse_cell/gene_exp_mat_time_10_5_100k.mtx",
    "../mouse_cell/gene_exp_mat_time_11_5_100k.mtx",
    "../mouse_cell/gene_exp_mat_time_12_5_100k.mtx",
    "../mouse_cell/gene_exp_mat_time_13_5_100k.mtx"
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
    cell_num.append(dense_mat.shape[0])

print("="*50)
min_cell_num = np.min(cell_num)
print("The minimal number of cells is ", min_cell_num)
all_data = [each[:min_cell_num, :] for each in all_data]

all_data = np.array(all_data)
print("All data shape is ", all_data.shape)
np.save("./sampling_cell_data.npy", all_data)
print("Finished saving.")

# all_data = np.concatenate(all_data)
# print(all_data.shape)