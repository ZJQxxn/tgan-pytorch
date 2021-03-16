Hi Jiaqi,
Here are the sparse gene expression matrices for each of the five time points (9.5, 10.5, 11.5, 12.5, 13.5) in the mouse cell atlas study (https://www.nature.com/articles/s41586-019-0969-x).

They have been subsampled so that there are 10000 cells across all time points. If you want the larger version with 100000 subsampled cells, just let me know.

To read the .mtx files, here is an example from my code:

---

import scipy.io
from scipy.sparse import csc_matrix, coo_matrix, csr_matrix

	...

sparse_mat = scipy.io.mmread(filename) #coo format
sparse_mat = coo_matrix.transpose(sparse_mat) #rows now correspond to samples
sparse_mat = coo_matrix.tocsr(sparse_mat) #easier to index into
num_cells, num_genes = np.shape(sparse_mat)

---

Best,
Jeremy