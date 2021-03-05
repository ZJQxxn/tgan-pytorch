import numpy as np
import pickle
import copy

def _update(prev, size, coeff = 1.0):
    return prev + coeff * np.random.normal(size = size)


def generateData(num_time_pts, num_cells, num_genes):
    '''
    Simulate time series data.
    :param num_time_pts: The number of time points.
    :param num_cells: The number of samples.
    :param num_genes: The number of features.
    :return: Simulated dataset (numpy.ndarray)
    '''

    # Initialization
    size = (num_cells, num_genes)
    every_step_data = np.random.uniform(-1.0, 1.0, size = size)
    all_data = [copy.deepcopy(every_step_data)]
    # Every following steps
    for t in range(1, num_time_pts):
        every_step_data = _update(every_step_data, size, coeff = 0.5)
        all_data.append(copy.deepcopy(every_step_data))
    return np.array(all_data)


if __name__ == '__main__':
    data = generateData(num_time_pts = 10, num_cells = 1000, num_genes = 1000)
    print("Data shape : ", data.shape)
    np.save("../data/sc_simulated_data.npy", data)