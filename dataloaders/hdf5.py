import h5py
import numpy as np


def hdf5_dataset_to_list(dataset):
    data = dataset[()]
    if isinstance(data, np.ndarray):
        data = data.tolist()
    return data


def hdf5_group_to_dict(group):
    result = {}
    for k, v in group.items():
        if isinstance(v, h5py.Dataset):
            result[k] = hdf5_dataset_to_list(v)
        elif isinstance(v, h5py.Group):
            result[k] = hdf5_group_to_dict(v)
    for k, v in group.attrs.items():
        if isinstance(v, np.ndarray):
            v = v.tolist()
        result[k] = v
    return result


def hdf5_to_dict(file_path):
    with h5py.File(file_path, 'r') as f:
        data = hdf5_group_to_dict(f)
    return data
