import gzip
import h5py
import numpy as np
import pickle as pkl


def hdf5_dataset_to_list(dataset):
    data = dataset[()]
    if isinstance(data, np.ndarray):
        data = data.tolist()
    return data

def hdf5_group_to_records(group):
    result = {}
    for k, v in group.items():
        if isinstance(v, h5py.Dataset):
            result[k] = hdf5_dataset_to_list(v)
        elif isinstance(v, h5py.Group):
            result[k] = hdf5_group_to_records(v)
    for k, v in group.attrs.items():
        if isinstance(v, np.ndarray):
            v = v.tolist()
        result[k] = v
    return result

def hdf5_to_records(file_path):
    with h5py.File(file_path, 'r') as f:
        data = hdf5_group_to_records(f)
    return data


def read_file(file):
    data = []
    while file.peek(1):
        obj = pkl.load(file)
        data.append(obj)
    return data

def pkl_to_records(file_path, with_gz=False):
    if with_gz:
        with gzip.open(file_path, 'rb') as f:
            return read_file(f)
    else:
        with open(file_path, 'rb') as f:
            return read_file(f)
