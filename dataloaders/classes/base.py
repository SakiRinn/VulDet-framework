from abc import ABCMeta, abstractmethod

from torch.utils.data import Dataset, Subset


class BaseDataset(Dataset, metaclass=ABCMeta):

    def __init__(self, file_path):
        self.data = self.load(file_path)

    def __len__(self):
        return len(self.data)

    @abstractmethod
    def __getitem__(self, idx):
        pass

    @abstractmethod
    def load(self, file_path) -> 'list':
        pass

    def train_test_split(self, test_size=0.2):
        if test_size < 0:
            raise ValueError("Parameter `test_size` must be greater than 0.")
        split_idx = int(len(self.data) * (1. - test_size)) if test_size < 1. \
            else len(self.data) - int(test_size)
        indices = list(range(len(self.data)))
        return {
            'train': Subset(self, indices[:split_idx]),
            'test': Subset(self, indices[split_idx:])
        }
