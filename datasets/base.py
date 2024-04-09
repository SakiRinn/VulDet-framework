from abc import ABCMeta, abstractmethod
from torch.utils.data import Dataset



class DataEntry(metaclass=ABCMeta):

    def __init__(self, idx, label=None):
        self.idx = idx
        self.label = label

    def __str__(self):
        string = f"*** Sample {self.idx} ***\n" + \
                 f"label: {self.label}"
        return string


class BaseDataset(Dataset, metaclass=ABCMeta):

    def __init__(self, file_path, is_train=False, validate_split=1.):
        self.validate_split = validate_split

        raw_data = self.load_data(file_path) if file_path else []
        self.data = self.preprocess(raw_data)

        split_idx = int(len(self.data) * (1 - self.validate_split))
        self.data = self.data[:split_idx] if is_train else self.data[split_idx:]

    def __len__(self):
        return len(self.data)

    @abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError

    @abstractmethod
    def load_data(self, file_path: str) -> list:
        raise NotImplementedError

    @abstractmethod
    def preprocess(self, raw_data) -> list[DataEntry]:
        raise NotImplementedError
