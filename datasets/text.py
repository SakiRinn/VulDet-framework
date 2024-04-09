from abc import abstractmethod
import torch

from datasets.base import BaseDataset, DataEntry


class TextEntry(DataEntry):

    def __init__(self, idx, tokens, ids, label=None):
        super().__init__(idx, label)
        self.tokens = tokens
        self.ids = ids

    def __str__(self):
        string = super().__str__()
        string += "input_tokens: {}\n".format([x.replace('\u0120', '_') for x in self.tokens])
        string += "input_ids: {}\n".format(' '.join(map(str, self.ids)))
        return string


class TextDataset(BaseDataset):

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx].ids), torch.tensor(self.data[idx].label)

    @staticmethod
    def text_tokenize(text, tokenizer, max_size=256):
        tokens = tokenizer.tokenize(text)[:max_size - 2]
        tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
        ids = tokenizer.convert_tokens_to_ids(tokens)
        padding_length = max_size - len(ids)
        ids += [tokenizer.pad_token_id] * padding_length
        return tokens, ids
