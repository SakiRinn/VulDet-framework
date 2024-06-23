import json

import pandas as pd
from dataloaders.text import TextDataset, TextEntry


class DevignDataset(TextDataset):

    def __init__(self, file_path, tokenizer, is_train=False, validate_split=1., max_size=256):
        self.tokenizer = tokenizer
        self.max_size = max_size
        super().__init__(file_path, is_train, validate_split)

    def load(self, file_path):
        with open(file_path, 'r') as f:
            raw_data = json.load(f)
        return raw_data

    def preprocess(self, raw_data):
        data = []
        for e in raw_data:
            code = ' '.join(e['func'].split())
            tokens, ids = self.text_tokenize(code, self.tokenizer)
            entry = TextEntry(tokens, tokens, ids, e['target'])
            data.append(entry)
        return data
