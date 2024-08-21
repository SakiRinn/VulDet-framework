import os
import sys
import json

root_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(root_dir)
sys.path.append(root_dir)
from dataloaders.prompt import INSTRUCTION, TAG_FALSE, TAG_TRUE


if __name__ == "__main__":
    with open('data/devign/devign.json', 'r') as f:
        raw_data = json.load(f)
        test_split = int((1 - 0.2) * len(raw_data))
        data = []
        for d in raw_data:
            sample = {
                'instruction': INSTRUCTION,
                'input': d['func'],
                'output': TAG_TRUE if d['target'] != 0 else TAG_FALSE,
            }
            data.append(sample)
        with open('data/devign/train.json', 'w') as ff:
            json.dump(data[:test_split], ff, indent=4)
        with open('data/devign/eval.json', 'w') as ff:
            json.dump(data[test_split:], ff, indent=4)
        with open('data/devign/all.json', 'w') as ff:
            json.dump(data, ff, indent=4)
