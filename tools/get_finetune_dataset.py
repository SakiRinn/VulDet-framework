import json

from dataloaders import INSTRUCTION


if __name__ == "__main__":
    with open('data/devign/devign.json', 'r') as f:
        raw_data = json.load(f)
        test_split = int((1 - 0.2) * len(raw_data))
        data = []
        for d in raw_data:
            sample = {
                'instruction': INSTRUCTION,
                'input': d['func'],
                'output': '[VULNERABLE]' if d['target'] != 0 else '[BENIGN]',
            }
            data.append(sample)
        with open('data/devign/train.json', 'w') as ff:
            json.dump(data[:test_split], ff, indent=4)
        with open('data/devign/eval.json', 'w') as ff:
            json.dump(data[test_split:], ff, indent=4)
