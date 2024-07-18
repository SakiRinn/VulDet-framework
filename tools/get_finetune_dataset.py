import json


if __name__ == "__main__":
    instruction = '''You are the best code auditor in the world, skilled at finding vulnerabilities in code. Review the given code carefully and thoroughly to determine whether it is vulnerable. Your output can only be [VULNERABLE] or [BENIGN], where [VULNERABLE] means the code is vulnerable and [BENIGN] means it is benign and non-vulnerable.'''
    with open('data/devign/devign.json', 'r') as f:
        raw_data = json.load(f)
        validate_split = int(0.2 * len(raw_data))
        data = []
        for d in raw_data:
            sample = {
                'instruction': instruction,
                'input': d['func'],
                'output': '[VULNERABLE]' if d['target'] != 0 else '[BENIGN]',
            }
            data.append(sample)
        with open('data/devign/train.json', 'w') as ff:
            json.dump(data[validate_split:], ff, indent=4)
        with open('data/devign/eval.json', 'w') as ff:
            json.dump(data[:validate_split], ff, indent=4)
