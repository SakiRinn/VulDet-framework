import json


if __name__ == "__main__":
    instruction = '''
You are the best code auditor in the world, skilled at finding vulnerabilities in code. Review the given code in the input section in detail and very thoroughly. Think step by step very carefully.
Now, you need to answer the question: Does this code contain any vulnerabilities?
Your answer can only be 1 or 0, where 1 means there are some vulnerabilities in the given code and 0 means there are no vulnerabilities.
'''
    with open('data/devign.json', 'r') as f:
        raw_data = json.load(f)
        validate_split = int(0.1 * len(raw_data))
        data = []
        for d in raw_data:
            sample = {
                'instruction': instruction,
                'input': d['func'],
                'output': str(d['target'])
            }
            data.append(sample)
        with open('data/train.json', 'w') as ff:
            json.dump(data[validate_split:], ff, indent=4)
        with open('data/eval.json', 'w') as ff:
            json.dump(data[:validate_split], ff, indent=4)
