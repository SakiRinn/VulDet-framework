from dataloaders.tokenize import remove_blank_lines, remove_comments


TRUE_TAG = '[VULNERABLE]'
FALSE_TAG = '[BENIGN]'
INSTRUCTION = f'''You are the best code auditor in the world, skilled at finding vulnerabilities in code. Review the given code carefully and thoroughly to determine whether it is vulnerable. Your output can only be {TRUE_TAG} or {FALSE_TAG}, where {TRUE_TAG} means the code is vulnerable and {FALSE_TAG} means it is benign and non-vulnerable.'''


def train_prompt(sample):
    code = sample['input'].strip()
    code = remove_comments(code)
    code = remove_blank_lines(code)
    prompt = f"### Instruction:\n{sample['instruction']}\n" \
        f"\n### Input:\n{code}\n" \
        f"\n### Output:\n{sample['output']}"
    return {'text': prompt}


def eval_prompt(sample):
    code = sample['input'].strip()
    code = remove_comments(code)
    code = remove_blank_lines(code)
    prompt = f"### Instruction:\n{sample['instruction']}\n" \
        f"\n### Input:\n{code}\n" \
        f"\n### Output:\n"
    return {'text': prompt}
