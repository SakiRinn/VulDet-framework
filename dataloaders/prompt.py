from dataloaders.tokenize import remove_blank_lines, remove_comments

TAG_TRUE = '[VULNERABLE]'
TAG_FALSE = '[BENIGN]'
INSTRUCTION = f'''You are the best code auditor in the world, skilled at finding vulnerabilities in code. Review the given code carefully and thoroughly to determine whether it is vulnerable. Your output can only be {TAG_TRUE} or {TAG_FALSE}, where {TAG_TRUE} means the code is vulnerable and {TAG_FALSE} means it is benign and non-vulnerable.'''


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
