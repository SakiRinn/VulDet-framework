import json
import logging
import os
import os.path as osp
import sys
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, SequentialSampler
from datasets import Dataset
from peft.peft_model import PeftModel

import utils
from utils.llm import resize_embedding_and_tokenizer, train_prompt, eval_prompt, setup_trainer
from utils.huggingface import load_transformers, DEFAULT_TOKENS

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def train(model_name_or_path):
    custom_tokens = []

    _, model, tokenizer = load_transformers(model_name_or_path, bits=8)
    train_dataset = Dataset.from_json('data/devign/train.json').map(train_prompt)
    eval_dataset = Dataset.from_json('data/devign/eval.json').map(train_prompt)

    custom_tokens = [token.strip() for token in custom_tokens]
    is_resized = resize_embedding_and_tokenizer(model, tokenizer, DEFAULT_TOKENS, custom_tokens)

    model.train()
    model.enable_input_require_grads()
    trainer = setup_trainer(model, tokenizer, train_dataset, eval_dataset,
                            is_resized=is_resized)
    trainer.train(resume_from_checkpoint=None)


def eval(model_name_or_path):
    custom_tokens = []
    max_length = 2048

    _, model, tokenizer = load_transformers(model_name_or_path, bits=8)
    eval_dataset = Dataset.from_json('data/devign/eval.json').map(eval_prompt)

    custom_tokens = [token.strip() for token in custom_tokens]
    resize_embedding_and_tokenizer(model, tokenizer, DEFAULT_TOKENS, custom_tokens)

    model.eval()

    dataset = eval_dataset.with_format('torch')
    dataloader = DataLoader(
        dataset,
        sampler=SequentialSampler(dataset),
        batch_size=6,
        num_workers=4,
        pin_memory=True)

    preds, labels = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            len_filter = [i for i, t in enumerate(batch['text']) if len(t) < max_length]
            label = np.array([1 if 'VULNERABLE' in t else 0
                              for t in batch['output']])[len_filter]

            input_texts = [t for t in batch['text'] if len(t) < max_length]
            inputs = tokenizer(input_texts, return_tensors="pt",
                               truncation=True, padding=True).to('cuda')
            outputs = model.generate(**inputs, max_new_tokens=16)
            output_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            pred_texts = [t.split("### Output:\n")[1].strip() for t in output_texts]
            pred = np.array([1 if 'VULNERABLE' in t else 0 for t in pred_texts])

            for t in pred_texts:
                print(f'### OUTPUTS:\n{t}\n', file=sys.stderr)

            preds.append(pred)
            labels.append(label)

    preds = np.concatenate(preds, 0)
    labels = np.concatenate(labels, 0)

    results = utils.Metric(preds, labels)()
    with open("eval.log", "w") as f:
        print(results, file=f)
        print(results, file=sys.stderr)
    return results


def lora_merge(lora_dir, base_model='', output_dir='output/lora_merge'):
    custom_tokens = []

    lora_dir = osp.realpath(lora_dir)
    output_dir = osp.realpath(output_dir)

    if base_model:
        logging.info(f"Using base model {base_model}")
    else:
        adapter_config_path = osp.join(lora_dir, "adapter_config.json")
        with open(adapter_config_path) as f:
            adapter_config = json.load(f)
        base_model = adapter_config["base_model_name_or_path"]
        logging.info(f"Base model not given, using {base_model}")

    _, base_model, tokenizer = load_transformers(base_model, bits=8)

    custom_tokens = [token.strip() for token in custom_tokens]
    resize_embedding_and_tokenizer(model, tokenizer, DEFAULT_TOKENS, custom_tokens)

    model = PeftModel.from_pretrained(
        base_model,
        lora_dir,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto"
    )

    logging.info("Merging model...")
    model = model.merge_and_unload()

    logging.info(f"Merge complete, saving model to {output_dir} ...", )
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    utils.Runner.setup_seed(114514)
    train('meta-llama/Llama-2-7b-hf')
    # lora_merge('saved_models/lora-07-17-02-59-11/checkpoint-3642',
    #            'meta-llama/Llama-2-7b-hf',
    #            '/root/autodl-tmp/llama-3-merged')
    # eval('/root/autodl-tmp/llama-3-merged')
