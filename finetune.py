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
from finetune import resize_tokenizer_and_embedding, to_prompt, setup_model, setup_trainer
from peft.peft_model import PeftModel

import utils
from utils.huggingface import load_models

DEFAULT_PAD_TOKEN = "<pad>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def train(model_name_or_path):
    custom_tokens = []
    special_tokens_dict = {}

    _, model, tokenizer = load_models(model_name_or_path, bits=8)
    train_dataset = Dataset.from_json('data/devign/train.json').map(to_prompt)
    eval_dataset = Dataset.from_json('data/devign/eval.json').map(to_prompt)

    custom_tokens = [token.strip() for token in custom_tokens]
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
    if special_tokens_dict or custom_tokens:
        resize_tokenizer_and_embedding(model, tokenizer, special_tokens_dict, custom_tokens)

    model = setup_model(model)
    model.train()
    model.enable_input_require_grads()
    trainer = setup_trainer(model, tokenizer, train_dataset, eval_dataset)
    trainer.train(resume_from_checkpoint=None)


def eval(model_name_or_path):
    custom_tokens = []
    special_tokens_dict = {}

    _, model, tokenizer = load_models(model_name_or_path, bits=8)
    eval_dataset = Dataset.from_json('data/devign/eval.json').map(to_prompt)

    custom_tokens = [token.strip() for token in custom_tokens]
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
    if special_tokens_dict or custom_tokens:
        resize_tokenizer_and_embedding(model, tokenizer, special_tokens_dict, custom_tokens)

    model = setup_model(model)
    model.eval()

    dataset = eval_dataset.with_format('torch')
    dataloader = DataLoader(
        dataset,
        sampler=SequentialSampler(dataset),
        batch_size=4,
        num_workers=4,
        pin_memory=True)

    preds, labels = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_text = tokenizer.bos_token + batch['text'][0].split("\n### Output:\n")[0] + \
                "\n### Output:\n" + tokenizer.eos_token
            inputs = tokenizer(input_text, return_tensors="pt",
                               truncation=True, padding=True).to('cuda')
            outputs = model.generate(**inputs, max_new_tokens=16).squeeze()
            output_text = tokenizer.decode(outputs, skip_special_tokens=True)

            output_split = output_text.split("\n### Output:\n")
            pred_text = output_split[1] if len(output_split) > 1 else ''
            print(pred_text, file=sys.stderr)

            pred = np.array([1 if 'VULNERABLE' in pred_text else 0])
            label = np.array([1 if 'VULNERABLE' in batch['output'][0] else 0])
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

    _, base_model, tokenizer = load_models(base_model, bits=8)

    custom_tokens = []
    special_tokens_dict = {}

    custom_tokens = [token.strip() for token in custom_tokens]
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
    if special_tokens_dict or custom_tokens:
        resize_tokenizer_and_embedding(base_model, tokenizer, special_tokens_dict, custom_tokens)

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
    # train('meta-llama/Meta-Llama-3-8B')
    # lora_merge('saved_models/lora-07-17-03-54-57/checkpoint-3642',
    #            'meta-llama/Meta-Llama-3-8B',
    #            '/root/autodl-tmp/llama-3-merged')
    eval('/root/autodl-tmp/llama-3-merged')
