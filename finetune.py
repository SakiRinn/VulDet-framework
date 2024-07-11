import json
import logging
import os
import os.path as osp
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler
from datasets import load_dataset
from finetune import resize_tokenizer_and_embedding, generate_prompt, setup_model, setup_trainer
from peft.peft_model import PeftModel

import utils
from utils.huggingface import load_models
from utils.runner import Runner, get_param_names

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def train(model_name_or_path):
    custom_tokens = []
    special_tokens_dict = {}

    _, model, tokenizer = load_models(model_name_or_path, int_bits=8)
    train_dataset = load_dataset('json', data_files='data/train.json', split='train').map(generate_prompt)
    eval_dataset = load_dataset('json', data_files='data/eval.json', split='train').map(generate_prompt)

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
    trainer = setup_trainer(model, tokenizer, train_dataset, eval_dataset)
    trainer.train(resume_from_checkpoint=None)


def eval(model_name_or_path):
    custom_tokens = []
    special_tokens_dict = {}

    _, model, tokenizer = load_models(model_name_or_path, int_bits=8)
    eval_dataset = load_dataset('json', data_files='data/eval.json', split='train').map(generate_prompt)

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
        batch_size=1,
        num_workers=4,
        pin_memory=True)

    preds, labels = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_text = tokenizer.bos_token + batch['text'][0].split("\n### Output:\n")[0] + \
                "\n### Output:\n" + tokenizer.eos_token
            inputs = tokenizer(input_text, return_tensors="pt").to('cuda')
            outputs = model.generate(**inputs, max_new_tokens=64)[0]
            output_text = tokenizer.decode(outputs, skip_special_tokens=True)

            output_split = output_text.split("\n### Output:\n")
            pred_text = output_split[1] if len(output_split) > 1 else ''

            pred = np.array([1 if '<YES>' in pred_text else 0])
            label = np.array([1 if '<YES>' in batch['output'][0] else 0])
            preds.append(pred)
            labels.append(label)

    preds = np.concatenate(preds, 0)
    labels = np.concatenate(labels, 0)

    results = utils.Metric(preds, labels)()
    print(results)
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

    _, model, tokenizer = load_models(base_model)
    lora_model = PeftModel.from_pretrained(
        model,
        lora_dir,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto"
    )

    logging.info("Merging model...")
    lora_model = lora_model.merge_and_unload()

    logging.info("Merge complete, saving model to %s ...", output_dir)
    os.makedirs(output_dir, exist_ok=True)
    lora_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    Runner.set_seed(42)
    # train('codellama/CodeLlama-7b-hf')
    # lora_merge('llm/codellama-07-11-09-18-54/checkpoint-520',
    #            'codellama/CodeLlama-7b-hf',
    #            'llm/codellama-520')
    eval('llm/codellama-520')
