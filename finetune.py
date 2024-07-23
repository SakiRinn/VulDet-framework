from datetime import datetime
import json
import logging
import os
import os.path as osp
import sys
import warnings
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, SequentialSampler
from datasets import Dataset
from peft.peft_model import PeftModel

import utils
from utils.finetune import find_all_linear_names, resize_embedding_and_tokenizer, train_prompt, eval_prompt
from utils.huggingface import load_transformers, DEFAULT_TOKENS

from peft import LoraConfig, get_peft_model
from peft.utils.save_and_load import get_peft_model_state_dict
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

from utils.misc import WarningCounter


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def setup_trainer(model, tokenizer, train_dataset, eval_dataset,
                  long_lora=False, is_resized=False, all_linear=False):
    output_dir = f"saved_models/llama3-{datetime.now().strftime('%m-%d-%H-%M-%S')}"
    batch_size = 8
    per_device_train_batch_size = 4
    gradient_accumulation_steps = batch_size // per_device_train_batch_size

    target_modules = None
    if all_linear:
        target_modules = find_all_linear_names(model)
        if is_resized:
            # Removing lm_head from target modules, will use in modules_to_save
            target_modules.pop(target_modules.index("lm_head"))

    modules_to_save = None
    if long_lora:
        modules_to_save = ["embed_tokens", "input_layernorm", "post_attention_layernorm", "norm"]
        if is_resized:
            modules_to_save += ["lm_head"]
    elif is_resized:
        modules_to_save = ["embed_tokens", "lm_head"]

    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
        modules_to_save=modules_to_save
    )
    # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
    if torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    train_args = SFTConfig(
        output_dir=output_dir,
        do_train=True,
        # train
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=2,
        # max_steps=400,              # override `num_train_epochs`
        # optimize
        optim="adamw_bnb_8bit",     # adamw_torch & adamw_bnb_8bit
        learning_rate=3e-4,
        lr_scheduler_type="linear",
        warmup_steps=100,
        weight_decay=0.,
        # eval
        # eval_strategy="steps",
        per_device_eval_batch_size=2 * per_device_train_batch_size,
        eval_steps=20,
        # load_best_model_at_end=True,
        # log & save
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="steps",
        save_steps=20,
        save_total_limit=5,
        # dataset
        dataset_text_field="text",
        dataloader_drop_last=True,
        group_by_length=False,
        # dtype
        bf16=True if torch.cuda.is_bf16_supported() else False,
        fp16=False if torch.cuda.is_bf16_supported() else True,
        # other
        gradient_checkpointing=True,
        # report
        report_to="tensorboard",        # wandb
        run_name=f"finetune-{datetime.now().strftime('%m-%d-%H-%M-%S')}"
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=train_args,
        peft_config=peft_config,
        data_collator=DataCollatorForCompletionOnlyLM(
            tokenizer.encode('\n### Output:\n', add_special_tokens=False)[1:-1],
            tokenizer=tokenizer,
            mlm=False),
        max_seq_length=2048,
    )
    return trainer


def train(model_name_or_path):
    resume_from_checkpoint = None
    custom_tokens = ['[VULNERABLE]', '[BENIGN]']

    custom_tokens = [token.strip() for token in custom_tokens]
    if osp.exists(model_name_or_path):
        adapter_config_path = osp.join(model_name_or_path, "adapter_config.json")
        with open(adapter_config_path) as f:
            adapter_config = json.load(f)
        base_model = adapter_config["base_model_name_or_path"]
        _, base_model, tokenizer = load_transformers(base_model, bits=8)
        is_resized = resize_embedding_and_tokenizer(base_model, tokenizer, DEFAULT_TOKENS, custom_tokens)
        model = PeftModel.from_pretrained(
            base_model,
            model_name_or_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map="auto"
        )
    else:
        _, model, tokenizer = load_transformers(model_name_or_path, bits=8)
        is_resized = resize_embedding_and_tokenizer(model, tokenizer, DEFAULT_TOKENS, custom_tokens)
    model.train()
    model.enable_input_require_grads()

    train_dataset = Dataset.from_json('data/devign/train.json').map(train_prompt)
    eval_dataset = Dataset.from_json('data/devign/eval.json').map(train_prompt)

    trainer = setup_trainer(model, tokenizer, train_dataset, eval_dataset,
                            is_resized=is_resized)
    with WarningCounter(
        match_text='Could not find response key',
        custom_message='The content after "### Output:" cannot be extracted due to truncation.'
    ):
        trainer.train(resume_from_checkpoint)


def eval(model_name_or_path):
    custom_tokens = ['[VULNERABLE]', '[BENIGN]']
    max_length = 2048
    lora_dir = ''

    _, model, tokenizer = load_transformers(model_name_or_path, bits=8)
    eval_dataset = Dataset.from_json('data/devign/eval.json').map(eval_prompt)

    custom_tokens = [token.strip() for token in custom_tokens]
    resize_embedding_and_tokenizer(model, tokenizer, DEFAULT_TOKENS, custom_tokens)

    model = PeftModel.from_pretrained(
        model,
        lora_dir,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto"
    )
    # model = model.merge_and_unload()
    model.eval()

    dataset = eval_dataset.with_format('torch')
    dataloader = DataLoader(
        dataset,
        sampler=SequentialSampler(dataset),
        batch_size=6,
        num_workers=8,
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
    resize_embedding_and_tokenizer(base_model, tokenizer, DEFAULT_TOKENS, custom_tokens)

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
    train('meta-llama/Meta-Llama-3-8B')
    # lora_merge('saved_models/lora-07-19-06-52-44/checkpoint-9105',
    #            'meta-llama/Llama-2-7b-hf',
    #            '/root/autodl-tmp/llama-3-merged')
    # eval('meta-llama/Meta-Llama-3-8B')
