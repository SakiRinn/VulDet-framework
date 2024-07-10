import json
import logging
import os
import os.path as osp

from peft.mapping import get_peft_model
from peft.peft_model import PeftModel
import torch
import bitsandbytes as bnb
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training
)
from utils.huggingface import load_models, load_datasets
from transformers import TrainingArguments, DataCollatorForSeq2Seq, Trainer
from trl import SFTTrainer

from tools.huggingface import load_models
from tools.metric import Metric


def resize_tokenizer_and_embedding(model, tokenizer,
                                   special_tokens_dict: 'dict[str, str]' = {},
                                   custom_tokens: 'list[str]' = []):
    """
    NOTE: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """

    if not special_tokens_dict and not custom_tokens:
        return ValueError("Parameter `special_tokens_dict` and `custom_tokens cannot` can't be empty at the same time.")

    logging.info("Resizing tokenizer and embedding...")
    logging.info("Special tokens dict: %s", special_tokens_dict)
    logging.info("Custom tokens: %s", custom_tokens)

    if special_tokens_dict:
        tokenizer.add_special_tokens(special_tokens_dict)
    if custom_tokens:
        tokenizer.add_tokens(custom_tokens, special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    num_new_tokens = len(list(special_tokens_dict.keys())) + (0 if custom_tokens is None else len(custom_tokens))
    logging.info("Number of new tokens: %d", num_new_tokens)

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def find_all_linear_names(model, int_bits=-1, add_lm_head=True):
    clazz = bnb.nn.Linear4bit if int_bits == 4 \
        else bnb.nn.Linear8bitLt if int_bits == 8 \
        else torch.nn.Linear
    linear_names = set()
    for name, module in model.named_modules():
        if isinstance(module, clazz):
            names = name.split('.')
            linear_names.add(names[0] if len(names) == 1 else names[-1])
    if add_lm_head and not "lm_head" in linear_names:
        logging.info("Adding lm_head to lora_module_names")
        linear_names.add("lm_head")
    return list(linear_names)


def to_prompt(sample):
    prompt = f"""
    ### Instruction:
        {sample["instruction"]}
    ### Input:
        {sample["input"]}
    ### Output:
        {sample["output"]}
    """
    return {'text': '<s>' + prompt + '</s>'}


def compute_metrics(preds):
    logits, labels = preds
    metrics = Metric(logits, labels)()
    return metrics


def setup_model(model):
    model.train()
    model.config.use_cache = False
    # old_state_dict = model.state_dict
    # model.state_dict = (lambda self, *_, **__:
    #                     get_peft_model_state_dict(self, old_state_dict())).__get__(model, type(model))
    # model = torch.compile(model)
    return model


def setup_trainer(model, tokenizer, train_dataset, eval_dataset,
                  long_lora=False, is_resized=False, all_linear=False):
    output_dir = "code-llama"
    batch_size = 128
    per_device_train_batch_size = 16
    gradient_accumulation_steps = batch_size // per_device_train_batch_size

    target_modules = None       # default
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
        lora_alpha=16,      # 2*r
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
    model = get_peft_model(model, peft_config)

    train_args = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        do_eval=True,
        # train
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=False,
        num_train_epochs=1,
        warmup_steps=100,
        max_steps=400,      # override `num_train_epochs`
        # optimize
        optim="adamw_torch",
        learning_rate=3e-4,
        lr_scheduler_type="linear",
        weight_decay=0.,
        # eval
        per_device_eval_batch_size=2*per_device_train_batch_size,
        eval_strategy="steps",
        eval_steps=20,
        # save
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="steps",
        save_steps=20,
        save_total_limit=10,
        # dtype
        bf16=True if torch.cuda.is_bf16_supported() else False,
        fp16=False if torch.cuda.is_bf16_supported() else True,
        # other
        dataloader_drop_last=True,
        load_best_model_at_end=True,
        group_by_length=True,
        report_to="tensorboard",
        # wandb (optional)
        # run_name=f"codellama-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=train_args,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=512,
        # compute_metrics=compute_metrics
    )
    return trainer


def lora_merge(lora_dir, base_model='', output_dir='output/lora_merge'):
    lora_dir = osp.realpath(lora_dir)
    output_dir = osp.realpath(output_dir)

    if base_model:
        logging.info("Using base model %s", base_model)
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
