from datetime import datetime
import logging

import torch
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

from utils.tokenize import remove_comments, remove_blank_lines


def resize_embedding_and_tokenizer(model, tokenizer,
                                   special_tokens_dict: 'dict[str, str]' = {},
                                   custom_tokens: 'list[str]' = []):
    """
    Use mean initialization.
    NOTE: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    special_tokens_dict = {k: v for k, v in special_tokens_dict.items()
                           if getattr(tokenizer, k, None) is None}

    logging.info("Resizing tokenizer and embedding...")
    logging.info(f"Special tokens dict: {special_tokens_dict}")
    logging.info(f"Custom tokens: {custom_tokens}")

    new_tokens = list(special_tokens_dict.values()) + custom_tokens
    if len(new_tokens) <= 0:
        return False
    logging.info(f"Number of new tokens: {len(new_tokens)}")

    input_embedding = model.get_input_embeddings()
    input_inits = []
    for token in new_tokens:
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        value = input_embedding(torch.tensor(token_ids)).mean(dim=0, keepdim=True)
        input_inits.append(value)
    input_inits = torch.cat(input_inits, dim=0)
    output_inits = model.get_output_embeddings().weight.data.mean(dim=0, keepdim=True)

    if special_tokens_dict:
        tokenizer.add_special_tokens(special_tokens_dict)
    if custom_tokens:
        tokenizer.add_tokens(custom_tokens, special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    new_input_embedding = model.get_input_embeddings().weight.data
    new_input_embedding[-len(new_tokens):] = input_inits
    new_output_embedding = model.get_output_embeddings().weight.data
    new_output_embedding[-len(new_tokens):] = output_inits

    return True


def find_all_linear_names(model, int_bits=-1, add_lm_head=True):
    clazz = bnb.nn.Linear4bit if int_bits == 4 \
        else bnb.nn.Linear8bitLt if int_bits == 8 \
        else torch.nn.Linear
    linear_names = set()
    for name, module in model.named_modules():
        if isinstance(module, clazz):
            names = name.split('.')
            linear_names.add(names[0] if len(names) == 1 else names[-1])
    if add_lm_head and "lm_head" not in linear_names:
        logging.info("Adding lm_head to lora_module_names")
        linear_names.add("lm_head")
    return list(linear_names)


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


def setup_trainer(model, tokenizer, train_dataset, eval_dataset,
                  long_lora=False, is_resized=False, all_linear=False):
    output_dir = f"saved_models/lora-{datetime.now().strftime('%m-%d-%H-%M-%S')}"
    batch_size = 12
    per_device_train_batch_size = 6
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

    model = get_peft_model(model, peft_config)
    # model = torch.compile(model)
    model.print_trainable_parameters()

    train_args = SFTConfig(
        output_dir=output_dir,
        do_train=True,
        # train
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=5,
        warmup_steps=100,
        # max_steps=400,              # override `num_train_epochs`
        # optimize
        optim="adamw_bnb_8bit",     # adamw_torch & adamw_bnb_8bit
        learning_rate=3e-4,
        lr_scheduler_type="linear",
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
        group_by_length=True,
        # dtype
        bf16=True if torch.cuda.is_bf16_supported() else False,
        fp16=False if torch.cuda.is_bf16_supported() else True,
        # other
        gradient_checkpointing=True,
        # report
        report_to="tensorboard",        # wandb
        run_name=f"lora-{datetime.now().strftime('%m-%d-%H-%M-%S')}"
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=train_args,
        peft_config=peft_config,
        max_seq_length=2048,
    )
    return trainer
