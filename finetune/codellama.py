import logging
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


def resize_tokenizer_and_embedding(special_tokens_dict, tokenizer, model,
                                   custom_tokens: 'list[str] | None' = None):
    """
    NOTE: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """

    if len(list(special_tokens_dict.keys())) > 0 or custom_tokens is not None:
        logging.info("Resizing tokenizer and embedding...")
        logging.info("Special tokens dict: %s", special_tokens_dict)
        logging.info("Custom tokens: %s", custom_tokens)
    else:
        return False

    if len(list(special_tokens_dict.keys())) > 0:
        tokenizer.add_special_tokens(special_tokens_dict)
    if custom_tokens is not None:
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

    return True


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


def find_all_linear_names(model, int_bits=-1, add_lm_head=True):
    clazz = bnb.nn.Linear4bit if int_bits == 4 \
        else bnb.nn.Linear8bitLt if int_bits == 8 \
        else torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, clazz):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if add_lm_head and not "lm_head" in lora_module_names:
        logging.info("Adding lm_head to lora_module_names")
        lora_module_names.add("lm_head")

    return list(lora_module_names)


def setup_peft_model(model):
    config = LoraConfig(
        task_type="CAUSAL_LM",
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ]
    )
    model.train()
    model = get_peft_model(model, config)
    # Pytorch optimization
    model.config.use_cache = False
    old_state_dict = model.state_dict
    model.state_dict = (lambda self, *_, **__:
        get_peft_model_state_dict(self, old_state_dict())).__get__(model, type(model))
    model = torch.compile(model)
    return model


def setup_peft_trainer(model, tokenizer, train_dataset, eval_dataset):
    output_dir = "code-llama"
    batch_size = 128
    per_device_train_batch_size = 16
    gradient_accumulation_steps = batch_size // per_device_train_batch_size

    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ]
    )
    train_args = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        do_eval=True,
        # train
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=1,
        warmup_steps=100,
        max_steps=400,      # override `num_train_epochs`
        # optimize
        optim="adamw_torch",
        learning_rate=3e-4,
        lr_scheduler_type="linear",
        weight_decay=0.,
        # eval
        per_device_eval_batch_size=2 * per_device_train_batch_size,
        eval_strategy="steps",
        eval_steps=20,
        # save
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="steps",
        save_steps=20,
        save_total_limit=10,
        # other
        dataloader_drop_last=True,
        load_best_model_at_end=True,
        group_by_length=True,
        bf16=True if torch.cuda.is_bf16_supported() else False,
        fp16=False if torch.cuda.is_bf16_supported() else True,
        # wandb (optional)
        # report_to="wandb",
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
        max_seq_length=512
    )
    return trainer
