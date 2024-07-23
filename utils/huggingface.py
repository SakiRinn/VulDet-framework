import torch
from datasets import load_dataset
from transformers import (get_linear_schedule_with_warmup,
                          get_cosine_schedule_with_warmup,
                          get_polynomial_decay_schedule_with_warmup,
                          get_constant_schedule,
                          get_inverse_sqrt_schedule)
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
from peft.utils.peft_types import TaskType


SCHEDULER_TYPES = {
    'linear': get_linear_schedule_with_warmup,
    'cosine': get_cosine_schedule_with_warmup,
    'polymonial': get_polynomial_decay_schedule_with_warmup,
    'constant': get_constant_schedule,
    'inverse_sqrt': get_inverse_sqrt_schedule,
}
PEFT_TASK_TYPES = [t.value for t in list(TaskType)]

DEFAULT_TOKENS = {
    "pad_token": "<pad>",
    "unk_token": "<unk>",
    "bos_token": "<s>",
    "eos_token": "</s>",
}


def get_quantization_config(bits=-1):
    if bits == 4:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True
        )
    elif bits == 8:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
    else:
        bnb_config = None
    return bnb_config


def load_datasets(dataset_name_or_path: str, validate_split=0.1):
    dataset = load_dataset(dataset_name_or_path)
    train_dataset, eval_dataset = dataset.train_test_split(test_size=validate_split).values()
    return train_dataset, eval_dataset


def load_transformers(model_name_or_path: str, config_name='', tokenizer_name='',
                      do_lower_case=False, bits=-1):
    config_name = model_name_or_path if not config_name else config_name
    tokenizer_name = model_name_or_path if not tokenizer_name else tokenizer_name

    try:
        config = AutoConfig.from_pretrained(config_name if config_name else model_name_or_path)
        # config.num_labels = 1       # binary classification
        config.use_cache = False
    except OSError:                 # Not found
        config = None

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        config=config,
        quantization_config=get_quantization_config(bits),
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        from_tf='.ckpt' in model_name_or_path,
        device_map="auto"
    )
    if bits != -1:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        do_lower_case=do_lower_case
    )
    tokenizer.add_eos_token = True
    tokenizer.padding_side = "left"

    return config, model, tokenizer
