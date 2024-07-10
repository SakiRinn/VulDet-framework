import torch
from datasets import load_dataset
from transformers import (get_linear_schedule_with_warmup,
                          get_cosine_schedule_with_warmup,
                          get_polynomial_decay_schedule_with_warmup,
                          get_constant_schedule,
                          get_inverse_sqrt_schedule)
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
# from peft import prepare_model_for_kbit_training


SCHEDULER_TYPES = {
    'linear': get_linear_schedule_with_warmup,
    'cosine': get_cosine_schedule_with_warmup,
    'polymonial': get_polynomial_decay_schedule_with_warmup,
    'constant': get_constant_schedule,
    'inverse_sqrt': get_inverse_sqrt_schedule,
}

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def get_quantization_config(int_bits=-1):
    if int_bits == 4:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True
        )
    elif int_bits == 8:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
        # optimizer = "adamw_bnb_8bit"
    else:
        bnb_config = None
    return bnb_config


def load_datasets(dataset_name, validate_split=0.1):
    dataset = load_dataset(dataset_name)
    train_dataset, eval_dataset = dataset.train_test_split(test_size=validate_split).values()
    return train_dataset, eval_dataset


def load_models(model_name: str, config_name='', tokenizer_name='',
                do_lower_case=False, int_bits=-1):
    config_name = model_name if not config_name else config_name
    tokenizer_name = model_name if not tokenizer_name else tokenizer_name

    config = AutoConfig.from_pretrained(config_name if config_name else model_name)
    config.num_labels = 1
    config.use_cache = False

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        quantization_config=get_quantization_config(int_bits),
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        from_tf='.ckpt' in model_name,
        # device_map="auto"
    )
    # if int_bits != -1:
    #     model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        do_lower_case=do_lower_case
    )
    tokenizer.add_eos_token = True
    tokenizer.padding_side = "left"
    # tokenizer.pad_token_id = PAD_TOKEN_ID
    # tokenizer.pad_token = tokenizer.convert_ids_to_tokens(PAD_TOKEN_ID)
    tokenizer.pad_token_id = 0

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    return config, model, tokenizer
