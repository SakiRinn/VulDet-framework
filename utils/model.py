import logging

import torch
import bitsandbytes as bnb


def resize_embedding_and_tokenizer(model, tokenizer,
                                   special_tokens_dict: 'dict[str, str]' = {},
                                   custom_tokens: 'list[str]' = []):
    """
    Use mean initialization.
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
    output_embedding = model.get_output_embeddings()
    input_mean = input_embedding.weight.data.mean(dim=0, keepdim=True)
    output_mean = output_embedding.weight.data.mean(dim=0, keepdim=True)

    input_orig_len = input_embedding.weight.data.size(0)
    input_inits = []
    for token in new_tokens:
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        value = input_embedding(torch.tensor(token_ids)).mean(dim=0, keepdim=True)
        input_inits.append(value)
    input_inits = torch.cat(input_inits, dim=0)

    if special_tokens_dict:
        tokenizer.add_special_tokens(special_tokens_dict)
    if custom_tokens:
        tokenizer.add_tokens(custom_tokens, special_tokens=False)
    # make embedding size can be divisible by 128 for optimization.
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=128)

    new_input_embedding = model.get_input_embeddings().weight.data
    new_input_embedding[input_orig_len: input_orig_len + len(new_tokens)] = input_inits
    new_input_embedding[input_orig_len + len(new_tokens):] = input_mean

    new_output_embedding = model.get_output_embeddings().weight.data
    new_output_embedding[-len(new_tokens):] = output_mean

    return True


def find_all_linear_names(model, quantization_bits=-1, add_lm_head=True):
    clazz = bnb.nn.Linear4bit if quantization_bits == 4 \
        else bnb.nn.Linear8bitLt if quantization_bits == 8 \
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
