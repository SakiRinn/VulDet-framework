import argparse
from datetime import datetime
import logging
import os
import os.path as osp
import yaml

import torch
import peft
from peft import PeftConfig, PeftModel
from datasets import Dataset

import utils
from utils import FinetuneRunner, WarningCounter
from utils import find_all_linear_names, resize_embedding_and_tokenizer
from utils.huggingface import DEFAULT_TOKENS, PEFT_TASK_TYPES, load_transformers
from dataloaders import eval_prompt, train_prompt


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=['train', 'eval', 'inference'], type=str)
    parser.add_argument("config", type=str,
                        help="Path to the YAML file used to set hyperparameters.")
    parser.add_argument("--checkpoint-dir", default=None, type=str,
                        help="The weight file (.pt) saved during training. if train, it's used for resumption.")
    parser.add_argument("--output-dir", default="", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--no-cuda", action='store_true',
                        help="Avoid using CUDA when available.")
    parser.add_argument("--no-eval-when-training", action='store_true',
                        help="Only takes effect when training. Don't run evaluation when training.")
    parser.add_argument("--log-level", default="INFO", type=str,
                        help="Only takes effect when training. Don't run evaluation when training.")
    return parser.parse_args()


def main():
    args = get_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    dataset_args = config['dataset']
    transformer_args = config['transformer']
    peft_args = config['peft']
    runner_args = config['runner']
    args.seed = config['seed']
    del config

    # Default output directory
    if args.output_dir == "":
        date = datetime.now().strftime("%m-%d-%H:%M:%S")
        output_dir = osp.join('./outputs', date)
        if not osp.exists(output_dir):
            os.makedirs(output_dir)
    else:
        output_dir = args.output_dir

    # - Model
    custom_tokens = [t.strip() for t in transformer_args.pop('custom_tokens', [])]
    _, base_model, tokenizer = load_transformers(**transformer_args)
    is_resized = resize_embedding_and_tokenizer(
        base_model, tokenizer, DEFAULT_TOKENS, custom_tokens)
    model = PeftModel.from_pretrained(
        base_model,
        args.checkpoint_dir,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto"
    ) if args.checkpoint_dir else base_model

    # - Runner
    runner = FinetuneRunner(model, tokenizer, output_dir, **runner_args)
    runner.setup_logger(log_level=getattr(logging, args.log_level.strip().upper()))
    runner.setup_device(args.no_cuda)
    runner.setup_seed(args.seed)

    logging.info('***** Basic information *****')
    logging.info(f"\tConfig: {args.config}")
    logging.info(f"\tTask: {args.task}, seed: {args.seed}")

    # - Dataset
    logging.info("Start Loading dataset...")
    file_path = dataset_args.pop('file_path')
    if isinstance(file_path, dict):
        # Format 1
        train_dataset = Dataset.from_json(file_path['train']).map(train_prompt)
        eval_dataset = Dataset.from_json(file_path['eval']).map(eval_prompt)
    else:
        # Format 2
        test_split = dataset_args.pop('test_split')
        dataset = Dataset.from_json(file_path['eval'])
        train_dataset, eval_dataset = dataset.train_test_split(test_split, shuffle=False)
    logging.info("Loading completed.")

    # - Peft
    peft_class = peft_args.pop('type', PeftConfig).strip().capitalize() + 'Config'
    peft_class = utils.get_classes(peft.tuners)[peft_class]
    peft_task = peft_args.pop('task', PeftConfig).strip().upper()
    if peft_task not in PEFT_TASK_TYPES:
        raise ValueError(f"Peft task `{peft_task}`is unsupported!\n"
                         f"Supported peft tasks: {PEFT_TASK_TYPES}")

    target_modules = peft_args.pop('target_modules', None)
    if peft_args.pop('all_linear', False):
        # Override
        target_modules = find_all_linear_names(runner.model)
        if is_resized:
            # Removing lm_head from target modules, will use in modules_to_save
            target_modules.pop(target_modules.index("lm_head"))

    modules_to_save = peft_args.pop('modules_to_save', [])
    if peft_args.pop('long_lora', False):
        # Override
        modules_to_save = ["embed_tokens", "input_layernorm",
                           "post_attention_layernorm", "norm"]
        if is_resized:
            modules_to_save += ["lm_head"]
    elif is_resized:
        modules_to_save = list(set(modules_to_save + ["embed_tokens", "lm_head"]))

    peft_config = peft_class(
        task_type=peft_task,
        target_modules=target_modules,
        modules_to_save=modules_to_save,
        **peft_args
    )

    # - Train
    if args.task == 'train':
        logging.info("Start training...")
        with WarningCounter(
            match_text='Could not find response key',
            custom_message='The content after "### Output:" cannot be extracted due to truncation.'
        ):
            if args.no_eval_when_training:
                runner.train(train_dataset, peft_config,
                             resume_from_checkpoint=args.checkpoint_dir)
            else:
                runner.train(train_dataset, peft_config, eval_dataset,
                             resume_from_checkpoint=args.checkpoint_dir)
        logging.info("Training completed.")
        eval_result = runner.eval(eval_dataset)
        if not args.no_eval_when_training:
            logging.info("Start evaluation...")
            eval_result = runner.eval(eval_dataset)
            logging.info("***** Evaluation results *****")
            for key in sorted(eval_result.keys()):
                logging.info(f"  {key} = {round(eval_result[key], 4)}")

    # - Eval
    if args.task == 'eval':
        if not args.checkpoint_dir:
            raise ValueError("When evaluating, `--checkpoint-dir` must be specified.")
        logging.info("Start evaluation...")
        eval_result = runner.eval(eval_dataset)
        logging.info("***** Evaluation results *****")
        for key in sorted(eval_result.keys()):
            logging.info(f"  {key} = {round(eval_result[key], 4)}")


if __name__ == "__main__":
    main()
