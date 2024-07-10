import argparse
from functools import partial
import inspect
import logging
import math

import torch
from transformers.tokenization_utils import PreTrainedTokenizer
import yaml

import dataloaders
import models
from tools import Runner
from tools.huggingface import SCHEDULER_TYPES, load_models


def get_classes(module):
    members = inspect.getmembers(module)
    classes = {member[0]: member[1] for member in members if inspect.isclass(member[1])}
    return classes


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=['train', 'eval', 'infer'], type=str)
    parser.add_argument("config", type=str, help="Path to the YAML file used to set hyperparameters.")
    parser.add_argument("--output-dir", default="./outputs", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--weight-path", type=str,
                        help="The weight file (.pt) saved during training. if train, it's used for resumption.")
    parser.add_argument("--local-rank", type=int, default=-1,
                        help="For distributed training: local_rank.")
    parser.add_argument("--no-cuda", action='store_true',
                        help="Avoid using CUDA when available.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit.")
    parser.add_argument("--no-eval-when-training", action='store_true',
                        help="Only takes effect when training. Don't run evaluation when training.")
    return parser.parse_args()


def main():
    args = get_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    args.dataset, args.model, args.optimizer = \
        config['class']['dataset'], config['class']['model'], config['class']['optimizer']
    dataset_args = config['dataset']
    model_args = config['model']
    optimizer_args = config['optimizer']
    train_args = config['train']
    eval_args = config['eval']
    args.seed = config['seed']
    del config

    # Get classes
    try:
        model_class = get_classes(models)[args.model]
    except KeyError:
        raise KeyError(f"Unsupported model class: {args.model}!")
    try:
        dataset_class = get_classes(dataloaders)[args.dataset]
    except KeyError:
        raise KeyError(f"Unsupported dataset class: {args.dataset}!")
    try:
        optimizer_class = get_classes(torch.optim)[args.optimizer]
    except KeyError:
        raise KeyError(f"Unsupported optimizer class: {args.optimizer}!")

    # - Model
    if 'transformer' in model_args.keys():
        config, transformer, tokenizer = load_models(**model_args['transformer'])
        del model_args['transformer']
        model = model_class(transformer, config, tokenizer, **model_args)
    else:
        model = model_class(config, tokenizer, **model_args)
        tokenizer = dataset_class.get_tokenizer()
    # Make sure only the first process in distributed training download model & vocab
    if args.local_rank == 0:
        torch.distributed.barrier()

    # - Runner
    train_args.update({'local_rank': args.local_rank, 'fp16': args.fp16})
    runner = Runner(model, tokenizer, train_args, eval_args, args.output_dir)
    runner.set_seed(args.seed)
    runner.set_device(args.no_cuda)

    logging.info('***** Basic information *****')
    logging.info(f"\tConfig: {args.config}")
    logging.info(f"\tModel: {args.model}, dataset: {args.dataset}, optimizer: {args.optimizer}")
    logging.info(f"\tTask: {args.task}, seed: {args.seed}")
    logging.info(f"\tDistributed training: {args.local_rank != -1}, fp16 training: {args.fp16}, "
                 f"device: {runner.device}, num GPUs: {runner.n_gpu}")

    # - Dataset
    logging.info("Start Loading dataset...")
    train_dataset = dataset_class(is_train=True, **dataset_args)
    eval_dataset = dataset_class(is_train=False, **dataset_args)
    logging.info("Loading completed.")

    # - Optimizer & scheduler
    if 'scheduler' in optimizer_args.keys():
        scheduler_args = optimizer_args['scheduler']
        scheduler_func = SCHEDULER_TYPES[scheduler_args['type']]
        del optimizer_args['scheduler'], scheduler_args['type']

        steps_per_epoch = math.ceil(len(train_dataset) / train_args['batch_size'])
        scheduler_args['num_training_steps'] = train_args['epochs'] * steps_per_epoch
        if scheduler_args['num_warmup_steps'] < 0:      # can be 0
            scheduler_args['num_warmup_steps'] = steps_per_epoch
    else:
        scheduler_func = None

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': optimizer_args['weight_decay']},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    del optimizer_args['weight_decay']
    optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_args)
    scheduler = scheduler_func(optimizer, **scheduler_args) if scheduler_func is not None else None

    # - Train
    if args.task == 'train':
        if args.weight_path:
            runner.load_weights(args.weight_path, optimizer, scheduler)
            logging.info(f"Resumed from {args.weight_path}.")
        logging.info("Start training...")
        if args.no_eval_when_training:
            runner.train(optimizer, train_dataset, scheduler)
        else:
            runner.train(optimizer, train_dataset, scheduler, eval_dataset)
        logging.info("Training completed.")

    # - Evaluate
    if args.task == 'eval' and args.local_rank in [-1, 0]:
        if not args.weight_path:
            raise ValueError("When evaluating, `--weight_path` must be specified.")
        runner.load_weights(args.weight_path)

        logging.info("Start evaluation...")
        eval_result = runner.eval(eval_dataset)
        logging.info("Evaluation completed.")

        logging.info("***** Evaluation results *****")
        for key in sorted(eval_result.keys()):
            logging.info(f"  {key} = {round(eval_result[key], 4)}")


if __name__ == "__main__":
    main()
