import argparse
from datetime import datetime
import logging
import os
import os.path as osp
import math

import torch
import yaml

import dataloaders
from dataloaders import TextDataset
import models
import utils
from utils import Runner
from utils.huggingface import SCHEDULER_TYPES, load_transformers


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=['train', 'eval', 'infer'], type=str)
    parser.add_argument("config", type=str,
                        help="Path to the YAML file used to set hyperparameters.")
    parser.add_argument("--output-dir", default="", type=str,
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

    # Get classes
    try:
        model_class = utils.get_classes(models)[args.model]
    except KeyError:
        raise KeyError(f"Unsupported model class: {args.model}!")
    try:
        dataset_class = utils.get_classes(dataloaders)[args.dataset]
    except KeyError:
        raise KeyError(f"Unsupported dataset class: {args.dataset}!")
    try:
        optimizer_class = utils.get_classes(torch.optim)[args.optimizer]
    except KeyError:
        raise KeyError(f"Unsupported optimizer class: {args.optimizer}!")

    # - Model
    if 'transformer' in model_args.keys():
        config, transformer, tokenizer = load_transformers(**model_args.pop('transformer'))
        model = model_class(transformer, config, tokenizer, **model_args)
    else:
        model = model_class(config, tokenizer, **model_args)
        tokenizer = dataset_class.get_tokenizer()
    # Make sure only the first process in distributed training download model & vocab
    if args.local_rank == 0:
        torch.distributed.barrier()

    # - Runner
    runner_args.update({'local_rank': args.local_rank, 'fp16': args.fp16})
    runner = Runner(model, tokenizer, output_dir, **runner_args)
    runner.setup_seed(args.seed)

    logging.info('***** Basic information *****')
    logging.info(f"\tConfig: {args.config}")
    logging.info(f"\tModel: {args.model}, dataset: {args.dataset}, optimizer: {args.optimizer}")
    logging.info(f"\tTask: {args.task}, seed: {args.seed}")
    logging.info(f"\tDistributed training: {args.local_rank != -1}, fp16 training: {args.fp16}, "
                 f"device: {runner.device}, num GPUs: {runner.n_gpu}")

    # - Dataset
    logging.info("Start Loading dataset...")
    file_path = dataset_args.pop('file_path')
    if isinstance(file_path, dict):
        # Format 1
        train_dataset = TextDataset(file_path['train'], **dataset_args)
        test_dataset = TextDataset(file_path['test'], **dataset_args)
    else:
        # Format 2
        test_split = dataset_args.pop('test_split')
        train_dataset, test_dataset = TextDataset(file_path, **dataset_args) \
            .train_test_split(test_split).values()
    logging.info("Loading completed.")

    # - Optimizer & scheduler
    if 'lr_scheduler' in optimizer_args.keys():
        lr_scheduler_args = optimizer_args.pop('lr_scheduler')
        lr_scheduler_type = SCHEDULER_TYPES[lr_scheduler_args.pop('type')]

        steps_per_epoch = math.ceil(len(train_dataset) / runner_args['train_batch_size'])
        lr_scheduler_args['num_training_steps'] = \
            runner_args['num_train_epochs'] * steps_per_epoch
        if lr_scheduler_args.get('num_warmup_steps', -1) < 0:       # can be 0
            lr_scheduler_args['num_warmup_steps'] = steps_per_epoch
    else:
        lr_scheduler_type = None

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': optimizer_args.pop('weight_decay')},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_args)
    lr_scheduler = lr_scheduler_type(optimizer, **lr_scheduler_args) \
        if lr_scheduler_type is not None else None

    # - Train
    if args.task == 'train':
        if args.weight_path:
            runner.load_weights(args.weight_path, optimizer, lr_scheduler)
            logging.info(f"Resumed from {args.weight_path}.")
        logging.info("Start training...")
        if args.no_eval_when_training:
            runner.train(optimizer, train_dataset, lr_scheduler)
        else:
            runner.train(optimizer, train_dataset, lr_scheduler, test_dataset)
        logging.info("Training completed.")

    # - Eval
    if args.task == 'eval' and args.local_rank in [-1, 0]:
        if not args.weight_path:
            raise ValueError("When evaluating, `--weight-path` must be specified.")
        runner.load_weights(args.weight_path)

        logging.info("Start evaluation...")
        eval_result = runner.eval(test_dataset)
        logging.info("Evaluation completed.")

        logging.info("***** Evaluation results *****")
        for key in sorted(eval_result.keys()):
            logging.info(f"  {key} = {round(eval_result[key], 4)}")


if __name__ == "__main__":
    main()
