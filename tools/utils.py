from transformers import (BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)

from argparse import Namespace
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from datasets import TextDataset
from models import ReGVD, Devign

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
}


class Args(Namespace):
    def __init__(self):
        # Path
        self.train_data_file = "../dataset/train.jsonl"
        self.eval_data_file = "../dataset/valid.jsonl"
        self.test_data_file = "../dataset/test.jsonl"
        self.output_dir = "./saved_models"
        self.checkpoint_path = ""

        # Name or path
        self.model_type = "roberta"
        self.model_name_or_path = "microsoft/codebert-base"
        self.tokenizer_name = "microsoft/codebert-base"
        self.config_name = ""
        self.cache_dir = ""

        # Process
        self.do_train = True
        self.do_eval = False
        self.evaluate_during_training = True

        # Model parameters
        self.gnn = "ResGatedGNN"
        self.block_size = 400
        self.hidden_size = 256
        self.feature_dim_size = 768
        self.num_classes = 2
        self.num_GNN_layers = 2
        self.format = "uni"

        # Tokenizer parameters
        self.do_lower_case = False

        # Dataset parameters
        self.train_batch_size = 128
        self.eval_batch_size = 128
        self.training_percent = 1.0

        # Optimizer parameters
        self.learning_rate = 5e-4
        self.max_grad_norm = 1.0
        self.weight_decay = 0.0
        self.adam_epsilon = 1e-8

        # Train
        self.epochs = 30
        self.max_steps = -1
        self.save_per_steps = -1
        self.warmup_steps = -1
        self.logging_steps = -1
        self.save_total_limit = None

        # Device
        self.no_cuda = True
        self.fp16 = False
        self.fp16_opt_level = 'O1'
        self.local_rank = -1

        # ReGVD
        self.model = "GNNs"
        self.window_size = 5
        self.remove_residual = False
        self.att_op = 'mul'
        self.alpha_weight = 1.0

        # Other
        self.seed = 123456


def get_args():
    return Args()


def set_seed(seed):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def set_logging(args):
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    return logging.getLogger(__name__)


def set_device(args):
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1

    args.device = device
    args.per_gpu_train_batch_size = args.train_batch_size // max(args.n_gpu, 1)
    args.per_gpu_eval_batch_size = args.eval_batch_size // max(args.n_gpu, 1)

    logger.warning(f"Process rank: {args.local_rank}, device: {device}, n_gpu: {args.n_gpu}, "
                   f"distributed training: {bool(args.local_rank != -1)}, 16-bits training: {args.fp16}")


def load_model_and_tokenizer(args, config_class, model_class, tokenizer_class):
    if args.local_rank not in [-1, 0]:
        # Barrier to make sure only the first process in distributed training download model & vocab
        torch.distributed.barrier()

    checkpoint_latest = os.path.join(args.output_dir, 'checkpoint-latest')
    if os.path.exists(checkpoint_latest) and os.listdir(checkpoint_latest):
        args.model_name_or_path = os.path.join(checkpoint_latest, 'pytorch_model.bin')
        args.config_name = os.path.join(checkpoint_latest, 'config.json')
        idx_file = os.path.join(checkpoint_latest, 'idx_file.txt')
        with open(idx_file, encoding='utf-8') as idxf:
            args.start_epoch = int(idxf.readlines()[0].strip()) + 1

        step_file = os.path.join(checkpoint_latest, 'step_file.txt')
        if os.path.exists(step_file):
            with open(step_file, encoding='utf-8') as stepf:
                args.start_step = int(stepf.readlines()[0].strip())

        logger.info("reload model from {}, resume from {} epoch".format(checkpoint_latest, args.start_epoch))

    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.num_labels = 1
    model = model_class.from_pretrained(args.model_name_or_path,
                                        from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=config,
                                        cache_dir=args.cache_dir if args.cache_dir else None)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                cache_dir=args.cache_dir if args.cache_dir else None,
                                                do_lower_case=args.do_lower_case)

    return config, model, tokenizer
