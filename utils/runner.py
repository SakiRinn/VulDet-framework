from datetime import datetime
import logging
import os
import random
import numpy as np
from tqdm import tqdm
import inspect

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F

from transformers.modeling_outputs import ModelOutput

import utils


def get_param_names(func):
    signature = inspect.signature(func)
    params = signature.parameters
    return [p for p in params if p not in ['args', 'kwargs']]


class Runner:

    def __init__(self, model, tokenizer,
                 train_args={}, eval_args={}, output_dir='./outputs'):
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = output_dir

        self.cur_epoch, self.cur_step = 0, 0
        self.local_rank = -1

        for k, v in train_args.items():
            setattr(self, k, v)
        for k, v in eval_args.items():
            setattr(self, 'eval_' + k, v)

        date = datetime.now().strftime("%m-%d-%H:%M:%S")
        self.output_dir = os.path.join(self.output_dir, date)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.set_device(False)
        self.set_logger()

    def train(self, optimizer, dataset, scheduler=None, eval_dataset=None):

        ##############################
        ###    Before training     ###
        ##############################

        # Prepare dataset
        self.batch_size = self.per_gpu_batch_size * max(1, self.n_gpu)
        sampler = RandomSampler(dataset) if self.local_rank == -1 else DistributedSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=self.batch_size,
                                num_workers=4, pin_memory=True)

        # Set hyperparameters
        self.save_per_steps = len(dataloader) if self.save_per_steps <= 0 else self.save_per_steps
        self.log_per_steps = 1 if self.log_per_steps <= 0 else self.log_per_steps

        # Fp16 (optional)
        if self.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            self.model, optimizer = amp.initialize(self.model, optimizer, opt_level=self.fp16_opt_level)

        # Parallel training (optional)
        # NOTE: should be after apex fp16 initialization.
        if self.n_gpu > 1:
            self.model = nn.DataParallel(self.model)
        if self.local_rank != -1:
            self.model = nn.parallel.DistributedDataParallel(self.model,
                                                             device_ids=[self.local_rank],
                                                             output_device=self.local_rank,
                                                             find_unused_parameters=True)

        ##############################
        ###        Training        ###
        ##############################

        # Print
        logging.info("***** Running training *****")
        logging.info(f"\tNum examples: {len(dataset)}")
        logging.info(f"\tNum epochs: {self.epochs}")
        logging.info(f"\tInstantaneous batch size per GPU: {self.per_gpu_batch_size}")
        logging.info("\tTotal batch size (w. parallel/distributed training): {}".format(
            self.batch_size * (torch.distributed.get_world_size() if self.local_rank != -1 else 1)))

        avg_loss, best_f1 = 0., 0.
        # - Epoch loop
        for epoch in tqdm(range(self.cur_epoch, self.epochs)):
            # For resume, retrain for an entire epoch.
            if self.cur_step % len(dataloader) != 0:
                self.cur_step -= self.cur_step % len(dataloader)

            # Prepare model
            self.model.train()
            self.model.zero_grad()

            total_loss = 0.

            # - Step loop
            for step, batch in enumerate(dataloader):
                # Predict
                code, label = batch
                inputs = self.tokenizer(code, return_tensors='pt',
                                        truncation=True, padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()
                          if k in get_param_names(self.model.forward)}
                inputs.update({'labels': label.to(self.device)})
                outputs = self.model(**inputs)
                loss, logits = outputs.to_tuple()[:2] \
                    if isinstance(outputs, ModelOutput) else outputs[:2]

                # Parallel training (optional)
                if self.n_gpu > 1:
                    loss = loss.mean()

                # Backward
                if self.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.max_grad_norm)
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                # Update
                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()

                # Statistic
                total_loss += loss.item()
                avg_loss = total_loss / (step + 1)
                if self.local_rank in [-1, 0]:
                    if self.log_per_steps > 0 and self.cur_step % self.log_per_steps == 0:
                        logging.info(f"[epoch {epoch + 1}/step {step + 1}] cur_loss: {round(loss.item(), 4)}, "
                                     f"avg_loss: {round(avg_loss, 4)}")
                    if self.cur_step % self.save_per_steps == 0:
                        self.save_weights(f'{self.model.__class__.__name__}_e{epoch + 1}s{step}',
                                          optimizer, scheduler)
                self.cur_step += 1

            logging.info(f"[epoch {epoch + 1}] Completed. avg_loss: {round(avg_loss, 4)}")
            self.cur_epoch += 1

            # Evaluate
            # NOTE: Only evaluate when single GPU, otherwise metrics may not average well.
            if self.local_rank == -1 and eval_dataset is not None:
                results = self.eval(eval_dataset, when_training=True)
                for key, value in results.items():
                    logging.info(f"{key} = {round(value, 4)}")
                if results['F1-score'] > best_f1:
                    best_f1 = results['F1-score']
                    logging.info(f"Best f1-score: {round(best_f1, 4)}!")
                    self.save_weights(f'{self.model.__class__.__name__}_best-f1',
                                      optimizer, scheduler)
        return self.model

    def eval(self, dataset, when_training=False):

        ##############################
        ###    Before evaluate     ###
        ##############################

        # Set hyperparameters
        self.eval_batch_size = self.per_gpu_eval_batch_size * max(1, self.n_gpu)

        # Prepare model
        self.model.eval()

        # Prepare dataset
        # NOTE: DistributedSampler samples randomly
        eval_sampler = SequentialSampler(dataset) if self.local_rank == -1 else \
            DistributedSampler(dataset)
        eval_dataloader = DataLoader(
            dataset,
            sampler=eval_sampler,
            batch_size=self.eval_batch_size,
            num_workers=4,
            pin_memory=True)

        # Parallel evaluation
        if not when_training and self.n_gpu > 1:
            self.model = nn.DataParallel(self.model)

        ##############################
        ###        Evaluate        ###
        ##############################

        # Print
        logging.info("***** Running evaluation *****")
        logging.info(f"  Num examples = {len(dataset)}", )
        logging.info(f"  Batch size = {self.eval_batch_size}")

        eval_step, total_loss = 0, 0.
        scores, labels = [], []

        with torch.no_grad():
            for batch in eval_dataloader:
                code, label = batch
                inputs = self.tokenizer(code, return_tensors="pt",
                                        padding=True, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()
                          if k in get_param_names(self.model.forward)}
                inputs.update({'labels': label.to(self.device)})
                outputs = self.model(**inputs)
                loss, logits = outputs.to_tuple()[:2] \
                    if isinstance(outputs, ModelOutput) else outputs[:2]

                eval_step += 1
                total_loss += loss.mean().item()
                prob = F.softmax(logits.mean(dim=1))

                scores.append(prob.cpu().numpy())
                labels.append(label.cpu().numpy())

        # Aggregate
        scores = np.concatenate(scores, 0)
        labels = np.concatenate(labels, 0)

        results = utils.Metric(scores, labels)()
        results.update({"Avg_loss": (total_loss / eval_step)})
        return results

    def inference(self, sample):
        code = ' '.join(sample.split())
        inputs = self.tokenizer(code, return_tensors="pt").to(self.device)
        inputs = {k: v for k, v in inputs.items()
                  if k in get_param_names(self.model.forward)}
        outputs = self.model(**inputs)
        loss, logits = outputs.to_tuple()[:2] \
            if isinstance(outputs, ModelOutput) else outputs[:2]
        prob = F.softmax(logits.mean(dim=1))
        return prob

    @staticmethod
    def set_seed(seed):
        random.seed(seed)
        os.environ['PYHTONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    def set_logger(self, filename='runner.log'):
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                      datefmt='%m/%d/%Y %H:%M:%S')
        # To stderr
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        # To file
        file_handler = logging.FileHandler(os.path.join(self.output_dir, filename))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        # logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO if self.local_rank in [-1, 0] else logging.WARN)
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    def set_device(self, no_cuda=False):
        if self.local_rank == -1:
            self.device = torch.device("cpu" if no_cuda else "cuda")
            self.n_gpu = torch.cuda.device_count()
        else:
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
            self.n_gpu = 1
            torch.distributed.init_process_group(backend='nccl')

        self.per_gpu_batch_size = self.batch_size // max(self.n_gpu, 1)
        self.per_gpu_eval_batch_size = self.eval_batch_size // max(self.n_gpu, 1)
        try:
            self.model = self.model.to(self.device)
        except ValueError:
            pass

    def save_weights(self, filename, optimizer=None, scheduler=None):
        weights = {
            'model': (self.model.module if hasattr(self.model, 'module') else self.model).state_dict(),
            'cur_epoch': self.cur_epoch,
            'cur_step': self.cur_step
        }
        if optimizer is not None:
            weights['optimizer'] = optimizer.state_dict()
        if scheduler is not None:
            weights['scheduler'] = scheduler.state_dict()
        save_dir = os.path.join(self.output_dir, f'{filename}.pt')
        torch.save(weights, save_dir)
        logging.info(f"Saving the dict of weights to {save_dir}.")

    def load_weights(self, weight_path, optimizer=None, scheduler=None):
        weights = torch.load(weight_path)
        try:
            self.model.load_state_dict(weights['model'])
            self.cur_epoch = weights['cur_epoch']
            self.cur_step = weights['cur_step']
            logging.info("Success to load model's weights; "
                         f"current epoch: {self.cur_epoch}, current step: {self.cur_step}.")
        except TypeError:
            logging.warning(f"Fail to load model's weights! They are not in {weight_path}.")
        if optimizer is not None:
            try:
                optimizer.load_state_dict(weights['optimizer'])
                logging.info("Success to load optimizer's weights.")
            except TypeError:
                logging.warning(f"Fail to load optimizer's weights! They are not in {weight_path}.")
        if scheduler is not None:
            try:
                scheduler.load_state_dict(weights['scheduler'])
                logging.info("Success to load scheduler's weights.")
            except TypeError:
                logging.warning(f"Fail to load scheduler's weights! They are not in {weight_path}.")
