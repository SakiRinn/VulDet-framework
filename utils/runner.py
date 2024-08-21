from abc import ABCMeta, abstractmethod
from datetime import datetime
import logging
import numpy as np
import os
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, DistributedSampler
from transformers.modeling_outputs import ModelOutput
from peft import get_peft_model, PeftModel
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

from dataloaders.prompt import TAG_FALSE, TAG_TRUE
import utils


class BaseRunner(metaclass=ABCMeta):

    SUPPORTED_KWARGS = [
        'train_batch_size',
        'eval_batch_size'
    ]

    def __init__(self, model, tokenizer, output_dir, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = output_dir

        for k, v in kwargs.items():
            if k not in self.SUPPORTED_KWARGS:
                raise TypeError(f"`{k}` is an invalid keyword argument for runner.")
            setattr(self, k, v)

        self.setup_device(not torch.cuda.is_available())

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def eval(self):
        pass

    @abstractmethod
    def infer(self):
        pass

    @staticmethod
    def setup_seed(seed):
        random.seed(seed)
        os.environ['PYHTONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    def setup_logger(self, filename='runner.log', log_level=logging.INFO):
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                      datefmt='%m/%d/%Y %H:%M:%S')
        # To stderr
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        # To file
        file_handler = logging.FileHandler(os.path.join(self.output_dir, filename))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        # logger
        logger = logging.getLogger()
        logger.setLevel(log_level if getattr(self, 'local_rank', -1) in [-1, 0]
                        else min(logging.WARN, log_level + 10))
        logger.handlers = [console_handler, file_handler]

    def setup_device(self, no_cuda=False):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        self.device = torch.device("cpu" if no_cuda else "cuda")
        self.n_gpu = 1 if no_cuda else torch.cuda.device_count()

        self.per_device_train_batch_size = self.train_batch_size // max(self.n_gpu, 1)
        self.per_device_eval_batch_size = self.eval_batch_size // max(self.n_gpu, 1)
        try:
            self.model = self.model.to(self.device)
        except ValueError:      # bitsandbytes
            pass


class Runner(BaseRunner):

    SUPPORTED_KWARGS = [
        'train_batch_size',
        'eval_batch_size',
        'num_train_epochs',
        'save_steps',
        'log_steps',
        'max_grad_norm',
    ]

    def __init__(self, model, tokenizer, output_dir='./run_outputs', **kwargs):
        self.local_rank = kwargs.pop('local_rank', -1)
        self.fp16 = kwargs.pop('fp16', False)
        self.cur_epoch, self.cur_step = 0, 0
        super().__init__(model, tokenizer, output_dir, **kwargs)

    def train(self, optimizer, dataset, lr_scheduler=None, eval_dataset=None):

        ##############################
        ###    Before training     ###
        ##############################

        # Prepare dataset
        self.train_batch_size = self.per_device_train_batch_size * max(1, self.n_gpu)
        sampler = RandomSampler(dataset) if self.local_rank == -1 else \
            DistributedSampler(dataset)
        dataloader = DataLoader(dataset,
                                sampler=sampler,
                                batch_size=self.train_batch_size,
                                num_workers=min(os.cpu_count() // 2, 8),
                                pin_memory=True)

        # Set hyperparameters
        self.save_steps = len(dataloader) if self.save_steps <= 0 else self.save_steps
        self.log_steps = 1 if self.log_steps <= 0 else self.log_steps

        # Fp16 (optional)
        if self.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            self.model, optimizer = amp.initialize(
                self.model, optimizer, opt_level=self.fp16_opt_level)

        # Parallel training (optional)
        # NOTE: should be after apex fp16 initialization.
        if self.n_gpu > 1:
            self.model = nn.DataParallel(self.model)
        if self.local_rank != -1:
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True
            )

        ##############################
        ###        Training        ###
        ##############################

        # Print
        logging.info("***** Running training *****")
        logging.info(f"\tNum examples: {len(dataset)}")
        logging.info(f"\tNum epochs: {self.num_train_epochs}")
        logging.info(f"\tInstantaneous batch size per GPU: {self.per_device_train_batch_size}")
        logging.info("\tTotal batch size (w. parallel/distributed training): {}".format(
            self.train_batch_size * (torch.distributed.get_world_size() if self.local_rank != -1 else 1)))

        avg_loss, best_f1 = 0., 0.
        # - Epoch loop
        for epoch in tqdm(range(self.cur_epoch, self.num_train_epochs)):
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
                          if k in utils.get_param_names(self.model.forward)}
                inputs.update({'labels': label.to(self.device)})
                outputs = self.model(**inputs)
                loss, _ = outputs.to_tuple()[:2] \
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
                if lr_scheduler is not None:
                    lr_scheduler.step()

                # Statistic
                total_loss += loss.item()
                avg_loss = total_loss / (step + 1)
                if self.local_rank in [-1, 0]:
                    if self.log_steps > 0 and self.cur_step % self.log_steps == 0:
                        logging.info(f"[epoch {epoch+1}/step {step+1}] cur_loss: {round(loss.item(), 4)}, "
                                     f"avg_loss: {round(avg_loss, 4)}")
                    if self.cur_step % self.save_steps == 0:
                        self.save_weights(f'{self.model.__class__.__name__}_e{epoch+1}s{step}',
                                          optimizer, lr_scheduler)
                self.cur_step += 1

            logging.info(f"[epoch {epoch+1}] Completed. avg_loss: {round(avg_loss, 4)}")
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
                                      optimizer, lr_scheduler)
        return self.model

    def eval(self, dataset, when_training=False):

        ##############################
        ###    Before evaluate     ###
        ##############################

        # Set hyperparameters
        self.eval_batch_size = self.per_device_eval_batch_size * max(1, self.n_gpu)

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
        probs, labels = [], []

        with torch.no_grad():
            for batch in eval_dataloader:
                code, label = batch
                inputs = self.tokenizer(code, return_tensors="pt",
                                        padding=True, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()
                          if k in utils.get_param_names(self.model.forward)}
                inputs.update({'labels': label.to(self.device)})
                outputs = self.model(**inputs)
                loss, logits = outputs.to_tuple()[:2] \
                    if isinstance(outputs, ModelOutput) else outputs[:2]

                eval_step += 1
                total_loss += loss.mean().item()
                prob = F.softmax(logits)

                probs.append(prob.cpu().numpy())
                labels.append(label.cpu().numpy())

        # Aggregate
        probs = np.concatenate(probs, 0)
        labels = np.concatenate(labels, 0)

        results = utils.Metric(probs, labels)()
        results.update({"Avg_loss": (total_loss / eval_step)})
        return results

    def infer(self, code_sample):
        tokens = ' '.join(code_sample.split())
        inputs = self.tokenizer(tokens, return_tensors="pt").to(self.device)
        inputs = {k: v for k, v in inputs.items()
                  if k in utils.get_param_names(self.model.forward)}
        outputs = self.model(**inputs)
        _, logits = outputs.to_tuple()[:2] \
            if isinstance(outputs, ModelOutput) else outputs[:2]
        prob = F.softmax(logits.mean(dim=1))
        return prob

    def setup_device(self, no_cuda=False):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        if self.local_rank == -1:
            self.device = torch.device("cpu" if no_cuda else "cuda")
            self.n_gpu = torch.cuda.device_count()
        else:
            # Initializes the distributed backend which will take care of sychronizing
            # nodes/GPUs
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
            self.n_gpu = 1
            torch.distributed.init_process_group(backend='nccl')

        self.per_device_train_batch_size = self.train_batch_size // max(self.n_gpu, 1)
        self.per_device_eval_batch_size = self.eval_batch_size // max(self.n_gpu, 1)
        self.model = self.model.to(self.device)

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
                logging.warning(
                    f"Fail to load optimizer's weights! They are not in {weight_path}.")
        if scheduler is not None:
            try:
                scheduler.load_state_dict(weights['scheduler'])
                logging.info("Success to load scheduler's weights.")
            except TypeError:
                logging.warning(
                    f"Fail to load scheduler's weights! They are not in {weight_path}.")


class FinetuneRunner(BaseRunner):

    SUPPORTED_KWARGS = [
        'train_batch_size',
        'eval_batch_size',
        'max_seq_length'
    ]

    def __init__(self, model, tokenizer, output_dir='./finetune_outputs', **kwargs):
        self.max_seq_length = kwargs.pop('max_seq_length',
                                         min(tokenizer.model_max_length, 1024))
        train_keys = [k for k in kwargs.keys() if k in utils.get_param_names(SFTConfig)]
        self.train_args = {k: kwargs[k] for k in train_keys}
        kwargs = dict(set(kwargs.items()) - set(self.train_args.items()))
        super().__init__(model, tokenizer, output_dir, **kwargs)

    def train(self, dataset, peft_config=None, eval_dataset=None, resume_from_checkpoint=None):
        self.model.train()
        if self.train_args['gradient_checkpointing']:
            self.model.enable_input_require_grads()

        if isinstance(self.model, PeftModel):
            peft_model = self.model
        else:
            assert peft_config is not None
            peft_model = get_peft_model(self.model, peft_config)
        peft_model.print_trainable_parameters()

        # Keep Trainer from trying its own DataParallelism when more than 1 gpu is available
        if self.n_gpu > 1:
            peft_model.is_parallelizable = True
            peft_model.model_parallel = True

        gradient_accumulation_steps = self.n_gpu * \
            self.train_args.pop('gradient_accumulation_steps', 1)
        train_args = SFTConfig(
            do_train=True,
            output_dir=self.output_dir,
            # run
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            # data
            dataset_text_field="text",
            bf16=True if torch.cuda.is_bf16_supported() else False,
            fp16=False if torch.cuda.is_bf16_supported() else True,
            # report
            report_to="tensorboard",
            run_name=f"finetune_{datetime.now().strftime('%m-%d-%H:%M:%S')}",
            # others
            **self.train_args
        )

        trainer = SFTTrainer(
            model=peft_model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config,
            args=train_args,
            data_collator=DataCollatorForCompletionOnlyLM(
                self.tokenizer.encode('\n### Output:\n', add_special_tokens=False)[1:-1],
                tokenizer=self.tokenizer,
                mlm=False),
            max_seq_length=self.max_seq_length
        )
        with utils.WarningCounter(
            match_text='Could not find response key',
            custom_message='The content after "### Output:" cannot be extracted due to truncation.'
        ):
            trainer.train(resume_from_checkpoint)

        self.model = trainer.model
        return self.model

    def eval(self, eval_dataset):
        model = self.model.merge_and_unload() if isinstance(self.model, PeftModel) \
            else self.model
        model.eval()

        dataloader = DataLoader(
            eval_dataset,
            sampler=SequentialSampler(eval_dataset),
            batch_size=self.eval_batch_size,
            num_workers=min(os.cpu_count() // 2, 8),
            pin_memory=True
        )

        preds, labels = [], []
        num = 0

        with torch.no_grad():
            for batch in tqdm(dataloader):
                len_filter = [i for i, t in enumerate(batch['text'])
                              if len(t) < self.max_seq_length]
                label = np.array([1 if TAG_TRUE in t else 0
                                 for t in batch['output']])[len_filter]

                input_texts = [t for t in batch['text'] if len(t) < self.max_seq_length]
                if not input_texts:
                    continue
                inputs = self.tokenizer(input_texts, return_tensors="pt",
                                        truncation=True, padding=True).to('cuda')
                outputs = model.generate(**inputs, max_new_tokens=16)
                output_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

                pred_texts = [t.split("### Output:\n")[1].strip() for t in output_texts]
                # pred = np.array([1 if TAG_TRUE in t else 0 for t in pred_texts])
                pred = np.array([1 if 'vulnerable' in t.lower() else 0 for t in pred_texts])

                for t, l in zip(pred_texts, label):
                    l_tag = TAG_TRUE if l else TAG_FALSE
                    num += 1
                    logging.info(f'### ({num}) Output: {t} / {l_tag}')

                preds.append(pred)
                labels.append(label)

        # Aggregate
        preds = np.concatenate(preds, 0)
        labels = np.concatenate(labels, 0)

        results = utils.Metric(preds, labels)()
        return results

    def infer(self, prompt):
        ...

    def merge_and_save(self, dirname, peft_dir=None):
        if isinstance(self.model, PeftModel):
            peft_model = self.model
        else:
            assert peft_dir is not None
            peft_model = PeftModel.from_pretrained(
                self.model,
                peft_dir,
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                device_map="auto"
            )
        save_dir = os.path.join(self.output_dir, dirname)

        logging.info("Merging model...")
        peft_model = peft_model.merge_and_unload()
        logging.info("Merge complete, saving the merged model...", )
        peft_model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        logging.info("Merge and save done.")
