import os
import os.path as osp
import logging
import random
import inspect
import json
from datetime import datetime
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers.modeling_outputs import ModelOutput
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer, SFTConfig

import utils
from utils.huggingface import DEFAULT_TOKENS, load_transformers
from utils.llm import find_all_linear_names, resize_embedding_and_tokenizer


def get_param_names(func):
    signature = inspect.signature(func)
    params = signature.parameters
    return [p for p in params if p not in ['args', 'kwargs']]


class Runner:

    SUPPORTED_KWARGS = [
        'train_batch_size',
        'eval_batch_size',
        'epochs',
        'save_per_steps',
        'log_per_steps',
        'max_grad_norm',
    ]

    def __init__(self, model, tokenizer, output_dir='./outputs', **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = output_dir

        date = datetime.now().strftime("%m-%d-%H:%M:%S")
        self.output_dir = os.path.join(self.output_dir, date)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        for k, v in kwargs.items():
            if k not in self.SUPPORTED_KWARGS:
                raise TypeError(f"`{k}` is an invalid keyword argument for runner.")
            setattr(self, k, v)

        self.cur_epoch, self.cur_step = 0, 0
        self.local_rank = -1

        self.setup_device(False)
        self.setup_logger()

    def train(self, optimizer, dataset, lr_scheduler=None, eval_dataset=None):

        ##############################
        ###    Before training     ###
        ##############################

        # Prepare dataset
        self.train_batch_size = self.train_batch_size_per_gpu * max(1, self.n_gpu)
        sampler = RandomSampler(dataset) if self.local_rank == -1 else DistributedSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=self.train_batch_size,
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
        logging.info(f"\tInstantaneous batch size per GPU: {self.train_batch_size_per_gpu}")
        logging.info("\tTotal batch size (w. parallel/distributed training): {}".format(
            self.train_batch_size * (torch.distributed.get_world_size() if self.local_rank != -1 else 1)))

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
                if lr_scheduler is not None:
                    lr_scheduler.step()

                # Statistic
                total_loss += loss.item()
                avg_loss = total_loss / (step + 1)
                if self.local_rank in [-1, 0]:
                    if self.log_per_steps > 0 and self.cur_step % self.log_per_steps == 0:
                        logging.info(f"[epoch {epoch + 1}/step {step + 1}] cur_loss: {round(loss.item(), 4)}, "
                                     f"avg_loss: {round(avg_loss, 4)}")
                    if self.cur_step % self.save_per_steps == 0:
                        self.save_weights(f'{self.model.__class__.__name__}_e{epoch + 1}s{step}',
                                          optimizer, lr_scheduler)
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
                                      optimizer, lr_scheduler)
        return self.model

    def eval(self, dataset, when_training=False):

        ##############################
        ###    Before evaluate     ###
        ##############################

        # Set hyperparameters
        self.eval_batch_size = self.eval_batch_size_per_gpu * max(1, self.n_gpu)

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
                          if k in get_param_names(self.model.forward)}
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

    def inference(self, code_sample):
        tokens = ' '.join(code_sample.split())
        inputs = self.tokenizer(tokens, return_tensors="pt").to(self.device)
        inputs = {k: v for k, v in inputs.items()
                  if k in get_param_names(self.model.forward)}
        outputs = self.model(**inputs)
        _, logits = outputs.to_tuple()[:2] \
            if isinstance(outputs, ModelOutput) else outputs[:2]
        prob = F.softmax(logits.mean(dim=1))
        return prob

    @staticmethod
    def setup_seed(seed):
        random.seed(seed)
        os.environ['PYHTONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    def setup_logger(self, filename='runner.log'):
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

    def setup_device(self, no_cuda=False):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        if self.local_rank == -1:
            self.device = torch.device("cpu" if no_cuda else "cuda")
            self.n_gpu = torch.cuda.device_count()
        else:
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
            self.n_gpu = 1
            torch.distributed.init_process_group(backend='nccl')

        self.train_batch_size_per_gpu = self.train_batch_size // max(self.n_gpu, 1)
        self.eval_batch_size_per_gpu = self.eval_batch_size // max(self.n_gpu, 1)
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
                logging.warning(f"Fail to load optimizer's weights! They are not in {weight_path}.")
        if scheduler is not None:
            try:
                scheduler.load_state_dict(weights['scheduler'])
                logging.info("Success to load scheduler's weights.")
            except TypeError:
                logging.warning(f"Fail to load scheduler's weights! They are not in {weight_path}.")


class FinetuneRunner(Runner):

    SUPPORTED_KWARGS = [
        'train_batch_size',
        'eval_batch_size',
        'epochs',
        'save_per_steps',
        'log_per_steps',
        'max_grad_norm',
    ]

    def __init__(self, model, tokenizer, output_dir='./outputs', **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = output_dir

        date = datetime.now().strftime("%m-%d-%H:%M:%S")
        self.output_dir = os.path.join(self.output_dir, date)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        for k, v in kwargs.items():
            if k not in self.SUPPORTED_KWARGS:
                raise TypeError(f"`{k}` is an invalid keyword argument for runner.")
            setattr(self, k, v)

        self.custom_tokens = [token.strip() for token in self.custom_tokens]
        self.is_resized = resize_embedding_and_tokenizer(
            self.model, self.tokenizer, DEFAULT_TOKENS, self.custom_tokens)

        self.setup_device(False)
        self.setup_logger()

    def train(self, dataset, eval_dataset=None):
        self.model.train()
        self.model.enable_input_require_grads()

        target_modules = None
        if self.all_linear:
            target_modules = find_all_linear_names(self.model)
            if self.is_resized:
                # Removing lm_head from target modules, will use in modules_to_save
                target_modules.pop(target_modules.index("lm_head"))

        if self.long_lora:
            modules_to_save = ["embed_tokens", "input_layernorm",
                               "post_attention_layernorm", "norm"]
            if self.is_resized:
                modules_to_save += ["lm_head"]
        elif self.is_resized:
            modules_to_save = ["embed_tokens", "lm_head"]

        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
            ],
            modules_to_save=modules_to_save
        )

        train_args = SFTConfig(
            output_dir=self.output_dir,
            do_train=True,
            # train
            per_device_train_batch_size=self.train_batch_size_per_gpu,
            gradient_accumulation_steps=self.n_gpu,
            num_train_epochs=5,
            warmup_steps=100,
            # max_steps=400,                  # override `num_train_epochs`
            # optimize
            optim='adamw_bnb_8bit',
            learning_rate=3e-4,
            lr_scheduler_type='linear',
            weight_decay=0.,
            # eval
            # eval_strategy="steps",
            per_device_eval_batch_size=2 * self.train_batch_size_per_gpu,
            eval_steps=20,
            # load_best_model_at_end=True,
            # log & save
            logging_strategy="steps",
            logging_steps=10,
            save_strategy="steps",
            save_steps=20,
            save_total_limit=5,
            # dataset
            dataset_text_field="text",
            dataloader_drop_last=True,
            group_by_length=True,
            # dtype
            bf16=True if torch.cuda.is_bf16_supported() else False,
            fp16=False if torch.cuda.is_bf16_supported() else True,
            # other
            gradient_checkpointing=True,
            # report
            report_to="tensorboard",        # wandb
            run_name=f"lora-{datetime.now().strftime('%m-%d-%H-%M-%S')}"
        )

        peft_model = get_peft_model(self.model, peft_config)
        # peft_model = torch.compile(peft_model)
        peft_model.print_trainable_parameters()

        trainer = SFTTrainer(
            model=peft_model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config,
            args=train_args,
            max_seq_length=2048
        )
        trainer.train(resume_from_checkpoint=None)

    def eval(self, eval_dataset):
        max_length = 2048
        self.model.eval()

        dataloader = DataLoader(
            eval_dataset,
            sampler=SequentialSampler(eval_dataset),
            batch_size=6,
            num_workers=4,
            pin_memory=True)

        preds, labels = [], []
        with torch.no_grad():
            for batch in tqdm(dataloader):
                len_filter = [i for i, t in enumerate(batch['text']) if len(t) < max_length]
                label = np.array([1 if 'VULNERABLE' in t else 0
                                  for t in batch['output']])[len_filter]

                input_texts = [t for t in batch['text'] if len(t) < max_length]
                inputs = self.tokenizer(input_texts, return_tensors="pt",
                                        truncation=True, padding=True).to('cuda')
                outputs = self.model.generate(**inputs, max_new_tokens=16)
                output_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

                pred_texts = [t.split("### Output:\n")[1].strip() for t in output_texts]
                pred = np.array([1 if 'VULNERABLE' in t else 0 for t in pred_texts])

                preds.append(pred)
                labels.append(label)

        # Aggregate
        preds = np.concatenate(preds, 0)
        labels = np.concatenate(labels, 0)

        results = utils.Metric(preds, labels)()
        return results

    def inference(self, code_sample):
        ...

    def lora_merge(self, lora_dir):
        if base_model:
            logging.info(f"Using base model {base_model}")
        else:
            adapter_config_path = osp.join(lora_dir, "adapter_config.json")
            with open(adapter_config_path) as f:
                adapter_config = json.load(f)
            base_model = adapter_config["base_model_name_or_path"]
            logging.info(f"Base model not given, using {base_model}")

        _, base_model, tokenizer = load_transformers(base_model, bits=8)

        model = PeftModel.from_pretrained(
            base_model,
            lora_dir,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map="auto"
        )

        logging.info("Merging model...")
        model = model.merge_and_unload()

        logging.info("Merge complete, saving the merged model...", )
        model.save_pretrained('model-merged')
        tokenizer.save_pretrained('model-merged')

    def save_weights(self):
        return NotImplementedError

    def load_weights(self):
        return NotImplementedError
