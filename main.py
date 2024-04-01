import os
import numpy as np
import torch
from tqdm import tqdm

from transformers import (AdamW, get_linear_schedule_with_warmup)
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from datasets import TextDataset
from models import ReGVD, Devign
from tools.utils import *


def train(args, dataset, model):

    ##############################
    ###    Before training     ###
    ##############################

    # Set hyperparameters
    args.max_steps = args.epochs * len(dataloader) if args.max_steps <= 0 else args.max_steps
    args.save_per_steps = len(dataloader) if args.save_per_steps <= 0 else args.save_per_steps
    args.warmup_steps = len(dataloader) if args.save_per_steps < 0 else args.save_per_steps       # can be 0
    args.logging_steps = len(dataloader) if args.save_per_steps <= 0 else args.save_per_steps

    # Prepare model
    model.to(args.device)
    model.train()
    model.zero_grad()

    # Prepare dataset
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    sampler = RandomSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.train_batch_size,
                            num_workers=4, pin_memory=True)

    # Prepare optimizer & scheduler (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)

    # Load optimizer & scheduler
    checkpoint_latest = os.path.join(args.output_dir, 'checkpoint-latest')
    scheduler_latest = os.path.join(checkpoint_latest, 'scheduler.pt')
    optimizer_latest = os.path.join(checkpoint_latest, 'optimizer.pt')
    if os.path.exists(scheduler_latest):
        scheduler.load_state_dict(torch.load(scheduler_latest))
    if os.path.exists(optimizer_latest):
        optimizer.load_state_dict(torch.load(optimizer_latest))

    # Fp16 (optional)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # Parallel training (optional)
    # NOTE: should be after apex fp16 initialization.
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)


    ##############################
    ###        Training        ###
    ##############################

    # Print
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num epochs = {args.epochs}")
    logger.info(f"  Num steps = {args.max_steps}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Instantaneous batch size per GPU = {args.per_gpu_train_batch_size}")
    logger.info("  Total batch size (w. parallel/distributed training) = {}".format(
        args.train_batch_size * (torch.distributed.get_world_size() if args.local_rank != -1 else 1)))

    # Statistic
    total_step = args.start_step
    cur_step, avg_loss, cur_loss = 0, 0., 0.
    best_acc = 0.0
    # model.resize_token_embeddings(len(tokenizer))

    # Epoch loop
    for epoch in tqdm(range(args.start_epoch, int(args.epochs))):
        cur_loss = 0

        # Step loop
        for step, batch in tqdm(enumerate(dataloader, start=1)):

            # Predict
            inputs = batch[0].to(args.device)
            labels = batch[1].to(args.device)
            loss, logits = model(inputs, labels)

            # Parallel training (optional)
            if args.n_gpu > 1:
                loss = loss.mean()

            # Backward
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            # Update
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            # Statistic
            total_step += 1
            cur_loss += loss.item()
            avg_loss = cur_loss / step
            if args.local_rank in [-1, 0] and args.logging_steps > 0 and total_step % args.logging_steps == 0:
                logger.info(f"[epoch {epoch}/step {step}] {avg_loss}")

            if args.local_rank in [-1, 0]:
                # Evaluate
                # NOTE: Only evaluate when single GPU, otherwise metrics may not average well
                if args.local_rank == -1 and args.evaluate_during_training:
                    results = evaluate(args, model, when_training=True)
                    for key, value in results.items():
                        logger.info(f"  {key} = {round(value, 4)}")

                # Save
                if total_step % args.save_per_steps == 0 or results['eval_acc'] > best_acc:
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-acc')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model
                    save_path = os.path.join(output_dir, f'{model.__name__}_e{epoch}s{step}.pt')
                    torch.save(model_to_save.state_dict(), save_path)

                    best_acc = results['eval_acc']
                    logger.info("  " + "*"*20)
                    logger.info(f"  Best acc: {round(best_acc, 4)}")
                    logger.info("  " + "*"*20)
                    logger.info(f"Saving model checkpoint to {save_path}")

        avg_loss = cur_loss / cur_step
        logger.info(f"[epoch {epoch}] loss {avg_loss}")


def evaluate(args, dataset, model, when_training=False):

    ##############################
    ###    Before evaluate     ###
    ##############################

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    # Set hyperparameters
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Prepare model
    model.to(args.device)
    model.eval()

    # Prepare dataset
    # NOTE: DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
    eval_dataloader = DataLoader(
        dataset,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size,
        num_workers=4,
        pin_memory=True)

    # Parallel evaluate
    if not when_training and args.n_gpu > 1:
        model = torch.nn.DataParallel(model)


    ##############################
    ###        Evaluate        ###
    ##############################

    # Print
    logger.info("***** Running evaluation *****")
    logger.info(f"  Num examples = {len(dataset)}", )
    logger.info(f"  Batch size = {args.eval_batch_size}")

    total_step, total_loss = 0, 0.
    logits, labels = [], []
    with torch.no_grad():
        for batch in eval_dataloader:
            inputs = batch[0].to(args.device)
            label = batch[1].to(args.device)
            loss, logit = model(inputs, label)

            total_step += 1
            total_loss += loss.mean().item()

            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())

    # Aggregate
    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)
    preds = logits[:, 0] > 0.5

    # Result
    avg_loss = total_loss / total_step
    eval_acc = np.mean(labels == preds)
    result = {
        "eval_loss": avg_loss.item().cpu(),
        "eval_acc": eval_acc
    }
    return result


def main():
    args = get_args()
    logger.info("Training/evaluation parameters: \n", args)

    set_seed(args.seed)
    set_logging(args)
    set_device(args)

    args.start_epoch, args.start_step = 0, 0
    config, base_model, tokenizer = load_model_and_tokenizer(args, *MODEL_CLASSES[args.model_type])

    # Our input block size will be the max possible for the model
    args.block_size = tokenizer.max_len_single_sentence if args.block_size <= 0 else \
        min(args.block_size, tokenizer.max_len_single_sentence)

    if args.model == "devign":
        model = Devign(base_model, config, tokenizer, args)
    else:
        model = ReGVD(base_model, config, tokenizer, args)
    # Make sure only the first process in distributed training download model & vocab
    if args.local_rank == 0:
        torch.distributed.barrier()

    # Train
    if args.do_train:
        # Make sure only the first process in distributed training process the
        # dataset, and the others will use the cache.
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()
        train_dataset = TextDataset(tokenizer, args, args.train_data_file, args.training_percent)
        if args.local_rank == 0:
            torch.distributed.barrier()

        train(args, train_dataset, model)

    # Evaluation
    if args.local_rank in [-1, 0]:
        model.load_state_dict(torch.load(args.checkpoint_path))
        model.to(args.device)

        if args.do_veal:
            eval_dataset = TextDataset(tokenizer, args, args.eval_data_file)
            eval_result = evaluate(args, eval_dataset, model)
            logger.info("***** Eval results *****")
            for key in sorted(eval_result.keys()):
                logger.info(f"  {key} = {round(eval_result[key], 4)}")


if __name__ == "__main__":
    main()
