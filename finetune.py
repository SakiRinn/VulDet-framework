import logging
import math
import os.path as osp

from datasets import load_dataset
from torch.utils.data import DataLoader
from finetune.codellama import lora_merge, resize_tokenizer_and_embedding, to_prompt, setup_model, setup_trainer
from utils.huggingface import load_models
from utils.runner import Runner


DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def main():
    Runner.set_seed(42)
    do_train = False
    do_eval = True
    last_checkpoint = None
    model_name_or_path = '/root/autodl-tmp/co'
    custom_tokens = []

    _, model, tokenizer = load_models(model_name_or_path, int_bits=8)
    train_dataset = load_dataset('json', data_files='data/train.json', split='train').map(to_prompt)
    eval_dataset = load_dataset('json', data_files='data/eval.json', split='train').map(to_prompt)

    custom_tokens = [token.strip() for token in custom_tokens]
    special_tokens_dict = {}
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
    if special_tokens_dict or custom_tokens:
        resize_tokenizer_and_embedding(model, tokenizer, special_tokens_dict, custom_tokens)

    model = setup_model(model)
    trainer = setup_trainer(model, tokenizer, train_dataset, eval_dataset)

    if do_train:
        logging.info("*** Train ***")

        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif model_name_or_path is not None and osp.isdir(model_name_or_path):
            checkpoint = model_name_or_path
        else:
            checkpoint = None
        train_result = trainer.train()
        metrics = train_result.metrics

        trainer.save_model()
        trainer.save_state()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

    if do_eval:
        logging.info("*** Evaluate ***")

        metrics = trainer.evaluate()
        perplexity = math.exp(metrics["eval_loss"])
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


def eval(model_name_or_path):
    _, model, tokenizer = load_models(model_name_or_path, int_bits=8)
    eval_dataset = load_dataset('json', data_files='data/eval.json', split='train').map(to_prompt)
    dataloader = DataLoader(eval_dataset, batch_size=32)

    custom_tokens = []
    special_tokens_dict = {}
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
    if special_tokens_dict or custom_tokens:
        resize_tokenizer_and_embedding(model, tokenizer, special_tokens_dict, custom_tokens)

    model = setup_model(model)

    eval_step, total_loss = 0, 0.
    logits, labels = [], []
    # with torch.no_grad():
    #     for batch in dataloader:
    #         input_ids = batch['text'].to('cuda')
    #         outputs = model(input_ids=input_ids)
    #         preds = torch.argmax(outputs.logits, dim=-1)

    #         eval_step += 1
    #         total_loss += loss.mean().item()

    #         logits.append(logit.cpu().numpy())
    #         labels.append(label.cpu().numpy())

    # # Aggregate
    # logits = np.concatenate(logits, 0)
    # labels = np.concatenate(labels, 0)

    # results = tools.Metric(logits, labels)()
    # results.update({"Avg_loss": (total_loss / eval_step)})


if __name__ == "__main__":
    # main()
    eval('/root/autodl-tmp/co')
