from datasets import load_dataset
from peft.tuners.lora import LoraConfig
from finetune.codellama import to_prompt, setup_peft_model, setup_peft_trainer
from tools.huggingface import load_models
from tools.runner import Runner


def main():
    Runner.set_seed(42)
    _, model, tokenizer = load_models('codellama/CodeLlama-7b-hf', int_bits=8)
    model.train()
    train_dataset = load_dataset('json', data_files='data/train.json', split='train').map(to_prompt)
    eval_dataset = load_dataset('json', data_files='data/eval.json', split='train').map(to_prompt)
    trainer = setup_peft_trainer(model, tokenizer, train_dataset, eval_dataset)
    trainer.train()


if __name__ == "__main__":
    main()