# Copyright 2023 The YelpSumm Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

import wandb
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

from dataset import YelpDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_data_file",
        type=str,
        default="./data/gen_dataset_train.json",
        help="Path to the train dataset file",
    )
    parser.add_argument(
        "--val_data_file",
        type=str,
        default="./data/gen_dataset_val.json",
        help="Path to the validation dataset file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./logs",
        help="Directory to save logs and checkpoints",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="A random seed for reproducible training.",
    )

    return parser.parse_args()


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print(f"Train loss: {result.metrics['train_loss']:.2f}")


def main():
    args = parse_args()
    set_seed(args.seed)

    wandb.login()
    wandb.init(project="yelp-summ")

    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")

    train_dataset = YelpDataset(args.train_data_file, tokenizer, use_attr=True)
    val_dataset = YelpDataset(args.val_data_file, tokenizer, use_attr=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="steps",
        logging_strategy="steps",
        save_strategy="steps",
        eval_steps=50,
        logging_steps=50,
        save_steps=50,
        save_total_limit=2,
        num_train_epochs=3,
        learning_rate=5e-5,
        weight_decay=0,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        fp16=True,
        dataloader_num_workers=4,
        load_best_model_at_end=True,
        report_to="all",
        push_to_hub=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    print("Start training...")
    print("Experiment 2: Controlled - attributes")
    result = trainer.train()

    print_summary(result)
    print(f"Best model checkpoint: {trainer.state.best_model_checkpoint}")


if __name__ == "__main__":
    main()
