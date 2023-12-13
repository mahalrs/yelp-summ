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
import logging
import math

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_scheduler

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
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
        "--model_name",
        type=str,
        default="facebook/bart-base",
        help="Model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of dataloader workers to use",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the warmup period if any) to use.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0,
        help="Weight decay to use.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=2,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help="Number of steps to accumulate before performing an update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        type=bool,
        default=True,
        help="Whether to use gradient checkpointing to save memory.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "fp8"],
        help="Whether to use mixed precision.",
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="A random seed for reproducible training.",
    )

    return parser.parse_args()


def get_dataloaders(training_args):
    tokenizer = AutoTokenizer.from_pretrained(training_args.model_name)

    train_dataset = YelpDataset(training_args.train_data_file, tokenizer)
    val_dataset = YelpDataset(training_args.val_data_file, tokenizer)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_args.per_device_train_batch_size,
        num_workers=training_args.num_workers,
        shuffle=True,
        pin_memory=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        num_workers=training_args.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    return train_dataloader, val_dataloader


def main():
    logger = get_logger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    args = parse_args()
    set_seed(args.seed)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )

    logger.info(accelerator.state, main_process_only=False)
    accelerator.wait_for_everyone()

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    train_dataloader, val_dataloader = get_dataloaders(args)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    per_device_training_batches = math.ceil(
        len(train_dataloader) / accelerator.num_processes
    )
    per_device_training_steps_per_epoch = math.ceil(
        per_device_training_batches / args.gradient_accumulation_steps
    )
    per_device_training_steps = (
        per_device_training_steps_per_epoch * args.num_train_epochs
    )
    total_training_steps = per_device_training_steps * accelerator.num_processes

    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=total_training_steps,
    )

    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    )

    num_training_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    num_training_steps = num_training_steps_per_epoch * args.num_train_epochs

    total_batch_size = (
        accelerator.num_processes
        * args.per_device_train_batch_size
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"Num Epochs: {args.num_train_epochs}")
    logger.info(f"Total training steps: {total_training_steps}")
    logger.info(f"Per device training steps: {num_training_steps}")
    logger.info(f"Warmup steps: {args.num_warmup_steps}")
    logger.info(f"Total batch size: {total_batch_size}")

    for epoch in range(args.num_train_epochs):
        pbar = tqdm(
            range(num_training_steps_per_epoch),
            disable=not accelerator.is_local_main_process,
        )
        pbar.set_postfix(
            epoch=f"{epoch + 1}/{args.num_train_epochs}", refresh=True
        )

        model.train()
        running_loss = 0.0
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                output = model(**batch)
                loss = output.loss
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                running_loss += loss.detach().float()

                pbar.set_postfix(
                    epoch=f"{epoch + 1}/{args.num_train_epochs}",
                    train_loss=loss.item(),
                    refresh=True,
                )
            if accelerator.sync_gradients:
                pbar.update(1)

        train_loss = running_loss / len(train_dataloader)
        pbar.set_postfix(
            epoch=f"{epoch + 1}/{args.num_train_epochs}",
            train_loss=train_loss.item(),
            refresh=True,
        )

        pbar.close()
        accelerator.wait_for_everyone()

        pbar = tqdm(
            len(val_dataloader),
            disable=not accelerator.is_local_main_process,
        )

        model.eval()
        running_loss = 0.0
        for batch in val_dataloader:
            with torch.no_grad():
                output = model(**batch)
                loss = output.loss
                running_loss += loss.detach().float()

                pbar.set_postfix(
                    val_loss=loss.item(),
                    refresh=True,
                )

        val_loss = running_loss / len(val_dataloader)
        pbar.set_postfix(
            val_loss=val_loss.item(),
            refresh=True,
        )

        pbar.close()
        accelerator.wait_for_everyone()

    accelerator.wait_for_everyone()

    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        args.output_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
    )


if __name__ == "__main__":
    main()
