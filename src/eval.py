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
import os
from statistics import mean

import evaluate
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from dataset import YelpDataset

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_file",
    type=str,
    default="./data/gen_dataset_exp3.json",
    help="Path to the dataset file",
)
parser.add_argument(
    "--model_path",
    type=str,
    default="./logs/checkpoint-600",
    help="Path to the model",
)
parser.add_argument(
    "--use_attr",
    type=bool,
    default=False,
    help="Whether to use business attributes",
)
parser.add_argument(
    "--use_rating",
    type=bool,
    default=False,
    help="Whether to use business ratings",
)
parser.add_argument(
    "--use_rating_as_text",
    type=bool,
    default=False,
    help="Whether to use business ratings as text",
)


def eval(model, tokenizer, test_loader, device, metric_names):
    metrics = [evaluate.load(metric) for metric in metric_names]

    for batch in tqdm(test_loader):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        summ1 = batch["summ1"]
        summ2 = batch["summ2"]
        summ3 = batch["summ3"]

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        summaries = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=3,
            max_length=128,
            early_stopping=True,
            length_penalty=0.6,
        )

        decoded_summaries = tokenizer.batch_decode(
            summaries,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
            return_tensors="pt",
        )

        decoded_summ1 = tokenizer.batch_decode(
            summ1,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
            return_tensors="pt",
        )

        decoded_summ2 = tokenizer.batch_decode(
            summ2,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
            return_tensors="pt",
        )

        decoded_summ3 = tokenizer.batch_decode(
            summ3,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
            return_tensors="pt",
        )

        for metric in metrics:
            metric.add_batch(
                predictions=decoded_summaries, references=decoded_summ1
            )
            metric.add_batch(
                predictions=decoded_summaries, references=decoded_summ2
            )
            metric.add_batch(
                predictions=decoded_summaries, references=decoded_summ3
            )

    return [
        metric.compute()
        if metric.name != "bert_score"
        else metric.compute(lang="en")
        for metric in metrics
    ]


def main():
    args = parser.parse_args()

    assert os.path.exists(args.data_file), f"{args.data_file} does not exist."
    assert os.path.isfile(args.data_file), f"{args.data_file} is not a file."

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")

    test_dataset = YelpDataset(
        args.data_file,
        tokenizer,
        label_gold=True,
        use_attr=args.use_attr,
        use_rating=args.use_rating,
        rating_as_text=args.use_rating_as_text,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    metrics = ["rouge", "bertscore", "bleu", "meteor"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    scores = eval(model, tokenizer, test_loader, device, metrics)

    rouge1 = round(scores[0]["rouge1"], 4)
    rouge2 = round(scores[0]["rouge2"], 4)
    rougeL = round(scores[0]["rougeL"], 4)
    rougeLsum = round(scores[0]["rougeLsum"], 4)

    precision = round(mean(scores[1]["precision"]), 4)
    recall = round(mean(scores[1]["recall"]), 4)
    f1 = round(mean(scores[1]["f1"]), 4)

    bleu = round(scores[2]["bleu"], 4)
    meteor = round(scores[3]["meteor"], 4)

    print("\n\n----------")
    print(f"Evaluation results for {args.data_file}:")
    print("----------")
    print(
        f"  ROUGE    : rouge1: {rouge1} rouge2: {rouge2} rougeL: {rougeL} rougeLsum: {rougeLsum}"
    )
    print(f"  BERTScore: precision: {precision} recall: {recall} f1: {f1}")
    print(f"  BLEU     : {bleu}")
    print(f"  METEOR   : {meteor}")
    print("----------\n\n")


if __name__ == "__main__":
    main()
