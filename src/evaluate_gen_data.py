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
import json
import os
from statistics import mean

import evaluate

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_file",
    type=str,
    default="./data/gen_dataset_exp1.json",
    help="Path to the dataset file",
)


def eval(data, metric_names):
    metrics = [evaluate.load(metric) for metric in metric_names]

    for business_id in data:
        gen_summ = data[business_id]["generated_summary"]
        for gold_summ in data[business_id]["summaries"]:
            for metric in metrics:
                metric.add(predictions=gen_summ, references=gold_summ)

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

    with open(args.data_file, "r") as f:
        data = json.load(f)

    metrics = ["rouge", "bertscore", "bleu", "meteor"]
    scores = eval(data, metrics)

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
