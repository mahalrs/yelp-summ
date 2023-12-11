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

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_root",
    type=str,
    default="./data/gold_summs",
    help="Directory containing the dataset",
)
parser.add_argument(
    "--out_dir",
    type=str,
    default="./data",
    help="Directory to store the processed dataset",
)


def make_eval_dataset(train_path, val_path, test_path):
    train = pd.read_csv(train_path, sep="\t")
    val = pd.read_csv(val_path, sep="\t")
    test = pd.read_csv(test_path, sep="\t")

    eval_df = pd.concat([train, val, test])
    eval_df.drop(columns=["cat"], inplace=True)
    eval_df.rename(columns={"group_id": "business_id"}, inplace=True)

    eval_dataset = dict()
    for _, row in eval_df.iterrows():
        eval_dataset[row["business_id"]] = {
            "reviews": [
                row["rev1"],
                row["rev2"],
                row["rev3"],
                row["rev4"],
                row["rev5"],
                row["rev6"],
                row["rev7"],
                row["rev8"],
            ],
            "summaries": [row["summ1"], row["summ2"], row["summ3"]],
        }

    return eval_dataset


def main():
    args = parser.parse_args()

    assert os.path.exists(args.data_root), f"{args.data_root} does not exist"
    assert os.path.isdir(args.data_root), f"{args.data_root} is not a directory"

    train_path = os.path.join(args.data_root, "train.csv")
    assert os.path.exists(train_path), f"{train_path} does not exist"
    assert os.path.isfile(train_path), f"{train_path} is not a file"

    val_path = os.path.join(args.data_root, "val.csv")
    assert os.path.exists(val_path), f"{val_path} does not exist"
    assert os.path.isfile(val_path), f"{val_path} is not a file"

    test_path = os.path.join(args.data_root, "test.csv")
    assert os.path.exists(test_path), f"{test_path} does not exist"
    assert os.path.isfile(test_path), f"{test_path} is not a file"

    print("Processing eval dataset...")
    eval_dataset = make_eval_dataset(train_path, val_path, test_path)

    print("Saving eval dataset...")
    eval_dataset_path = os.path.join(args.out_dir, "eval_dataset.json")

    with open(eval_dataset_path, "w") as f:
        json.dump(eval_dataset, f)


if __name__ == "__main__":
    main()
