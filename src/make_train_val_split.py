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
import random

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_file",
    type=str,
    default="./data/gen_dataset.json",
    help="Path to the generated eval dataset file",
)
parser.add_argument(
    "--out_dir",
    type=str,
    default="./data",
    help="Directory to store the processed dataset",
)
parser.add_argument(
    "--out_train_file",
    type=str,
    default="gen_dataset_train.json",
    help="Name of the train dataset file to use when storing the dataset",
)
parser.add_argument(
    "--out_val_file",
    type=str,
    default="gen_dataset_val.json",
    help="Name of the val dataset file to use when storing the dataset",
)


def make_split(data_file):
    with open(data_file, "r") as f:
        dataset = json.load(f)

    keys = list(dataset.keys())
    keys.sort()

    random.shuffle(keys)

    train_split = dict()
    for k in keys[: int(0.9 * len(keys))]:
        train_split[k] = dataset[k]

    val_split = dict()
    for k in keys[int(0.9 * len(keys)) :]:
        val_split[k] = dataset[k]

    return train_split, val_split


def main():
    args = parser.parse_args()
    random.seed(42)

    assert os.path.exists(args.data_file), f"{args.data_file} does not exist."
    assert os.path.isfile(args.data_file), f"{args.data_file} is not a file."

    assert os.path.exists(args.out_dir), f"{args.out_dir} does not exist."
    assert os.path.isdir(args.out_dir), f"{args.out_dir} is not a directory."

    train_split, val_split = make_split(args.data_file)

    print(f"Train split size: {len(train_split)}")
    print(f"Val split size  : {len(val_split)}")

    with open(os.path.join(args.out_dir, args.out_train_file), "w") as f:
        json.dump(train_split, f)

    with open(os.path.join(args.out_dir, args.out_val_file), "w") as f:
        json.dump(val_split, f)


if __name__ == "__main__":
    main()
