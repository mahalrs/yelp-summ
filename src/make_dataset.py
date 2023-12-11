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

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_file",
    type=str,
    default="./data/processed_dataset.json",
    help="Path to the dataset file",
)
parser.add_argument(
    "--out_dir",
    type=str,
    default="./data",
    help="Directory to store the processed dataset",
)
parser.add_argument(
    "--out_file",
    type=str,
    default="mini_dataset.json",
    help="Name of the processed dataset file to use when storing the dataset",
)
parser.add_argument(
    "--num_businesses",
    type=int,
    default=100,
    help="Number of businesses to include in the dataset",
)
parser.add_argument(
    "--max_reviews_per_business",
    type=int,
    default=5,
    help="Maximum number of reviews to include per business",
)


def make_dataset(data_file, num_businesses, max_reviews_per_business):
    df = pd.read_json(data_file, lines=True)

    sampled_businesses = pd.merge(
        df["business_id"].sample(num_businesses, random_state=42),
        df,
        on="business_id",
        how="inner",
    )

    sampled_reviews = sampled_businesses.groupby(["business_id"]).sample(
        max_reviews_per_business, random_state=42
    )

    return sampled_reviews


def main():
    args = parser.parse_args()

    assert os.path.exists(args.data_file), f"{args.data_file} does not exist."
    assert os.path.isfile(args.data_file), f"{args.data_file} is not a file."

    assert os.path.exists(args.out_dir), f"{args.out_dir} does not exist."
    assert os.path.isdir(args.out_dir), f"{args.out_dir} is not a directory."

    dataset = make_dataset(
        args.data_file, args.num_businesses, args.max_reviews_per_business
    )

    print("Saving dataset...")
    print(f"No. of businesses: {dataset['business_id'].nunique()}")
    print(f"No. of reviews: {dataset.shape[0]}")

    dataset.to_json(
        os.path.join(args.out_dir, args.out_file), orient="records", lines=True
    )


if __name__ == "__main__":
    main()
