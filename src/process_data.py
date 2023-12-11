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
    "--data_root",
    type=str,
    default="../data",
    help="Directory containing the dataset",
)
parser.add_argument(
    "--min_review_count",
    type=int,
    default=15,
    help="Minimum review count for a business to be included in the dataset",
)
parser.add_argument(
    "--min_stars",
    type=float,
    default=3.0,
    help="Minimum star rating for a business to be included in the dataset",
)
parser.add_argument(
    "--min_review_length",
    type=int,
    default=42,
    help="Minimum number of words in a review to be included in the dataset",
)
parser.add_argument(
    "--max_review_length",
    type=int,
    default=149,
    help="Maximum number of words in a review to be included in the dataset",
)


def process_businesses(businesses_path, min_review_count, min_stars):
    df = pd.read_json(businesses_path, lines=True)
    df = df[["business_id", "stars", "review_count", "categories"]]
    df.dropna(inplace=True)

    filtered_df = df[
        (df["review_count"] >= min_review_count) & (df["stars"] >= min_stars)
    ]

    return filtered_df


def process_reviews(reviews_path, min_stars, min_review_len, max_review_len):
    df = pd.read_json(reviews_path, lines=True)
    df = df[["review_id", "business_id", "stars", "text"]]
    df.dropna(inplace=True)

    filtered_df = df[
        (df["stars"] >= min_stars)
        & (
            df["text"].apply(
                lambda x: (
                    len(x.split()) >= min_review_len
                    and len(x.split()) <= max_review_len
                )
            )
        )
    ]

    return filtered_df


def make_dataset(businesses_df, reviews_df, min_review_count, out_path):
    merged_df = pd.merge(
        reviews_df, businesses_df, on="business_id", how="inner"
    )
    merged_df.drop(columns=["review_count", "categories"], inplace=True)
    merged_df.rename(
        columns={"stars_x": "review_stars", "stars_y": "business_stars"},
        inplace=True,
    )

    filtered_df = merged_df.groupby(["business_id"]).filter(
        lambda x: len(x) >= min_review_count
    )

    print("Writing dataset to disk...")
    print(f"No. of businesses: {filtered_df['business_id'].nunique()}")
    print(f"No. of reviews: {filtered_df.shape[0]}")

    filtered_df.to_json(out_path, orient="records", lines=True)


def main():
    args = parser.parse_args()

    assert os.path.exists(args.data_root), "Data root does not exist"
    assert os.path.isdir(args.data_root), f"{args.data_root} is not a directory"

    businesses_path = os.path.join(
        args.data_root, "yelp_academic_dataset_business.json"
    )
    assert os.path.exists(businesses_path), f"{businesses_path} does not exist"
    assert os.path.isfile(businesses_path), f"{businesses_path} is not a file"

    reviews_path = os.path.join(
        args.data_root, "yelp_academic_dataset_review.json"
    )
    assert os.path.exists(reviews_path), f"{reviews_path} does not exist"
    assert os.path.isfile(reviews_path), f"{reviews_path} is not a file"

    print("Processing businesses...")
    businesses_df = process_businesses(
        businesses_path, args.min_review_count, args.min_stars
    )

    print("Processing reviews...")
    reviews_df = process_reviews(
        reviews_path,
        args.min_stars,
        args.min_review_length,
        args.max_review_length,
    )

    print("Making dataset...")
    dataset_path = os.path.join(args.data_root, "processed_dataset.json")
    make_dataset(businesses_df, reviews_df, args.min_review_count, dataset_path)


if __name__ == "__main__":
    main()
