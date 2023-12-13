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

import pandas as pd
import vertexai
from tqdm import tqdm
from vertexai.language_models import TextGenerationModel

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_file",
    type=str,
    default="./data/mini_dataset.json",
    help="Path to the dataset file",
)
parser.add_argument(
    "--out_dir",
    type=str,
    default="./data",
    help="Directory to store the generated dataset",
)
parser.add_argument(
    "--out_file",
    type=str,
    default="gen_dataset.json",
    help="Name of the generated dataset file to use when storing the dataset",
)
parser.add_argument(
    "--gcp_project",
    type=str,
    default=None,
    help="GCP project to use for the TextGenerationModel",
)
parser.add_argument(
    "--gcp_region",
    type=str,
    default=None,
    help="GCP region to use for the TextGenerationModel",
)

prompt_summ = """Given a set of reviews and business attributes for a business on Yelp, generate a summary of the reviews that captures the key aspects of the business. The summary should be concise and easy to read, and it should give potential customers a good idea of what to expect from the business.

The reviews are as follows:
-----
{reviews}
-----

The business attributes are as follows:
-----
{keywords}
-----

Given the reviews and business attributes above, generate a summary that captures the key aspects of the business."""

prompt_keywords = """Given a set of reviews for a business on Yelp, generate a list of keys business attributes ordered by importance/frequency.

The reviews are as follows:
-----
{reviews}
-----

Given the reviews above, generate a list of key business attributes. Only mention business attributes, separated by comma."""


def generate_dataset(model, data):
    kw_params = {
        "candidate_count": 1,
        "max_output_tokens": 16,
        "temperature": 0.3,
        "top_k": 5,
    }

    summ_params = {
        "candidate_count": 1,
        "max_output_tokens": 128,
        "temperature": 0.5,
        "top_k": 40,
    }

    dataset = dict()
    for business_id in tqdm(data, total=len(data)):
        fmt_reviews = "\n\n".join(r for r in data[business_id]["reviews"])
        kw = model.predict(
            prompt_keywords.format(reviews=fmt_reviews), **kw_params
        ).text.strip()
        kw = kw.lower()

        resp = model.predict(
            prompt_summ.format(reviews=fmt_reviews, keywords=kw), **summ_params
        ).text.strip()

        dataset[business_id] = {
            "reviews": data[business_id]["reviews"],
            "rating": data[business_id]["rating"],
            "generated_summary": resp,
            "keywords": kw,
        }

    return dataset


def fmt_dataset(data_file):
    df = pd.read_json(data_file, lines=True)

    grouped = pd.merge(
        df.groupby(["business_id"])["business_stars"].apply(mean).reset_index(),
        df.groupby(["business_id"])["text"].apply(list).reset_index(),
        on="business_id",
        how="inner",
    )

    dataset = dict()
    for _, row in grouped.iterrows():
        dataset[row["business_id"]] = {
            "reviews": row["text"],
            "rating": row["business_stars"],
        }

    return dataset


def main():
    args = parser.parse_args()

    assert os.path.exists(args.data_file), f"{args.data_file} does not exist."
    assert os.path.isfile(args.data_file), f"{args.data_file} is not a file."

    assert os.path.exists(args.out_dir), f"{args.out_dir} does not exist."
    assert os.path.isdir(args.out_dir), f"{args.out_dir} is not a directory."

    vertexai.init(project=args.gcp_project, location=args.gcp_region)
    model = TextGenerationModel.from_pretrained("text-unicorn@001")

    print("Formatting dataset...")
    data = fmt_dataset(args.data_file)

    print("Generating dataset...")
    dataset = generate_dataset(model, data)

    print("Saving dataset...")
    with open(os.path.join(args.out_dir, args.out_file), "w") as f:
        json.dump(dataset, f)


if __name__ == "__main__":
    main()
