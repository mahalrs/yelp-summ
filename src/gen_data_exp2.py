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

import vertexai
from tqdm import tqdm
from vertexai.language_models import TextGenerationModel

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_file",
    type=str,
    default="./data/eval_dataset.json",
    help="Path to the dataset file",
)
parser.add_argument(
    "--out_dir",
    type=str,
    default="./data",
    help="Directory to store the generated dataset",
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

prompt_sentiment = """Given a set of reviews for a business on Yelp, generate a summary of the reviews that captures the {sentiment} aspects of the business. The summary should be concise and easy to read, and it should give potential customers a good idea of what to expect from the business.

The reviews are as follows:
-----
{reviews}
-----

Given the reviews above, generate a summary that captures the  most important {sentiment} things that were mentioned in the reviews. It should not include any {sentiment_exclude} things that were mentioned."""

prompt_aggregate = """Given a positive and a negative summary of reviews of a Yelp business, summarize both the negative and positive reviews summary to generate a business summary capturing both sentiments. The business summary should be neutral and objective.

The reviews are as follows:
-----
Positive reviews summary: {pos_summary}

Negative reviews summary: {neg_summary}
-----

Given the review summaries above, generate a business summary that captures the positive and negative things about the business."""


def generate_dataset(model, model_params, data):
    dataset = dict()
    for business_id in tqdm(data, total=len(data)):
        fmt_reviews = "\n\n".join(r for r in data[business_id]["reviews"])

        pos_summ = model.predict(
            prompt_sentiment.format(
                sentiment="positive",
                sentiment_exclude="negative",
                reviews=fmt_reviews,
            ),
            **model_params,
        ).text.strip()

        neg_summ = model.predict(
            prompt_sentiment.format(
                sentiment="negative",
                sentiment_exclude="positive",
                reviews=fmt_reviews,
            ),
            **model_params,
        ).text.strip()

        agg_summ = model.predict(
            prompt_aggregate.format(
                pos_summary=pos_summ,
                neg_summary=neg_summ,
            ),
            **model_params,
        ).text.strip()

        dataset[business_id] = {
            "reviews": data[business_id]["reviews"],
            "summaries": data[business_id]["summaries"],
            "generated_summary": agg_summ,
        }

    return dataset


def main():
    args = parser.parse_args()

    assert os.path.exists(args.data_file), f"{args.data_file} does not exist."
    assert os.path.isfile(args.data_file), f"{args.data_file} is not a file."

    assert os.path.exists(args.out_dir), f"{args.out_dir} does not exist."
    assert os.path.isdir(args.out_dir), f"{args.out_dir} is not a directory."

    vertexai.init(project=args.gcp_project, location=args.gcp_region)
    parameters = {
        "candidate_count": 1,
        "max_output_tokens": 128,
        "temperature": 0.5,
        "top_k": 40,
    }
    model = TextGenerationModel.from_pretrained("text-unicorn@001")

    print("Generating dataset...")
    with open(args.data_file, "r") as f:
        data = json.load(f)

    dataset = generate_dataset(model, parameters, data)

    print("Saving dataset...")
    with open(os.path.join(args.out_dir, "gen_dataset_exp2.json"), "w") as f:
        json.dump(dataset, f)


if __name__ == "__main__":
    main()
