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

import json

from torch.utils.data import Dataset


class YelpDataset(Dataset):
    def __init__(
        self,
        data_file,
        tokenizer,
        sep_token="</s>",
        use_attr=False,
        use_rating=False,
        rating_as_text=False,
        rating_cutoff=4.0,
        label_gold=False,
    ):
        self.tokenizer = tokenizer
        self.sep_token = sep_token
        self.use_attr = use_attr
        self.use_rating = use_rating
        self.rating_as_text = rating_as_text
        self.rating_cutoff = rating_cutoff
        self.label_gold = label_gold

        with open(data_file, "r") as f:
            self.data = list(json.load(f).values())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prefix = []

        if self.use_attr:
            prefix.append(f"business attributes: {self.data[idx]['keywords']}")

        if self.use_rating:
            if "rating" not in self.data[idx]:
                rating = 4.0
            else:
                rating = self.data[idx]["rating"]

            if self.rating_as_text:
                if rating >= self.rating_cutoff:
                    prefix.append("rating: business has high ratings.")
                else:
                    prefix.append("rating: business has low ratings.")
            else:
                prefix.append(f"rating: {rating}/5")

        inp = f",{self.sep_token}".join(r for r in self.data[idx]["reviews"])
        if prefix:
            inp = f"{' '.join(p for p in prefix)}{self.sep_token}{inp}"

        if self.label_gold:
            label = self.data[idx]["summaries"]
        else:
            label = self.data[idx]["generated_summary"]

        inputs = self.tokenizer(
            inp,
            max_length=1024,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        labels = self.tokenizer(
            label,
            max_length=128,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        if self.label_gold:
            return {
                "input_ids": inputs["input_ids"].view(-1),
                "attention_mask": inputs["attention_mask"].view(-1),
                "labels": labels["input_ids"][0],
                "summ1": labels["input_ids"][0],
                "summ2": labels["input_ids"][1],
                "summ3": labels["input_ids"][2],
            }
        else:
            return {
                "input_ids": inputs["input_ids"].view(-1),
                "attention_mask": inputs["attention_mask"].view(-1),
                "labels": labels["input_ids"].view(-1),
            }
