# Opinion Summarization of Yelp Reviews

This repository contains source code for various experiments to explore the feasibility of using a small, high-quality synthetic dataset, generated using a large language model such as PaLM 2, to fine-tune much smaller model to generate high-quality summaries.

## Install dependencies

```sh
pip install -r requirements.txt
```

## Download Data

Our project uses Yelp Dataset, consisting of over 6.9 million reviews and 150k businesses. To download the dataset, visit [Yelp Dataset](https://www.yelp.com/dataset).

## Process Data

The downloaded dataset requires some preprocessing before we can use it. It involves filtering out low rating businesses, remove excessively long or short reviews, etc.

Requirements:

- 5 GB free disk space
- 8 vCPUs with 64 GB RAM

Before running the preprocessing script, make sure to untar the downloaded data:

```sh
mkdir data
tar -xvf yelp_dataset.tar -C ./data
```

Now run the following commands to process the data.
Processing will take about 10 minutes.

```sh
cd src

# --data_root: directory containing untar yelp dataset json files
python process_data.py --data_root ./data
```

## Make Mini-Dataset

For experimentation, let's create a mini-dataset.

```sh
cd src

python make_dataset.py --data_file ./data/processed_dataset.json --out_dir ./data --out_file mini_dataset.json --num_businesses 10000 --max_reviews_per_business 8
```

## Download Test Dataset

To evaluate our generated dataset, prompts, and fine-tuned models, we use Yelp dataset released by Brazinskas et al. (2020). The dataset contains 100 human-written summaries generated by Amazon Mechanical Turk (AMT) workers, who summarized 8 reviews per business. Each business has 3 generated summaries by 3 workers, one summary per worker.

To download the dataset, run:

```sh
mkdir ./data/gold_summs

curl "https://raw.githubusercontent.com/abrazinskas/FewSum/master/artifacts/yelp/gold_summs/train.csv" --output "./data/gold_summs/train.csv"

curl "https://raw.githubusercontent.com/abrazinskas/FewSum/master/artifacts/yelp/gold_summs/val.csv" --output "./data/gold_summs/val.csv"

curl "https://raw.githubusercontent.com/abrazinskas/FewSum/master/artifacts/yelp/gold_summs/test.csv" --output "./data/gold_summs/test.csv"

curl "https://s3.us-east-2.amazonaws.com/unsup-sum/summaries_0-200_cleaned.csv" --output "./data/chu_dataset.csv"
```

## Preprocess Test Dataset

Now process the downloaded test dataset. This includes merging train/val/test splits together to make an evaluation dataset of human-written summaries; 100 businesses, 800 reviews, and 300 summaries.

```sh
cd src

python process_eval_dataset.py --data_root ./data/gold_summs --out_dir ./data
```

## Dataset Generation using PaLM 2 API

We conducted multiple experiments to evaluate our methods to generate data using PaLM 2 API. The experiments are as follows:

- `gen_data_exp1.py`: Simple prompt to generate data.
- `gen_data_exp2.py`: 2-step approach; first generate positive and negative summaries and then aggregate.
- `gen_data_exp3.py`: First generate business attributes and then use these attributes to generate summary focused on these top attributes.

These experiments are conducted on `eval_dataset` which contains human-written summaries.

To run each experiment:

```sh
# NOTE: Make sure you use GCP VM and have Vertex AI API enabled.
python gen_data_exp1.py --data_file ./data/eval_dataset.json --gcp_project my_project --gcp_region us-central1
```

To evaluate the generated data, run:

```sh
python evaluate_gen_data.py --data_file ./data/gen_dataset_exp1.json
```

### Generate dataset for training

Based on our experiemnts, attribute-based approach resulted in best generated summaries. So, to generate dataset for training models, run:

```sh
# NOTE: If you get quota errors, request higher quota for the PaLM API requests.
python generate_data.py --data_file ./data/mini_dataset.json
```

### Create train/validation split

Now we will use the generated dataset to create a training and validation split. For testing, we will use the human-written `eval_dataset` we processed above.

To create splits, run :

```sh
python make_train_val_split.py --data_file ./data/gen_dataset.json
```

## Fine-tuning BART

We used BART base model to experiment with different approaches as follows:

- `trainer_exp1.py`: vanilla model; given input, generate summary.
- `trainer_exp2.py`: controlled model; uses business attribute to make summary focus on these attributes.
- `trainer_exp3.py`: controlled model; uses business ratings as an input prefix.
- `trainer_exp4.py`: controlled model; uses business ratings as a text prompt as an input prefix.

To run these experiments:

```sh
python trainer_exp1.py
python trainer_exp2.py
python trainer_exp3.py
python trainer_exp4.py
```

## Evaluate Models

We will evaluate our models on the human-written `eval_dataset`.

We used Weights & Biases for logging.

To evaluate, run:

```sh
python eval.py --data_file ./data/gen_dataset_exp3.json --model_path ./exp1/checkpoint-600
python eval.py --data_file ./data/gen_dataset_exp3.json --model_path ./exp2/checkpoint-600 --use_attr True
python eval.py --data_file ./data/gen_dataset_exp3.json --model_path ./exp3/checkpoint-600 --use_rating True
python eval.py --data_file ./data/gen_dataset_exp3.json --model_path ./exp4/checkpoint-600 --use_rating True --use_rating_as_text True
```
