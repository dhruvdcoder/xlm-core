import os
import datasets
from transformers import GPT2TokenizerFast
import dotenv

dotenv.load_dotenv(override=False)
dotenv.load_dotenv(".secrets.env", override=True)
datasets.utils.logging.set_verbosity(
    datasets.utils.logging.log_levels["debug"]
)


def filter_fn(example, tokenizer):
    token_ids = tokenizer.encode(example["text"], add_special_tokens=False)
    return len(token_ids) <= 1024


def main():
    dataset_name = "Skylion007/openwebtext"
    num_proc = 10
    dataset = datasets.load_dataset(
        dataset_name,
        num_proc=num_proc,
        trust_remote_code=True,
    )

    # tokenizer using GPT2Tokenizer and remove sequences longer than 1024 tokens
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    print(f"Before filtering:\n {dataset}")

    dataset = dataset.filter(
        filter_fn,
        num_proc=num_proc,
        fn_kwargs={"tokenizer": tokenizer},
        desc="Filtering dataset",
    )
    print(f"After filtering:\n {dataset}")

    # split the dataset into train and val
    # When downloaded, it has only one split "train"
    split_datasets = dataset["train"].train_test_split(
        test_size=10000, seed=2357, shuffle=True
    )
    # rename the test split to validation
    split_datasets["validation"] = split_datasets.pop("test")
    print(f"After splitting:\n {split_datasets}")

    # push to huggingface hub
    split_datasets.push_to_hub(
        "dhruveshpatel/owt-gpt2-1024-split", token=os.getenv("HF_HUB_KEY")
    )


if __name__ == "__main__":
    main()
