from datasets import load_dataset, Dataset
from concurrent.futures import ProcessPoolExecutor
import tqdm
import os
import pandas as pd

# Global tokenizer (must be initialized in child processes)
tokenizer = None

def init_tokenizer(model_path):
    global tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code = True)

def tokenize_and_filter_row(row):
    try:
        prompt_len = len(tokenizer.tokenize(row['prompt']))
        response_len = len(tokenizer.tokenize(row['response']))
        return (prompt_len + response_len) <= 1024
    except Exception as e:
        print(f"Error tokenizing row: {e}")
        return False

def main():
    # Step 1: Load and rename columns
    output_dir = '/scratch4/workspace/dmaram_umass_edu-dmaram/learned-correctors/data/opencoder-stage2-edu'
    data = load_dataset("OpenCoder-LLM/opc-sft-stage2", "educational_instruct")
    data = data.rename_column("instruction", "prompt")
    data = data.rename_column("output", "response")

    train_data = data["train"]
    
    model_path = "Dream-org/Dream-Coder-v0-Base-7B"

    # Step 2: Filter using multiprocessing
    records = [dict(train_data[i]) for i in range(len(train_data))]
    
    with ProcessPoolExecutor(initializer=init_tokenizer, initargs=(model_path,)) as executor:
        results = list(tqdm.tqdm(executor.map(tokenize_and_filter_row, records), total=len(records)))

    filtered_indices = [i for i, keep in enumerate(results) if keep]
    print(f"Total filtered samples (<=1024 tokens): {len(filtered_indices)}")

    filtered_dataset = train_data.select(filtered_indices)  # This is still a Dataset

    # Step 3: Shuffle and split
    filtered_dataset = filtered_dataset.shuffle(seed=42)
    
    eval_size = 1000
    total_size = len(filtered_dataset)
    if total_size < eval_size:
        print(f"Warning: Only {total_size} samples available. Using all for training.")
        train_dataset = filtered_dataset
        eval_dataset = Dataset.from_dict({})  # empty dataset
    else:
        eval_dataset = filtered_dataset.select(range(eval_size))
        train_dataset = filtered_dataset.select(range(eval_size, total_size))

    # Step 4: Convert to Pandas and save
    os.makedirs(output_dir, exist_ok=True)

    # Convert to Pandas before saving
    train_df = train_dataset.to_pandas()
    eval_df = eval_dataset.to_pandas()

    train_df.to_parquet(os.path.join(output_dir, "train_data.parquet"), index=False)
    eval_df.to_parquet(os.path.join(output_dir, "eval_data.parquet"), index=False)

    print(f"Saved {len(train_df)} train samples and {len(eval_df)} eval samples.")
    
if __name__ == '__main__':
    main()