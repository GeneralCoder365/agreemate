# downloader.py
import os
from datasets import load_dataset


def download_dataset(path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")):
    """
    Download the dataset to a specific location
    """
    # create the directory if it doesn't exist
    os.makedirs(path, exist_ok=True)

    # set HuggingFace cache directory to our desired location
    os.environ['HF_HOME'] = path
    os.environ['HUGGINGFACE_HUB_CACHE'] = os.path.join(path, 'hub')
    os.environ['TRANSFORMERS_CACHE'] = os.path.join(path, 'transformers')

    print(f"Downloading dataset to {path}...")

    try:
        # load and cache the dataset
        dataset = load_dataset("stanfordnlp/craigslist_bargains", cache_dir=path, trust_remote_code=True)

        # save each split to JSON for easier access
        for split_name, split_data in dataset.items():
            output_file = os.path.join(path, f"{split_name}.json")
            split_data.to_json(output_file)
            print(f"Saved {split_name} split to {output_file}")

        print("\nDataset info:")
        print(f"Number of splits: {len(dataset)}")
        for split in dataset:
            print(f"{split}: {len(dataset[split])} examples")

        return dataset

    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None


if __name__ == "__main__":
    download_dataset()