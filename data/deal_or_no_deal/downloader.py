# downloader.py
import os
from datasets import load_dataset


def download_dataset(path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")):
    """
    Download the 'deal_or_no_dialog' dataset and save it in its original text-based format.
    """
    # create the directory if it doesn't exist
    os.makedirs(path, exist_ok=True)

    # set HuggingFace cache directory to our desired location
    os.environ['HF_HOME'] = path
    os.environ['HUGGINGFACE_HUB_CACHE'] = os.path.join(path, 'hub')
    os.environ['TRANSFORMERS_CACHE'] = os.path.join(path, 'transformers')

    print(f"Downloading dataset to {path}...")
    try: # load and cache the dataset
        dataset = load_dataset("mikelewis0/deal_or_no_dialog", cache_dir=path, trust_remote_code=True)

        # save each split in text format as in the original dataset
        for split_name, split_data in dataset.items():
            output_file = os.path.join(path, f"{split_name}.txt")
            with open(output_file, "w", encoding="utf-8") as f:
                for example in split_data:
                    serialized = serialize_example(example)
                    if serialized:
                        f.write(serialized + "\n")
            print(f"Saved {split_name} split to {output_file}")

        print("\nDataset info:")
        print(f"Number of splits: {len(dataset)}")
        for split in dataset:
            print(f"{split}: {len(dataset[split])} examples")

        return dataset

    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None


def serialize_example(example):
    """
    Serialize a single data example into the original text format with tags.
    """
    try: # flatten and serialize fields into the tagged format
        input_data = " ".join(map(str, flatten_sequence(example["input"])))
        partner_input_data = " ".join(map(str, flatten_sequence(example["partner_input"])))
        dialogue = example["dialogue"].replace("\n", " ")  # Ensure no unintended newlines
        output_data = example["output"]

        return (
            f"<input> {input_data} </input> "
            f"<dialogue> {dialogue} </dialogue> "
            f"<output> {output_data} </output> "
            f"<partner_input> {partner_input_data} </partner_input>"
        )
    except (KeyError, TypeError) as e:
        print(f"Error serializing example: {example}, Error: {e}")
        return None


def flatten_sequence(sequence):
    """
    Flatten a sequence of dictionaries (e.g., 'count' and 'value') into a single list.
    """
    if "count" in sequence and "value" in sequence:
        return [val for pair in zip(sequence["count"], sequence["value"]) for val in pair]
    else:
        raise KeyError("Missing 'count' or 'value' in sequence")


if __name__ == "__main__":
    download_dataset()