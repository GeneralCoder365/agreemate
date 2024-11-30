# model_loader.py
import os, psutil
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
import torch
from pathlib import Path
from typing import Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer


class ModelLoader:
    """
    Simplified model loader specifically for Llama-3.2-1B-Instruct finetuning.
    Handles basic model loading, caching, and testing functionality.
    """
    MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize with optional cache directory."""
        # setup default cache directory in project structure
        finetuning_dir = os.path.dirname(os.path.abspath(__file__))
        self.cache_dir = Path(cache_dir or finetuning_dir)

        # ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"Using cache directory: {self.cache_dir}")

        # track loaded components
        self.model = None
        self.tokenizer = None

    def format_memory_size(self, size_in_bytes):
        """Converts raw bytes into a string with appropriate units (e.g., GB)."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_in_bytes < 1024:
                return f"{size_in_bytes:.1f}{unit}"
            size_in_bytes /= 1024
        return f"{size_in_bytes:.1f}TB"

    def load_model_and_tokenizer(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype: torch.dtype = torch.bfloat16,
        local_only: bool = False
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load Llama model and tokenizer."""

        try:
            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.MODEL_ID,
                    cache_dir=self.cache_dir,
                    trust_remote_code=True,
                    local_files_only=local_only
                )
                print("✓ Tokenizer loaded successfully")

            if self.model is None:
                # calculate memory limits (85% of GPU memory, 60% of CPU memory)
                total_gpu_memory = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
                gpu_memory_limit = total_gpu_memory * 0.85
                total_cpu_memory = psutil.virtual_memory().total
                cpu_memory_limit = total_cpu_memory * 0.60
                gpu_memory_limit_str = self.format_memory_size(gpu_memory_limit)
                cpu_memory_limit_str = self.format_memory_size(cpu_memory_limit)

                self.model = AutoModelForCausalLM.from_pretrained(
                    self.MODEL_ID,
                    cache_dir=self.cache_dir,
                    torch_dtype=torch_dtype,
                    device_map="auto",
                    local_files_only=local_only,
                    max_memory={ # set memory limits for GPU and CPU
                        0: gpu_memory_limit_str,
                        "cpu": cpu_memory_limit_str
                    },
                    offload_folder="./offload", # enable offloading unused layers to disk
                    trust_remote_code=True
                )
                print("✓ Model loaded successfully")

            return self.model, self.tokenizer

        except Exception as e:
            raise RuntimeError(f"Error loading Llama model: {str(e)}")

    def test_model(self) -> bool:
        """Run basic model test to verify functionality."""
        try:
            # load model if not already loaded
            if self.model is None or self.tokenizer is None:
                self.load_model_and_tokenizer()

            # test message with clear instruction format
            test_message = [{
                "role": "system",
                "content": "You are a helpful assistant. Please respond exactly as instructed."
            }, {
                "role": "user", 
                "content": "Please respond with exactly these words: 'test successful'"
            }]

            # tokenize with proper attention mask
            chat_input = self.tokenizer.apply_chat_template(
                test_message,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            )

            # create attention mask
            attention_mask = torch.ones_like(chat_input)

            # move to correct device
            inputs = {
                "input_ids": chat_input.to(self.model.device),
                "attention_mask": attention_mask.to(self.model.device)
            }

            # generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=20,
                    temperature=None, # removed temperature since we're not sampling
                    do_sample=False, # kept deterministic for test
                    top_p=None, # removed top_p since we're not sampling
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # decode response
            response = self.tokenizer.batch_decode(
                outputs[:, chat_input.shape[1]:],
                skip_special_tokens=True
            )[0]

            print(f"Test response: {response}")
            return "test successful" in response.lower()

        except Exception as e:
            print(f"Test failed: {str(e)}")
            return False

    def reload_model(self, model_path: str) -> AutoModelForCausalLM:
        """Helper to reload a model from its specified path."""
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            return model
        except Exception as e:
            raise RuntimeError(f"Error loading model from {model_path}: {str(e)}")


def main():
    """Test the model loader functionality."""
    loader = ModelLoader()

    print(f"Using torch device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print("Starting Llama model test...")

    try:
        # try loading model
        loader.load_model_and_tokenizer()
        print("Model and tokenizer loaded successfully")

        # run model test
        if loader.test_model():
            print("✓ Model test passed")
        else:
            print("✗ Model test failed")

        # verify cache
        cached_files = list(Path(loader.cache_dir).glob("**/*"))
        if cached_files:
            print(f"✓ Cache directory populated with {len(cached_files)} files")
        else:
            print("✗ Cache directory is empty!")

    except Exception as e:
        print(f"Setup failed: {str(e)}")

if __name__ == "__main__":
    main()