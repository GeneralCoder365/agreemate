# model_loader.py
import os, torch
from typing import Dict, Optional, Tuple
from pathlib import Path
from functools import lru_cache
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..config import MODEL_CONFIGS


class ModelLoader:
    """
    Model loading and caching utilities for the AgreeMate baseline system.
    Handles loading and caching of pretrained models and tokenizers.
    """
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize model loader with optional cache directory."""
        baseline_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        agreemate_dir = os.path.dirname(baseline_dir)
        pretrained_dir = os.path.join(agreemate_dir, "models", "pretrained")
        self.cache_dir = cache_dir or pretrained_dir
        self.loaded_models = {}
        self.loaded_tokenizers = {}

        if not os.path.exists(self.cache_dir):
            raise ValueError(f"Cache directory does not exist: {self.cache_dir}")
        else:
            print(f"Using cache directory: {self.cache_dir}")

    @lru_cache(maxsize=3)
    def load_model_and_tokenizer(
        self,
        model_key: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype: torch.dtype = torch.bfloat16 # using bfloat16 as per NVIDIA example
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load model and tokenizer from HuggingFace, with caching.
        """
        if model_key not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model key: {model_key}")

        config = MODEL_CONFIGS[model_key]
        model_name = config["name"]

        try:
            # load tokenizer
            if model_name not in self.loaded_tokenizers:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=self.cache_dir # check cache first
                )
                self.loaded_tokenizers[model_name] = tokenizer

            # load model
            if model_name not in self.loaded_models:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    cache_dir=self.cache_dir, # check cache first
                    torch_dtype=torch_dtype,
                    device_map="auto" # let transformers handle multi-GPU
                )
                self.loaded_models[model_name] = model

            return self.loaded_models[model_name], self.loaded_tokenizers[model_name]

        except Exception as e:
            raise RuntimeError(f"Error loading model {model_key}: {str(e)}")

    def get_model_config(self, model_key: str) -> Dict:
        """Get configuration for specified model."""
        if model_key not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model key: {model_key}")
        return MODEL_CONFIGS[model_key].copy()

    async def generate_response(
        self,
        model_key: str,
        messages: list, # using messages format as shown in NVIDIA example
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Generate response from specified model using chat format.
        """
        model, tokenizer = self.load_model_and_tokenizer(model_key)
        config = self.get_model_config(model_key)

        # use config defaults if not specified
        max_new_tokens = max_new_tokens or 4096 # default from NVIDIA example
        temperature = temperature or config["temperature"]

        # format chat messages using model's chat template
        tokenized_message = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True
        )

        inputs = {
            'input_ids': tokenized_message['input_ids'].to(model.device),
            'attention_mask': tokenized_message['attention_mask'].to(model.device)
        }

        # generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        # extract only the new tokens (response)
        generated_tokens = outputs[:, len(tokenized_message['input_ids'][0]):]
        response = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

        return response.strip()


def model_loader():
    """Loads models, and tests tokenization, caching for all models."""
    loader = ModelLoader()

    # iterate over all models in the config
    for model_key, config in MODEL_CONFIGS.items():
        print(f"Loading and testing model: {model_key}")
        try:
            # load model and tokenizer
            model, tokenizer = loader.load_model_and_tokenizer(model_key)

            # apply a test chat message
            test_message = [{"role": "user", "content": "How many r in strawberry?"}]
            tokenized = tokenizer.apply_chat_template(
                test_message,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            )
            assert tokenized is not None, f"Tokenization failed for {model_key}"
            print(f"✓ {model_key} loaded, tokenized, and ready")
        except Exception as e:
            print(f"✗ Error testing {model_key}: {str(e)}")

    # confirm that cache directory is populated
    pretrained_dir = loader.cache_dir
    cached_files = list(Path(pretrained_dir).glob("**/*"))
    if not cached_files:
        print("✗ Cache directory is empty!")
    else:
        print(f"✓ Cache directory populated with {len(cached_files)} files")

    print("All model tests complete.")

if __name__ == "__main__":
    loader = model_loader()