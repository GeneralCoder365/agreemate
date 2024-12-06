# model_loader.py
import os, gc, psutil
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
import torch
from pathlib import Path
from typing import Optional, Tuple
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import PretrainedConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class ModelLoader:
    """
    Simplified model loader specifically for Llama-3.1-8B-Instruct finetuning.
    Handles basic model loading, caching, and testing functionality.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize with optional cache directory."""
        # setup default cache directory in project structure
        finetuning_dir = os.path.dirname(os.path.abspath(__file__))

        if cache_dir: # point to snapshot cache if provided
            snapshot_dir = os.path.join(cache_dir, "snapshots")
            snapshot_folder = next(os.walk(snapshot_dir))[1][0] # get first folder
            self.MODEL_ID = os.path.join(snapshot_dir, snapshot_folder)
            self.cache_dir = Path(os.path.join(snapshot_dir, snapshot_folder))
        else: # point to online cache
            self.MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
            self.cache_dir = Path(finetuning_dir)

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
        torch_dtype: torch.dtype = torch.float32,
        local_only: bool = False
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load base Llama model from web or local (sharded or single) and tokenizer."""

        try:
            # Load tokenizer first with minimal configuration, without model reference
            if self.tokenizer is None:
                # load minimal config first
                minimal_config = PretrainedConfig.from_pretrained(
                    self.MODEL_ID,
                    trust_remote_code=True,
                    local_files_only=local_only
                )

                # extract only the essential tokenizer attributes
                tokenizer_config = {
                    "model_max_length": getattr(minimal_config, "max_position_embeddings", 4096),
                    "padding_side": "right",
                    "truncation_side": "right",
                    "clean_up_tokenization_spaces": True
                }

                # initialize tokenizer with essential config
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.MODEL_ID,
                    cache_dir=self.cache_dir,
                    trust_remote_code=True,
                    local_files_only=local_only,
                    use_fast=True,
                    # keep essential configs
                    **tokenizer_config
                )

                # ensure essential token IDs are set
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

                print("✓ Tokenizer loaded successfully without model reference")

            if self.model is None: # download or load model if not already loaded
                max_memory = { # calculate max GPU (85%) and CPU (70%) memory utilization for model
                    0: self.format_memory_size(
                        torch.cuda.get_device_properties(0).total_memory * 0.85
                    ) if torch.cuda.is_available() else None,
                    "cpu": self.format_memory_size(
                        psutil.virtual_memory().total * 0.70
                    )
                }

                # load model with memory constraints and offloading
                self.model = AutoModelForCausalLM.from_pretrained(
                    pretrained_model_name_or_path=self.MODEL_ID,
                    cache_dir=self.cache_dir,
                    torch_dtype=torch_dtype,
                    device_map="auto", # auto-distribute across devices
                    max_memory=max_memory, # apply memory constraints
                    offload_folder="./offload", # enable offloading unused layers to disk
                    trust_remote_code=True,
                    local_files_only=local_only
                )
                print("✓ Model loaded successfully")

            # ensure no lingering cross-references between model and tokenizer
            if hasattr(self.tokenizer, 'config'):
                if hasattr(self.tokenizer.config, 'architectures'):
                    delattr(self.tokenizer.config, 'architectures')
                if hasattr(self.tokenizer.config, '_name_or_path'):
                    delattr(self.tokenizer.config, '_name_or_path')

            return self.model, self.tokenizer

        except Exception as e:
            raise RuntimeError(f"Error loading Llama model: {str(e)}")

    def reload_model(self, model_path: str, torch_dtype: torch.dtype = torch.float16) -> AutoModelForCausalLM:
        """Reload a local model (sharded or single) from its specified path, with 4-bit quantization and LoRA config."""
        try:
            # calculate max GPU (85%) and CPU (70%) memory utilization for model
            max_memory = {
                0: self.format_memory_size(
                    torch.cuda.get_device_properties(0).total_memory * 0.85
                ) if torch.cuda.is_available() else None,
                "cpu": self.format_memory_size(
                    psutil.virtual_memory().total * 0.70
                )
            }

            # quantization config (4-bit with double quantization)
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

            # reload model with 4-bit quantization
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_path,
                quantization_config=quantization_config,
                device_map="auto",
                max_memory=max_memory,
                trust_remote_code=True
            )

            # prepare model for k-bit training
            model = prepare_model_for_kbit_training(model)

            # LoRA config parameters (approximated for 8B model)
            peft_config = LoraConfig(
                lora_alpha=16, # scaling factor
                lora_dropout=0.1, # dropout probability
                r=32, # rank of update matrices
                bias="none", # don't train biases
                task_type="CAUSAL_LM", # for language modeling
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj", # attention
                    "gate_proj", "up_proj", "down_proj" # mlp
                ]
            )

            # apply LoRA
            model = get_peft_model(model, peft_config)

            # print trainable parameters info
            model.print_trainable_parameters()

            return model

        except Exception as e:
            raise RuntimeError(f"Error reloading model from {model_path}: {str(e)}")

    def unload_model(self):
        """Unload the base model from memory."""
        if self.model:
            print("Unloading model from memory...")
            del self.model
            self.model = None

        # force garbage collection and GPU memory cleanup
        torch.cuda.empty_cache()
        torch.cuda.synchronize() # ensure all GPU operations are finished
        gc.collect()
        print("Memory cleanup completed.")


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


if __name__ == "__main__":
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