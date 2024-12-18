# dspy_manager.py
import os, logging, dspy
from typing import Dict, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from config import MODEL_CONFIGS, ModelConfig
from strategies import STRATEGIES

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@dataclass
class DSPyLMConfig:
    """Extended configuration for DSPy language models."""
    base_config: ModelConfig
    strategy_name: Optional[str] = None
    role: Optional[str] = None # 'buyer' or 'seller'

    def get_context_config(self) -> Dict:
        """Get strategy and role-specific configuration."""
        config = {
            'temperature': self.base_config.temperature,
            'max_tokens': self.base_config.max_tokens
        }

        if self.strategy_name:
            strategy = STRATEGIES[self.strategy_name]
            # adjust temperature based on strategy
            if strategy['risk_tolerance'] == 'high':
                config['temperature'] *= 1.2
            elif strategy['risk_tolerance'] == 'low':
                config['temperature'] *= 0.8

            # adjust max tokens based on communication style
            if strategy['communication_style'].startswith('Clear'):
                config['max_tokens'] = min(config['max_tokens'], 256)

        return config


class DSPyManager:
    """
    Manages DSPy language models with strategy-aware configurationsfor the AgreeMate negotiation system.
    Handles initialization, caching, parallel execution contexts, and strategy-specific configurations.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize DSPy manager with optional cache directory."""
        self.model_configs = MODEL_CONFIGS
        self.lm_cache: Dict[str, dspy.LM] = {}
        self.context_configs: Dict[str, DSPyLMConfig] = {}

        # set up caching dir
        if cache_dir:
            os.environ['DSPY_CACHE_DIR'] = cache_dir

        self.executor = ThreadPoolExecutor(max_workers=4)

    def _create_lm(self, model_key: str, config: DSPyLMConfig) -> dspy.LM:
        """Create a new DSPy LM instance with configuration."""
        context_config = config.get_context_config()

        try:
            return dspy.LM(
                config.base_config.name,
                temperature=context_config['temperature'],
                max_tokens=context_config['max_tokens'],
                cache=True # always cache for efficiency
            )
        except Exception as e:
            logger.error(f"Failed to create LM for {model_key}: {str(e)}")
            raise

    def get_lm(
        self,
        model_key: str,
        strategy_name: Optional[str] = None,
        role: Optional[str] = None
    ) -> dspy.LM:
        """
        Get or create a DSPy LM with specific configuration.

        Args:
            model_key: Key from MODEL_CONFIGS
            strategy_name: Optional strategy name for specialized config
            role: Optional role (buyer/seller) for specialized config

        Returns:
            Configured DSPy LM instance
        """
        if model_key not in self.model_configs:
            raise ValueError(f"Unknown model: {model_key}")

        # create context-specific key
        context_key = f"{model_key}_{strategy_name}_{role}"

        # check cache first
        if context_key in self.lm_cache:
            return self.lm_cache[context_key]

        # create new configuration
        config = DSPyLMConfig(
            base_config=self.model_configs[model_key],
            strategy_name=strategy_name,
            role=role
        )
        self.context_configs[context_key] = config

        # create and cache new LM
        lm = self._create_lm(model_key, config)
        self.lm_cache[context_key] = lm

        return lm

    async def run_parallel(self, tasks: list) -> list:
        """Execute multiple LM tasks in parallel."""
        futures = []
        for task in tasks:
            future = self.executor.submit(task)
            futures.append(future)

        results = []
        for future in futures:
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Task failed: {str(e)}")
                results.append(None)

        return results

    def configure_negotiation(
        self,
        buyer_model: str,
        seller_model: str,
        buyer_strategy: str,
        seller_strategy: str
    ) -> tuple:
        """
        Configure DSPy LMs for a negotiation pair.

        Returns:
            Tuple of (buyer_lm, seller_lm)
        """
        buyer_lm = self.get_lm(
            buyer_model,
            strategy_name=buyer_strategy,
            role='buyer'
        )

        seller_lm = self.get_lm(
            seller_model,
            strategy_name=seller_strategy,
            role='seller'
        )

        return buyer_lm, seller_lm

    def clear_cache(self):
        """Clear all cached LMs."""
        self.lm_cache.clear()
        self.context_configs.clear()


def test_dspy_manager():
    """Test DSPy manager functionality."""
    manager = DSPyManager()

    # test basic LM creation
    lm = manager.get_lm("llama-3.1-8b")
    assert lm is not None

    # test strategy-specific configuration
    lm_aggressive = manager.get_lm(
        "llama-3.1-8b",
        strategy_name="aggressive",
        role="buyer"
    )
    assert lm_aggressive is not None

    # test negotiation pair configuration
    buyer_lm, seller_lm = manager.configure_negotiation(
        "llama-3.1-8b",
        "llama-3.1-8b",
        "cooperative",
        "fair"
    )
    assert buyer_lm is not None
    assert seller_lm is not None

    print("✓ All DSPy manager tests passed")
    return manager

if __name__ == "__main__":
    manager = test_dspy_manager()