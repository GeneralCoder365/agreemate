# config.py
"""
Central configuration for AgreeMate baseline system.
This file defines all configurable parameters and provides validation utilities
to ensure configurations are complete and coherent.
"""
from typing import Dict, List
from dataclasses import dataclass

from strategies import STRATEGIES


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    name: str # HuggingFace model name
    max_tokens: int
    temperature: float
    prompt_template: str

@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""
    num_scenarios: int
    max_turns: int
    turn_timeout: float
    models: List[str]
    strategies: List[str]


# core model configurations
MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "llama-3.1-8b": ModelConfig(
        name="meta-llama/Llama-3.1-8B-Instruct",
        max_tokens=128000,
        temperature=0.7,
        prompt_template=(
            "You are a {role} negotiating for {item}.\n"
            "Your strategy is: {strategy}\n\n"
            "Current conversation:\n{history}\n\n"
            "Your target price is: ${target_price}\n"
            "Respond as {role}:"
        )
    ),
    "llama-3.1-70b": ModelConfig(
        name="meta-llama/Llama-3.1-70B-Instruct",
        max_tokens=128000,
        temperature=0.7,
        prompt_template=(
            "You are a {role} negotiating for {item}.\n"
            "Your strategy is: {strategy}\n\n"
            "Current conversation:\n{history}\n\n"
            "Your target price is: ${target_price}\n"
            "Respond as {role}:"
        )
    ),
    "nemotron-70b": ModelConfig(
        name="nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
        max_tokens=128000,
        temperature=0.7,
        prompt_template=(
            "You are a {role} negotiating for {item}.\n"
            "Your strategy is: {strategy}\n\n"
            "Current conversation:\n{history}\n\n"
            "Your target price is: ${target_price}\n"
            "Respond as {role}:"
        )
    )
}

# pre-defined experiment configurations
EXPERIMENT_CONFIGS: Dict[str, ExperimentConfig] = {
    "baseline": ExperimentConfig(
        num_scenarios=100,
        max_turns=20,
        turn_timeout=30.0,
        models=["llama-3.1-8b"],
        strategies=["cooperative", "fair", "aggressive"]
    ),
    "model_comparison": ExperimentConfig(
        num_scenarios=200,
        max_turns=20,
        turn_timeout=30.0,
        models=["llama-3.1-8b", "llama-3.1-70b", "nemotron-70b"],
        strategies=["cooperative", "fair"]
    ),
    "strategy_analysis": ExperimentConfig(
        num_scenarios=150,
        max_turns=25,
        turn_timeout=30.0,
        models=["llama-3.1-70b"],
        strategies=["cooperative", "fair", "aggressive"]
    )
}

# analysis configuration
ANALYSIS_CONFIG = {
    "metrics": [
        "deal_rate",
        "avg_utility",
        "turns_to_completion",
        "strategy_adherence"
    ],
    "visualizations": [
        "outcome_dashboard",
        "process_visualization",
        "behavioral_analysis"
    ]
}


def validate_config(config: ExperimentConfig) -> bool:
    """Validate experiment configuration completeness and coherence."""
    for model in config.models: # check models exist
        if model not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model}")

    for strategy in config.strategies: # check strategies exist
        if strategy not in STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy}")

    # validate numeric parameters
    if config.num_scenarios < 1:
        raise ValueError("num_scenarios must be positive")
    if config.max_turns < 1:
        raise ValueError("max_turns must be positive")
    if config.turn_timeout <= 0:
        raise ValueError("turn_timeout must be positive")

    return True