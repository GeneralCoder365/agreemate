# AgreeMate: Teaching LLMs to Haggle

A framework for training Large Language Models (LLMs) to perform strategic price negotiations through natural language. AgreeMate combines role-specialized fine-tuning with systematic model comparison across scales and architectures.

[![Listen to Paper Summary](https://img.shields.io/badge/🎧_Listen_to_Paper_Summary-NotebookLM-blue)](https://notebooklm.google.com/notebook/603ffc75-00a3-4269-8a8a-e10d4d9634ec/audio)

## Overview

AgreeMate presents a comprehensive finetuning/testing framework and negotiation system that explores LLMs' capabilities in strategic price bargaining. Our approach combines:

- Role-specialized fine-tuning (buyer, seller, generalist)
- Systematic model comparison across scales (3B to 70B parameters)
- Chain of thought prompting
- Personality-driven behavior modeling
- Attention mechanism analysis

### Related Work

Our work builds on [*"Decoupling Strategy and Generation in Negotiation Dialogues"*](https://arxiv.org/abs/1808.09637) by He et al. (2018), which introduced a modular approach to negotiation using coarse dialogue acts for strategy control.

AgreeMate advances this by leveraging LLMs to unify strategy and generation without requiring explicit intermediate representations, enabling more nuanced and scalable negotiation behaviors.

## Key Features

- **Model Variants**: Specialized fine-tuned models for buyer, seller, and generalist roles
- **Multiple Scales**: Support for LLaMA models from 3B to 70B parameters
- **Personality Types**: Configurable negotiation personalities (aggressive, fair, passive)
- **Chain of Thought**: Enhanced reasoning capabilities through CoT prompting
- **Comprehensive Metrics**: Novel measurements for fairness, bias, efficiency, and more

## Installation

```bash
git clone https://github.com/GeneralCoder365/agreemate
cd agreemate
pip install -r requirements.txt
```

## Project Structure

- `data/`: Training datasets and preprocessing utilities
  - `craigslist_bargains/`: 6.6k human negotiation dialogues
  - `deal_or_no_deal/`: 12k multi-issue bargaining dialogues

- `finetuning/`: Model training and fine-tuning components
  - Fine-tuning configurations
  - Training logs and checkpoints
  - TensorBoard visualization data

- `results/`: Analysis notebooks and evaluation results
  - Performance metrics across model combinations
  - Comparative analysis visualizations
  - Example negotiation transcripts

## Key Technologies

- Parameter-Efficient Fine-Tuning (PEFT) with LoRA
- 4-bit quantization for memory efficiency
- Gradient checkpointing
- Mixed precision training
- Advanced learning rate scheduling

## Usage

### Basic Negotiation

```python
from agreemate import Negotiator

# initialize buyer and seller agents
buyer = Negotiator(role="buyer", model="llama-3.2-3B-buyer")
seller = Negotiator(role="seller", model="llama-3.2-3B-seller")

# run negotiation
dialogue = negotiate(buyer, seller, scenario="car_charger")
```

### Personality Configuration

```python
# create agents with specific personalities
aggressive_buyer = Negotiator(
    role="buyer",
    personality="aggressive",
    model="llama-3.2-3B-buyer"
)

fair_seller = Negotiator(
    role="seller",
    personality="fair",
    model="llama-3.2-3B-seller"
)
```

## Results & Findings

- Larger models demonstrate improved agreement rates and fairness
- Chain of Thought prompting enhances exploratory behavior
- Personality traits significantly influence negotiation dynamics
- Attention analysis reveals semantic understanding of negotiation concepts

## Citation

```bibtex
@article{chatterjee2024agreemate,
  title={AgreeMate: Teaching LLMs to Haggle},
  author={Chatterjee, Ainesh and Miller, Samuel and Parepally, Nithin},
  year={2024}
}
```

## Contributors

- Ainesh Chatterjee - Dataset preparation, Fine-tuning, and personality analysis
- Samuel Miller - Probing analysis and visualization
- Nithin Parepally - Architecture design and evaluation framework

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Media

Listen to a summary of our research paper on [NotebookLM](https://notebooklm.google.com/notebook/603ffc75-00a3-4269-8a8a-e10d4d9634ec/audio)

## Acknowledgments

- **Stanford NLP Group**: For providing the [CraigslistBargains dataset](https://huggingface.co/datasets/stanfordnlp/craigslist_bargains), which comprises over 6,600 human-human negotiation dialogues across various product categories, facilitating research in negotiation strategies and language generation. 

- **Facebook Research**: For the [Deal or No Deal Negotiator dataset](https://huggingface.co/datasets/mikelewis0/deal_or_no_dialog), a substantial collection of human-human negotiations on a multi-issue bargaining task, enabling the development of end-to-end models for negotiation. 

- **Meta AI**: For developing the LLaMA model family, which served as the foundation for our large language model experiments.

- **He et al. (2018)**: For their foundational work, [*"Decoupling Strategy and Generation in Negotiation Dialogues"*](https://arxiv.org/abs/1808.09637), introducing a modular approach to negotiation by leveraging coarse dialogue acts for strategy control, inspiring our efforts to unify strategy and generation in negotiation through LLMs.
