# experiment_runner.py
import logging, asyncio
from typing import Dict, List, Optional
from pathlib import Path

from config import EXPERIMENT_CONFIGS
from dspy_manager import DSPyManager
from scenario_manager import ScenarioManager
from negotiation_runner import NegotiationRunner, NegotiationConfig
from experiment_state import ExperimentTracker
from metrics_collector import MetricsCollector
from utils.data_loader import DataLoader

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ExperimentRunner:
    """
    Main experiment orchestrator for AgreeMate baseline system.
    Handles experiment setup, execution, and result collection while
    preserving agent autonomy in negotiations.
    """

    def __init__(
        self,
        config_name: str,
        output_dir: str,
        experiment_name: Optional[str] = None
    ):
        """
        Initialize experiment runner.

        Args:
            config_name: Name of config from EXPERIMENT_CONFIGS
            output_dir: Directory for experiment outputs
            experiment_name: Optional unique name for this run
        """
        if config_name not in EXPERIMENT_CONFIGS:
            raise ValueError(f"Unknown configuration: {config_name}")
        self.config = EXPERIMENT_CONFIGS[config_name]

        # setup output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # initialize components
        self.data_loader = DataLoader()
        self.scenario_manager = ScenarioManager(self.data_loader)
        self.dspy_manager = DSPyManager()
        self.negotiation_runner = NegotiationRunner(self.dspy_manager)
        self.metrics = MetricsCollector()

        # initialize experiment state tracker
        self.tracker = ExperimentTracker(
            output_dir=str(self.output_dir),
            experiment_name=experiment_name or config_name,
            config=self.config
        )

        self.model_combinations = self._generate_model_combinations()

    def _generate_model_combinations(self) -> List[Dict]:
        """Generate all model and strategy combinations to test."""
        combinations = []
        for buyer_model in self.config.models:
            for seller_model in self.config.models:
                for buyer_strategy in self.config.strategies:
                    for seller_strategy in self.config.strategies:
                        combinations.append({
                            'buyer_model': buyer_model,
                            'seller_model': seller_model,
                            'buyer_strategy': buyer_strategy,
                            'seller_strategy': seller_strategy
                        })
        return combinations

    async def _run_single_combination(
        self,
        combination: Dict,
        scenarios: List[str]
    ):
        """Run negotiations for one model/strategy combination."""
        configs = []
        for scenario_id in scenarios:
            scenario = self.scenario_manager.get_scenario(scenario_id)
            configs.append(
                NegotiationConfig(
                    scenario=scenario,
                    buyer_model=combination['buyer_model'],
                    seller_model=combination['seller_model'],
                    buyer_strategy=combination['buyer_strategy'],
                    seller_strategy=combination['seller_strategy'],
                    max_turns=self.config.max_turns,
                    turn_timeout=self.config.turn_timeout
                )
            )

        # run negotiations
        results = await self.negotiation_runner.run_batch(configs)

        # process results
        for scenario_id, metrics in results.items():
            if metrics.final_price is not None: # successful negotiation
                self.tracker.record_completion(
                    scenario_id,
                    metrics,
                    combination['buyer_model'],
                    combination['seller_model']
                )

                # analyze negotiation
                self.metrics.analyze_negotiation(
                    metrics=metrics,
                    buyer_model=combination['buyer_model'],
                    seller_model=combination['seller_model'],
                    buyer_strategy=combination['buyer_strategy'],
                    seller_strategy=combination['seller_strategy'],
                    scenario_id=scenario_id,
                    initial_price=configs[0].scenario.list_price,
                    target_prices={
                        'buyer': configs[0].scenario.buyer_target,
                        'seller': configs[0].scenario.seller_target
                    }
                )
            else: # failed negotiation
                self.tracker.record_failure(
                    scenario_id,
                    Exception("No agreement reached"),
                    combination
                )

    async def run(self):
        """Execute complete experiment."""
        logger.info(f"Starting experiment with config: {self.config}")

        # get scenarios
        scenarios = self.scenario_manager.create_evaluation_batch(
            split='test',
            size=self.config.num_scenarios,
            balanced_categories=True
        )
        scenario_ids = [s.scenario_id for s in scenarios]

        # run all combinations
        for combination in self.model_combinations:
            logger.info(f"Running combination: {combination}")
            await self._run_single_combination(combination, scenario_ids)

            # checkpoint after each combination
            self.tracker.save_checkpoint()

        # save final results
        self.tracker.save_final_results()

        # export analysis
        analysis = self.metrics.export_analysis()
        analysis_path = self.output_dir / 'analysis.json'
        with open(analysis_path, 'w') as f:
            import json
            json.dump(analysis, f, indent=2)

        return {
            'config': self.config,
            'results': self.tracker.state,
            'analysis': analysis
        }


async def main():
    """Run experiment from command line."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='baseline')
    parser.add_argument('--output', required=True)
    parser.add_argument('--name', default=None)
    args = parser.parse_args()
    
    runner = ExperimentRunner(
        config_name=args.config,
        output_dir=args.output,
        experiment_name=args.name
    )
    results = await runner.run()
    logger.info(f"Experiment complete: {results}")

if __name__ == "__main__":
    asyncio.run(main())