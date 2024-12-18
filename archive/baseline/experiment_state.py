# experiment_state.py
"""
Manages experiment state, checkpointing, and results persistence for AgreeMate.
Tracks experiment progress and enables experiment recovery if needed.
"""
import os, json, logging
from typing import Dict, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd

from config import ExperimentConfig
from negotiation_runner import NegotiationMetrics

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@dataclass
class ExperimentState:
    """Current state of an experiment run."""
    experiment_name: str
    config: ExperimentConfig
    start_time: datetime
    scenarios_total: int
    scenarios_completed: int = 0
    scenarios_failed: int = 0
    last_checkpoint: Optional[datetime] = None

@dataclass
class ModelPairMetrics:
    """Metrics for a specific model pair combination."""
    buyer_model: str
    seller_model: str
    num_negotiations: int = 0
    deal_rate: float = 0.0
    avg_turns: float = 0.0
    avg_buyer_utility: float = 0.0
    avg_seller_utility: float = 0.0
    avg_duration: float = 0.0
    strategy_adherence: Dict[str, float] = None


class ExperimentTracker:
    """
    Tracks experiment progress and manages results persistence.
    Handles checkpointing and experiment recovery.
    """

    def __init__(
        self,
        output_dir: str,
        experiment_name: str,
        config: ExperimentConfig
    ):
        """
        Initialize experiment tracker.

        Args:
            output_dir: Directory for output files
            experiment_name: Unique experiment identifier
            config: Experiment configuration
        """
        self.output_dir = output_dir
        self.results_dir = os.path.join(output_dir, 'results')
        self.checkpoints_dir = os.path.join(output_dir, 'checkpoints')

        # create dirs
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        # initialize state
        self.state = ExperimentState(
            experiment_name=experiment_name,
            config=config,
            start_time=datetime.now(),
            scenarios_total=config.num_scenarios
        )

        # initialize results tracking
        self.completed_negotiations: Dict[str, NegotiationMetrics] = {}
        self.failed_negotiations: Dict[str, Dict] = {}
        self.model_pair_metrics: Dict[str, ModelPairMetrics] = {}

    def record_completion(
        self,
        scenario_id: str,
        metrics: NegotiationMetrics,
        buyer_model: str,
        seller_model: str
    ):
        """Record successful negotiation completion."""
        self.completed_negotiations[scenario_id] = metrics
        self.state.scenarios_completed += 1

        # update model pair metrics
        pair_key = f"{buyer_model}_{seller_model}"
        if pair_key not in self.model_pair_metrics:
            self.model_pair_metrics[pair_key] = ModelPairMetrics(
                buyer_model=buyer_model,
                seller_model=seller_model
            )

        pair_metrics = self.model_pair_metrics[pair_key]
        pair_metrics.num_negotiations += 1
        pair_metrics.deal_rate = (
            len([n for n in self.completed_negotiations.values() 
                 if n.final_price is not None]) / 
            pair_metrics.num_negotiations
        )
        pair_metrics.avg_turns += (
            metrics.turns_taken - pair_metrics.avg_turns
        ) / pair_metrics.num_negotiations
        if metrics.buyer_utility:
            pair_metrics.avg_buyer_utility += (
                metrics.buyer_utility - pair_metrics.avg_buyer_utility
            ) / pair_metrics.num_negotiations
        if metrics.seller_utility:
            pair_metrics.avg_seller_utility += (
                metrics.seller_utility - pair_metrics.avg_seller_utility
            ) / pair_metrics.num_negotiations
        pair_metrics.avg_duration += (
            metrics.compute_duration() - pair_metrics.avg_duration
        ) / pair_metrics.num_negotiations

    def record_failure(
        self,
        scenario_id: str,
        error: Exception,
        context: Dict
    ):
        """Record negotiation failure."""
        self.failed_negotiations[scenario_id] = {
            'error': str(error),
            'context': context,
            'timestamp': datetime.now().isoformat()
        }
        self.state.scenarios_failed += 1

    def save_checkpoint(self):
        """Save experiment checkpoint."""
        timestamp = datetime.now()
        checkpoint_path = os.path.join(
            self.checkpoints_dir,
            f"checkpoint_{timestamp:%Y%m%d_%H%M%S}.json"
        )

        checkpoint = {
            'state': asdict(self.state),
            'completed_negotiations': {
                k: asdict(v) for k, v in self.completed_negotiations.items()
            },
            'failed_negotiations': self.failed_negotiations,
            'model_pair_metrics': {
                k: asdict(v) for k, v in self.model_pair_metrics.items()
            }
        }

        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2, default=str)
            
        self.state.last_checkpoint = timestamp
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load experiment state from checkpoint."""
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)

        # restore state
        self.state = ExperimentState(**checkpoint['state'])
        self.completed_negotiations = {
            k: NegotiationMetrics(**v)
            for k, v in checkpoint['completed_negotiations'].items()
        }
        self.failed_negotiations = checkpoint['failed_negotiations']
        self.model_pair_metrics = {
            k: ModelPairMetrics(**v)
            for k, v in checkpoint['model_pair_metrics'].items()
        }

        logger.info(f"Loaded checkpoint from {checkpoint_path}")

    def save_final_results(self):
        """Save final experiment results."""
        results_path = os.path.join(
            self.results_dir,
            f"{self.state.experiment_name}_results.json"
        )

        # create summary DataFrame
        summary_data = []
        for scenario_id, metrics in self.completed_negotiations.items():
            summary_data.append({
                'scenario_id': scenario_id,
                'turns': metrics.turns_taken,
                'duration': metrics.compute_duration(),
                'final_price': metrics.final_price,
                'buyer_utility': metrics.buyer_utility,
                'seller_utility': metrics.seller_utility,
                'strategy_adherence_buyer': metrics.strategy_adherence['buyer'],
                'strategy_adherence_seller': metrics.strategy_adherence['seller']
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(
            os.path.join(self.results_dir, f"{self.state.experiment_name}_summary.csv"),
            index=False
        )

        # save complete results package
        results = {
            'experiment_name': self.state.experiment_name,
            'config': asdict(self.state.config),
            'summary': {
                'total_scenarios': self.state.scenarios_total,
                'completed': self.state.scenarios_completed,
                'failed': self.state.scenarios_failed,
                'duration': (datetime.now() - self.state.start_time).total_seconds()
            },
            'model_pair_metrics': {
                k: asdict(v) for k, v in self.model_pair_metrics.items()
            },
            'completed_negotiations': {
                k: asdict(v) for k, v in self.completed_negotiations.items()
            },
            'failed_negotiations': self.failed_negotiations
        }

        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        logger.info(f"Saved final results to {results_path}")


def test_experiment_tracker():
    """Test experiment tracker functionality."""
    from config import EXPERIMENT_CONFIGS
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        # create tracker
        tracker = ExperimentTracker(
            output_dir=temp_dir,
            experiment_name="test_experiment",
            config=EXPERIMENT_CONFIGS['baseline']
        )

        # test recording completion
        metrics = NegotiationMetrics(
            start_time=datetime.now(),
            end_time=datetime.now(),
            turns_taken=5,
            final_price=100.0,
            buyer_utility=0.8,
            seller_utility=0.7,
            strategy_adherence={'buyer': 0.9, 'seller': 0.85}
        )

        tracker.record_completion(
            "test_scenario_1",
            metrics,
            "llama-3.1-8b",
            "llama-3.1-8b"
        )

        # test checkpoint save/load
        tracker.save_checkpoint()
        checkpoint_file = os.path.join(
            tracker.checkpoints_dir,
            os.listdir(tracker.checkpoints_dir)[0]
        )

        new_tracker = ExperimentTracker(
            output_dir=temp_dir,
            experiment_name="test_experiment",
            config=EXPERIMENT_CONFIGS['baseline']
        )
        new_tracker.load_checkpoint(checkpoint_file)
        assert new_tracker.state.scenarios_completed == 1

        # test final results
        tracker.save_final_results()
        results_file = os.path.join(
            tracker.results_dir,
            f"{tracker.state.experiment_name}_results.json"
        )
        assert os.path.exists(results_file)

        print("✓ All experiment tracker tests passed")
        return tracker

if __name__ == "__main__":
    tracker = test_experiment_tracker()