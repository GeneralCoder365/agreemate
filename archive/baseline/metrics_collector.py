# metrics_collector.py
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from negotiation_runner import NegotiationMetrics
from strategies import STRATEGIES

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@dataclass
class StrategyMetrics:
    """Metrics for strategy analysis."""
    strategy_name: str
    adherence_scores: List[float] = field(default_factory=list)
    success_rate: float = 0.0
    avg_turns: float = 0.0
    avg_utility: float = 0.0
    language_metrics: Dict[str, float] = field(default_factory=dict)

    def update(self, success: bool, turns: int, utility: float, adherence: float):
        """Update metrics with new negotiation result."""
        self.adherence_scores.append(adherence)
        n = len(self.adherence_scores)

        # update running averages
        self.success_rate = ((n-1) * self.success_rate + float(success)) / n
        self.avg_turns = ((n-1) * self.avg_turns + turns) / n
        self.avg_utility = ((n-1) * self.avg_utility + utility) / n


@dataclass
class NegotiationAnalysis:
    """Complete analysis of a negotiation session."""
    # core identifiers
    scenario_id: str
    buyer_model: str
    seller_model: str

    # basic metrics
    duration: float
    turns_taken: int
    final_price: Optional[float]

    # strategy analysis
    buyer_strategy: str
    seller_strategy: str
    buyer_adherence: float
    seller_adherence: float

    # price trajectory
    initial_price: float
    target_prices: Dict[str, float] # buyer/seller targets
    price_history: List[float]

    # interaction analysis
    message_lengths: List[int]
    response_times: List[float]

    def compute_metrics(self) -> Dict[str, float]:
        """Compute derived metrics."""
        metrics = {
            'success': self.final_price is not None,
            'efficiency': self._compute_efficiency(),
            'fairness': self._compute_fairness(),
            'avg_response_time': np.mean(self.response_times),
            'price_convergence': self._compute_convergence()
        }
        return metrics

    def _compute_efficiency(self) -> float:
        """Compute negotiation efficiency score (0-1)."""
        if not self.final_price:
            return 0.0

        # consider turns, time, and price movement
        max_expected_turns = 20 # from config
        turn_score = 1 - (self.turns_taken / max_expected_turns)

        time_per_turn = self.duration / self.turns_taken
        time_score = np.exp(-time_per_turn / 30) # 30 sec baseline

        price_movement = np.diff(self.price_history)
        directness = 1 - (np.abs(price_movement).sum() / 
                         abs(self.price_history[-1] - self.price_history[0]))

        return np.mean([turn_score, time_score, directness])

    def _compute_fairness(self) -> float:
        """Compute negotiation fairness score (0-1)."""
        if not self.final_price:
            return 0.0

        # distance from midpoint
        fair_price = (self.target_prices['buyer'] + 
                     self.target_prices['seller']) / 2
        price_fairness = 1 - (abs(self.final_price - fair_price) /
                             abs(self.target_prices['buyer'] - 
                                 self.target_prices['seller']))

        # balance of concessions
        buyer_movement = abs(self.final_price - self.price_history[0])
        seller_movement = abs(self.final_price - self.price_history[1])
        concession_balance = 1 - abs(buyer_movement - seller_movement) / \
                               (buyer_movement + seller_movement)

        return np.mean([price_fairness, concession_balance])

    def _compute_convergence(self) -> float:
        """Compute price convergence score (0-1)."""
        if len(self.price_history) < 3:
            return 0.0

        # measure how directly prices moved toward final agreement
        price_diffs = np.diff(self.price_history)
        ideal_path = abs(self.price_history[-1] - self.price_history[0])
        actual_path = np.abs(price_diffs).sum()

        return ideal_path / actual_path if actual_path > 0 else 0.0


class MetricsCollector:
    """
    Collects and analyzes metrics from negotiation experiments.
    Provides both real-time tracking and post-experiment analysis.
    """

    def __init__(self):
        """Initialize metrics collector."""
        self.strategy_metrics: Dict[str, StrategyMetrics] = {
            name: StrategyMetrics(strategy_name=name)
            for name in STRATEGIES.keys()
        }

        self.model_pairs: Dict[str, List[NegotiationAnalysis]] = {}
        self.negotiations: Dict[str, NegotiationAnalysis] = {}

    def analyze_negotiation(
        self,
        metrics: NegotiationMetrics,
        buyer_model: str,
        seller_model: str,
        buyer_strategy: str,
        seller_strategy: str,
        scenario_id: str,
        initial_price: float,
        target_prices: Dict[str, float]
    ) -> NegotiationAnalysis:
        """Analyze a completed negotiation."""

        analysis = NegotiationAnalysis(
            scenario_id=scenario_id,
            buyer_model=buyer_model,
            seller_model=seller_model,
            duration=metrics.compute_duration(),
            turns_taken=metrics.turns_taken,
            final_price=metrics.final_price,
            buyer_strategy=buyer_strategy,
            seller_strategy=seller_strategy,
            buyer_adherence=metrics.strategy_adherence['buyer'],
            seller_adherence=metrics.strategy_adherence['seller'],
            initial_price=initial_price,
            target_prices=target_prices,
            price_history=[p for p in metrics.messages if 'price' in p],
            message_lengths=[len(m['content']) for m in metrics.messages],
            response_times=[m.get('response_time', 0) for m in metrics.messages]
        )

        # update strategy metrics
        computed = analysis.compute_metrics()
        self.strategy_metrics[buyer_strategy].update(
            success=computed['success'],
            turns=metrics.turns_taken,
            utility=metrics.buyer_utility or 0.0,
            adherence=metrics.strategy_adherence['buyer']
        )
        self.strategy_metrics[seller_strategy].update(
            success=computed['success'],
            turns=metrics.turns_taken,
            utility=metrics.seller_utility or 0.0,
            adherence=metrics.strategy_adherence['seller']
        )

        # store analysis
        self.negotiations[scenario_id] = analysis
        pair_key = f"{buyer_model}_{seller_model}"
        if pair_key not in self.model_pairs:
            self.model_pairs[pair_key] = []
        self.model_pairs[pair_key].append(analysis)

        return analysis

    def get_strategy_summary(self) -> pd.DataFrame:
        """Get summary statistics for each strategy."""
        records = []
        for strategy_name, metrics in self.strategy_metrics.items():
            records.append({
                'strategy': strategy_name,
                'success_rate': metrics.success_rate,
                'avg_turns': metrics.avg_turns,
                'avg_utility': metrics.avg_utility,
                'adherence_mean': np.mean(metrics.adherence_scores),
                'adherence_std': np.std(metrics.adherence_scores)
            })
        return pd.DataFrame.from_records(records)

    def get_model_pair_summary(self) -> pd.DataFrame:
        """Get summary statistics for each model pair combination."""
        records = []
        for pair_key, analyses in self.model_pairs.items():
            buyer_model, seller_model = pair_key.split('_')

            # compute aggregate metrics
            success_rate = np.mean([
                bool(a.final_price) for a in analyses
            ])
            avg_efficiency = np.mean([
                a.compute_metrics()['efficiency'] for a in analyses
            ])
            avg_fairness = np.mean([
                a.compute_metrics()['fairness'] for a in analyses
            ])

            records.append({
                'buyer_model': buyer_model,
                'seller_model': seller_model,
                'num_negotiations': len(analyses),
                'success_rate': success_rate,
                'avg_efficiency': avg_efficiency,
                'avg_fairness': avg_fairness
            })
        return pd.DataFrame.from_records(records)

    def export_analysis(self) -> Dict:
        """Export complete analysis results."""
        return {
            'strategy_summary': self.get_strategy_summary().to_dict('records'),
            'model_summary': self.get_model_pair_summary().to_dict('records'),
            'negotiations': {
                sid: analysis.compute_metrics()
                for sid, analysis in self.negotiations.items()
            }
        }


def test_metrics_collector():
    """Test metrics collector functionality."""
    from datetime import datetime, timedelta

    collector = MetricsCollector()

    # create test metrics
    metrics = NegotiationMetrics(
        start_time=datetime.now() - timedelta(minutes=5),
        end_time=datetime.now(),
        turns_taken=10,
        final_price=150.0,
        buyer_utility=0.8,
        seller_utility=0.7,
        strategy_adherence={'buyer': 0.9, 'seller': 0.85},
        messages=[
            {'role': 'buyer', 'content': 'Offer: $100', 'price': 100},
            {'role': 'seller', 'content': 'Counter: $200', 'price': 200},
            {'role': 'buyer', 'content': 'Accept: $150', 'price': 150}
        ]
    )

    # test analysis
    analysis = collector.analyze_negotiation(
        metrics=metrics,
        buyer_model='llama-3.1-8b',
        seller_model='llama-3.1-8b',
        buyer_strategy='cooperative',
        seller_strategy='fair',
        scenario_id='test_1',
        initial_price=200.0,
        target_prices={'buyer': 100.0, 'seller': 180.0}
    )
    assert analysis.turns_taken == 10
    assert analysis.final_price == 150.0

    # test summaries
    strategy_summary = collector.get_strategy_summary()
    assert len(strategy_summary) == len(STRATEGIES)

    model_summary = collector.get_model_pair_summary()
    assert len(model_summary) > 0

    print("✓ All metrics collector tests passed")
    return collector

if __name__ == "__main__":
    collector = test_metrics_collector()