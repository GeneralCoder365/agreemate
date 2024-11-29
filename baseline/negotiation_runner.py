# negotiation_runner.py
import logging, asyncio
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from agents.buyer import BuyerAgent
from agents.seller import SellerAgent
from scenario_manager import NegotiationScenario
from dspy_manager import DSPyManager

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@dataclass
class NegotiationConfig:
    """Configuration for a single negotiation."""
    scenario: NegotiationScenario
    buyer_model: str
    seller_model: str
    buyer_strategy: str
    seller_strategy: str
    max_turns: int
    turn_timeout: float

@dataclass
class NegotiationMetrics:
    """Metrics collected during negotiation."""
    start_time: datetime
    end_time: Optional[datetime] = None
    turns_taken: int = 0
    final_price: Optional[float] = None
    buyer_utility: Optional[float] = None
    seller_utility: Optional[float] = None
    strategy_adherence: Dict[str, float] = None
    messages: List[Dict] = None

    def compute_duration(self) -> float:
        """Compute negotiation duration in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


class NegotiationRunner:
    """
    Manages execution of individual negotiation sessions for the AgreeMate system.
    Handles agent initialization, turn management, and metrics collection.
    """

    def __init__(
        self,
        dspy_manager: DSPyManager,
        max_concurrent: int = 4
    ):
        """
        Initialize negotiation runner.

        Args:
            dspy_manager: DSPy LM manager instance
            max_concurrent: Maximum concurrent negotiations
        """
        self.dspy_manager = dspy_manager
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.active_negotiations: Dict[str, NegotiationMetrics] = {}

    def _initialize_agents(
        self,
        config: NegotiationConfig
    ) -> Tuple[BuyerAgent, SellerAgent]:
        """Initialize buyer and seller agents for negotiation."""
        # get DSPy LMs with strategy-specific configurations
        buyer_lm, seller_lm = self.dspy_manager.configure_negotiation(
            config.buyer_model,
            config.seller_model,
            config.buyer_strategy,
            config.seller_strategy
        )

        # create agents with scenario context
        buyer = BuyerAgent(
            strategy_name=config.buyer_strategy,
            target_price=config.scenario.buyer_target,
            category=config.scenario.category,
            max_price=config.scenario.list_price,
            lm=buyer_lm
        )

        seller = SellerAgent(
            strategy_name=config.seller_strategy,
            target_price=config.scenario.seller_target,
            category=config.scenario.category,
            initial_price=config.scenario.list_price,
            min_price=config.scenario.seller_target * 0.9, # 10% below target
            lm=seller_lm
        )

        return buyer, seller


    def _validate_price_movement(
        self,
        agent_role: str,
        new_price: float,
        metrics: NegotiationMetrics
    ) -> bool:
        """Validate price movements follow negotiation rules."""
        if not metrics.messages:
            return True # first offer is always valid

        last_price = next(
            (m['price'] for m in reversed(metrics.messages) 
            if m['price'] is not None),
            None
        )

        if last_price is None:
            return True

        # buyer should offer higher, seller should offer lower
        if agent_role == 'buyer':
            return new_price >= last_price
        else:
            return new_price <= last_price

    async def _run_negotiation_turn(
        self,
        buyer: BuyerAgent,
        seller: SellerAgent,
        metrics: NegotiationMetrics,
        timeout: float
    ) -> bool:
        """
        Execute one turn of negotiation.

        Returns:
            bool: True if negotiation should continue
        """
        try:
            # alternate between buyer and seller
            current_agent = buyer if metrics.turns_taken % 2 == 0 else seller

            # execute turn with timeout
            async with asyncio.timeout(timeout):
                response = await current_agent.step()

                # validate message structure
                if not isinstance(response, dict) or 'role' not in response:
                    raise ValueError("Invalid message format")

                # ensure required fields
                response.setdefault('price', None)
                response.setdefault('status', 'counter')

                # validate price movement
                if response['price'] is not None:
                    valid = self._validate_price_movement(
                        agent_role=response['role'],
                        new_price=response['price'],
                        metrics=metrics
                    )
                    if not valid:
                        response['status'] = 'reject'
                        logger.warning(f"Invalid price movement: {response}")

            # update metrics
            metrics.turns_taken += 1
            metrics.messages.append(response)

            # handle completion
            if response['status'] in ['accept', 'reject']:
                metrics.end_time = datetime.now()
                metrics.final_price = (
                    response['price'] if response['status'] == 'accept' 
                    else None
                )
                return False

            return True

        except asyncio.TimeoutError:
            logger.warning(f"Turn timeout after {timeout}s")
            metrics.end_time = datetime.now()
            return False
        except Exception as e:
            logger.error(f"Turn error: {str(e)}")
            metrics.end_time = datetime.now()
            return False

    async def run_negotiation(
        self,
        config: NegotiationConfig
    ) -> NegotiationMetrics:
        """
        Execute a complete negotiation session.

        Args:
            config: Negotiation configuration

        Returns:
            Completed negotiation metrics
        """
        async with self.semaphore:
            # initialize metrics tracking
            metrics = NegotiationMetrics(
                start_time=datetime.now(),
                strategy_adherence={
                    'buyer': 1.0, # will be updated during negotiation
                    'seller': 1.0
                }
            )

            try:
                # initialize agents
                buyer, seller = self._initialize_agents(config)

                # track active negotiation
                self.active_negotiations[config.scenario.scenario_id] = metrics

                # run negotiation turns
                continue_negotiation = True
                while (continue_negotiation and 
                       metrics.turns_taken < config.max_turns):
                    continue_negotiation = await self._run_negotiation_turn(
                        buyer, seller, metrics, config.turn_timeout
                    )

                # compute final metrics
                self._compute_final_metrics(metrics, buyer, seller)

                return metrics

            except Exception as e:
                logger.error(f"Negotiation failed: {str(e)}")
                metrics.end_time = datetime.now()
                return metrics
            finally: # remove from active negotiations
                self.active_negotiations.pop(config.scenario.scenario_id, None)


    def _compute_final_metrics(
        self,
        metrics: NegotiationMetrics,
        buyer: BuyerAgent,
        seller: SellerAgent
    ):
        """Compute final negotiation metrics."""
        if metrics.final_price:
            # compute utilities
            metrics.buyer_utility = (
                buyer.compute_utility(metrics.final_price)
                if hasattr(buyer, 'compute_utility') else None
            )
            metrics.seller_utility = (
                seller.compute_utility(metrics.final_price)
                if hasattr(seller, 'compute_utility') else None
            )

            # update strategy adherence
            if hasattr(buyer, 'get_strategy_adherence'):
                metrics.strategy_adherence['buyer'] = buyer.get_strategy_adherence()
            if hasattr(seller, 'get_strategy_adherence'):
                metrics.strategy_adherence['seller'] = seller.get_strategy_adherence()

    async def run_batch(
        self,
        configs: List[NegotiationConfig]
    ) -> Dict[str, NegotiationMetrics]:
        """Run a batch of negotiations in parallel."""
        tasks = []
        for config in configs:
            task = asyncio.create_task(self.run_negotiation(config))
            tasks.append((config.scenario.scenario_id, task))

        results = {}
        for scenario_id, task in tasks:
            try:
                metrics = await task
                results[scenario_id] = metrics
            except Exception as e:
                logger.error(f"Batch task failed: {str(e)}")

        return results


def test_negotiation_runner():
    """Test negotiation runner functionality."""
    from scenario_manager import ScenarioManager
    from utils.data_loader import DataLoader

    # initialize components
    data_loader = DataLoader()
    scenario_manager = ScenarioManager(data_loader)
    dspy_manager = DSPyManager()
    runner = NegotiationRunner(dspy_manager)

    # create test scenario
    scenarios = scenario_manager.create_evaluation_batch(
        split='test',
        size=1
    )

    # create test configuration
    config = NegotiationConfig(
        scenario=scenarios[0],
        buyer_model="llama-3.1-8b",
        seller_model="llama-3.1-8b",
        buyer_strategy="cooperative",
        seller_strategy="fair",
        max_turns=5,
        turn_timeout=30.0
    )

    # run test negotiation
    async def run_test():
        metrics = await runner.run_negotiation(config)
        assert metrics.turns_taken > 0
        assert metrics.end_time is not None
        print("✓ Negotiation completed successfully")
        return metrics

    import asyncio
    metrics = asyncio.run(run_test())

    print("✓ All negotiation runner tests passed")
    return runner, metrics

if __name__ == "__main__":
    runner, metrics = test_negotiation_runner()