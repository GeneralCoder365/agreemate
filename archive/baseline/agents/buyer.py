# buyer.py
from typing import Dict, Optional
import dspy

from base_agent import BaseAgent


class BuyerStateAnalysis(dspy.Signature):
    """Analyzes negotiation state from buyer's perspective."""
    current_price: Optional[float] = dspy.InputField()
    target_price: float = dspy.InputField()
    strategy_name: str = dspy.InputField()
    category: str = dspy.InputField()
    num_turns: int = dspy.InputField()

    price_sentiment: str = dspy.OutputField(desc="how good/bad current price is relative to target")
    bargaining_power: str = dspy.OutputField(desc="current negotiating position strength")
    recommended_flexibility: float = dspy.OutputField(desc="how much to deviate from target (0-1)")


class BuyerAgent(BaseAgent):
    """
    Buyer agent for the AgreeMate baseline negotiation system.
    Implements buyer-specific negotiation behavior and strategy interpretation.
    """

    def __init__(
        self,
        strategy_name: str,
        target_price: float,
        category: str,
        max_price: Optional[float] = None,
        lm: dspy.LM = None
    ):
        """
        Initialize buyer agent.

        Args:
            strategy_name: Name of strategy from STRATEGIES
            target_price: Target purchase price
            category: Item category
            max_price: Maximum acceptable price (defaults to 10% above target)
            lm: DSPy language model for response generation
        """
        super().__init__(
            strategy_name=strategy_name,
            target_price=target_price,
            category=category,
            is_buyer=True,
            lm=lm
        )

        self.max_price = max_price or (target_price * 1.1)
        self.state_analyzer = dspy.ChainOfThought(BuyerStateAnalysis)
        self.best_offer_seen = float('inf') # track lowest offer

        # track negotiation progress
        self.total_concessions = 0
        self.moves_since_concession = 0

    def _analyze_state(self) -> Dict:
        """Analyze current negotiation state from buyer perspective."""
        if self.current_price is None:
            return {
                'price_sentiment': 'unknown',
                'bargaining_power': 'strong',
                'recommended_flexibility': 0.1 # start conservative
            }

        analysis = self.state_analyzer(
            current_price=self.current_price,
            target_price=self.target_price,
            strategy_name=self.strategy["name"],
            category=self.category,
            num_turns=self.num_turns
        )

        return {
            'price_sentiment': analysis.price_sentiment,
            'bargaining_power': analysis.bargaining_power,
            'recommended_flexibility': analysis.recommended_flexibility
        }

    def update_state(self, message: Dict[str, str]):
        """Update state with buyer-specific tracking."""
        super().update_state(message) # update base state
                                        # (conversation,price history, actions, etc)

        # track best offer
        if self.current_price is not None:
            self.best_offer_seen = min(self.best_offer_seen, self.current_price)

        # track concessions
        if len(self.price_history) >= 2:
            latest_change = self.price_history[-1] - self.price_history[-2]
            if latest_change > 0: # price went up (buyer concession)
                self.total_concessions += latest_change
                self.moves_since_concession = 0
            else:
                self.moves_since_concession += 1

    def predict_action(self) -> Dict:
        """Override to add buyer-specific strategy considerations."""
        prediction = super().predict_action() # get base prediction

        # add buyer context
        analysis = self._analyze_state()
        prediction['state_analysis'] = analysis

        # adjust based on max price if needed
        if self.current_price and self.current_price > self.max_price:
            if prediction['action'] == 'accept':
                prediction['action'] = 'reject'
                prediction['rationale'] += f"\nHowever, price (${self.current_price}) exceeds maximum (${self.max_price})"
                prediction['counter_price'] = self.max_price * 0.95 # slightly below max

        return prediction

    def generate_response(self, action: str, price: Optional[float] = None) -> str:
        """Generate buyer-specific responses."""
        analysis = self._analyze_state()

        # add buyer context to predictor input
        prediction = self.response_predictor(
            conversation_history=self.conversation_history,
            action=action,
            price=price,
            strategy_name=self.strategy["name"],
            category=self.category,
            is_buyer=True,
            # add buyer-specific context
            price_sentiment=analysis['price_sentiment'],
            bargaining_power=analysis['bargaining_power']
        )

        return prediction.response


def test_buyer_agent():
    """Test buyer agent functionality."""
    import os

    # set up test LM
    baseline_dir = os.path.dirname(os.path.abspath(__file__))
    agreemate_dir = os.path.dirname(baseline_dir)
    pretrained_dir = os.path.join(agreemate_dir, "models", "pretrained")

    test_lm = dspy.LM(
        model="meta-llama/Llama-3.1-8B-Instruct",
        api_base="http://localhost:11434",
        api_key="",
        cache_dir=pretrained_dir
    )

    # create buyer agent
    buyer = BuyerAgent(
        strategy_name="cooperative",
        target_price=100.0,
        category="electronics",
        max_price=120.0,
        lm=test_lm
    )

    # test initialization
    assert buyer.role == "buyer"
    assert buyer.max_price == 120.0
    assert buyer.best_offer_seen == float('inf')

    # test offer handling
    message = {
        "role": "seller",
        "content": "I can offer it for $150"
    }
    buyer.update_state(message)
    assert buyer.current_price == 150.0
    assert buyer.best_offer_seen == 150.0

    # test counter-offer generation
    response = buyer.step()
    assert response["role"] == "buyer"
    assert "content" in response

    print("✓ All buyer agent tests passed")
    return buyer

if __name__ == "__main__":
    buyer = test_buyer_agent()