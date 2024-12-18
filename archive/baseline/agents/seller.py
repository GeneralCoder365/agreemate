# seller.py
from typing import Dict, Optional
import dspy

from base_agent import BaseAgent


class SellerStateAnalysis(dspy.Signature):
    """Analyzes negotiation state from seller's perspective."""
    current_price: Optional[float] = dspy.InputField()
    target_price: float = dspy.InputField()
    strategy_name: str = dspy.InputField()
    category: str = dspy.InputField()
    num_turns: int = dspy.InputField()
    initial_price: Optional[float] = dspy.InputField()

    price_sentiment: str = dspy.OutputField(desc="how good/bad current offer is relative to target")
    market_position: str = dspy.OutputField(desc="strength of current market position")
    recommended_flexibility: float = dspy.OutputField(desc="how much to deviate from target (0-1)")


class SellerAgent(BaseAgent):
    """
    Seller agent for the AgreeMate baseline negotiation system.
    Implements seller-specific negotiation behavior and strategy interpretation.
    """

    def __init__(
        self,
        strategy_name: str,
        target_price: float,
        category: str,
        min_price: Optional[float] = None,
        initial_price: Optional[float] = None,
        lm: dspy.LM = None
    ):
        """
        Initialize seller agent.
        
        Args:
            strategy_name: Name of strategy from STRATEGIES
            target_price: Target selling price
            category: Item category
            min_price: Minimum acceptable price (defaults to 10% below target)
            initial_price: Initial listing price (defaults to 20% above target)
            lm: DSPy language model for response generation
        """
        super().__init__(
            strategy_name=strategy_name,
            target_price=target_price,
            category=category,
            is_buyer=False,
            lm=lm
        )

        self.min_price = min_price or (target_price * 0.9)
        self.initial_price = initial_price or (target_price * 1.2)
        self.state_analyzer = dspy.ChainOfThought(SellerStateAnalysis)
        self.best_offer_seen = 0 # track highest offer

        # track negotiation progress
        self.total_discounts = 0
        self.moves_since_discount = 0
        self.initial_offer_made = False

    def _analyze_state(self) -> Dict:
        """Analyze current negotiation state from seller perspective."""
        if self.current_price is None:
            return {
                'price_sentiment': 'initial',
                'market_position': 'strong',
                'recommended_flexibility': 0.1 # start conservative
            }

        analysis = self.state_analyzer(
            current_price=self.current_price,
            target_price=self.target_price,
            strategy_name=self.strategy["name"],
            category=self.category,
            num_turns=self.num_turns,
            initial_price=self.initial_price
        )

        return {
            'price_sentiment': analysis.price_sentiment,
            'market_position': analysis.market_position,
            'recommended_flexibility': analysis.recommended_flexibility
        }

    def update_state(self, message: Dict[str, str]):
        """Update state with seller-specific tracking."""
        super().update_state(message) # update base state
                                        # (conversation,price history, actions, etc)

        # track best offer
        if self.current_price is not None:
            self.best_offer_seen = max(self.best_offer_seen, self.current_price)

        # track discounts
        if len(self.price_history) >= 2:
            latest_change = self.price_history[-1] - self.price_history[-2]
            if latest_change < 0: # price went down (seller discount)
                self.total_discounts += abs(latest_change)
                self.moves_since_discount = 0
            else:
                self.moves_since_discount += 1

    def predict_action(self) -> Dict:
        """Override to add seller-specific strategy considerations."""
        # handle initial offer
        if not self.initial_offer_made and not self.conversation_history:
            return {
                'action': 'offer',
                'counter_price': self.initial_price,
                'rationale': f"Making initial offer at ${self.initial_price}",
                'state_analysis': self._analyze_state()
            }

        # get base prediction
        prediction = super().predict_action()

        # add seller context
        analysis = self._analyze_state()
        prediction['state_analysis'] = analysis

        # adjust based on minimum price if needed #!(GUARDRAIL: NEVER ACCEPT BELOW MIN)
        if self.current_price and self.current_price < self.min_price:
            if prediction['action'] == 'accept':
                prediction['action'] = 'reject'
                prediction['rationale'] += f"\nHowever, offer (${self.current_price}) below minimum (${self.min_price})"
                prediction['counter_price'] = self.min_price * 1.05 # slightly above min

        # flag that initial offer has been made
        self.initial_offer_made = True

        return prediction

    def generate_response(self, action: str, price: Optional[float] = None) -> str:
        """Generate seller-specific responses."""
        analysis = self._analyze_state()

        # add seller context to predictor input
        prediction = self.response_predictor(
            conversation_history=self.conversation_history,
            action=action,
            price=price,
            strategy_name=self.strategy["name"],
            category=self.category,
            is_buyer=False,
            # add seller-specific context
            price_sentiment=analysis['price_sentiment'],
            market_position=analysis['market_position']
        )

        return prediction.response


def test_seller_agent():
    """Test seller agent functionality."""
    import os

    # set up test LM
    baseline_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    agreemate_dir = os.path.dirname(baseline_dir)
    pretrained_dir = os.path.join(agreemate_dir, "models", "pretrained")

    test_lm = dspy.LM(
        model="meta-llama/Llama-3.1-8B-Instruct",
        api_base="http://localhost:11434",
        api_key="",
        cache_dir=pretrained_dir
    )

    # create seller agent
    seller = SellerAgent(
        strategy_name="cooperative",
        target_price=100.0,
        category="electronics",
        min_price=80.0,
        initial_price=120.0,
        lm=test_lm
    )

    # test initialization
    assert seller.role == "seller"
    assert seller.min_price == 80.0
    assert seller.initial_price == 120.0
    assert seller.best_offer_seen == 0

    # test initial offer
    response = seller.step()
    assert response["role"] == "seller"
    assert "120" in response["content"] # should include initial price

    # test offer handling
    message = {
        "role": "buyer",
        "content": "I can offer $90"
    }
    seller.update_state(message)
    assert seller.current_price == 90.0
    assert seller.best_offer_seen == 90.0

    # test counter-offer generation
    response = seller.step()
    assert response["role"] == "seller"
    assert "content" in response

    print("✓ All seller agent tests passed")
    return seller

if __name__ == "__main__":
    seller = test_seller_agent()