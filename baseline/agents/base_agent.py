# base_agent.py
import os, dspy
from typing import Dict, List, Optional

from ..strategies import STRATEGIES, CATEGORY_CONTEXT


class StateExtractor(dspy.Signature):
    """Extracts structured state information from negotiation messages."""
    message_content: str = dspy.InputField()
    is_buyer: bool = dspy.InputField()

    extracted_price: Optional[float] = dspy.OutputField(desc="price mentioned in message, if any")
    detected_action: str = dspy.OutputField(desc="detected action: offer/counter/accept/reject/none")
    reasoning: str = dspy.OutputField(desc="explanation of extraction")

class NegotiationState(dspy.Signature):
    """
    Tracks the state of a negotiation conversation.
    Used to predict next action in negotiation based on current state.
    """
    conversation_history: List[Dict] = dspy.InputField()
    target_price: float = dspy.InputField()
    current_price: float = dspy.InputField()
    strategy_name: str = dspy.InputField()
    category: str = dspy.InputField()
    is_buyer: bool = dspy.InputField()
    num_turns: int = dspy.InputField()

    rationale: str = dspy.OutputField(desc="reasoning about next action")
    action: str = dspy.OutputField(desc="next action: accept/reject/counter")
    counter_price: Optional[float] = dspy.OutputField(desc="if action is counter, the counter-offer price")

class NegotiationResponse(dspy.Signature):
    """Generates a natural language response during negotiation."""
    complete_prompt: str = dspy.InputField(desc="Full formatted prompt with strategy & context")
    conversation_history: List[Dict] = dspy.InputField()
    action: str = dspy.InputField()
    price: Optional[float] = dspy.InputField()
    strategy_name: str = dspy.InputField()
    category: str = dspy.InputField()
    is_buyer: bool = dspy.InputField()

    response: str = dspy.OutputField(desc="natural language response following strategy guidance")

class StrategyAnalysis(dspy.Signature):
    """
    Analyzes how well an utterance adheres to assigned negotiation strategy.
    This signature helps ensure agents maintain consistent strategy execution.
    """
    message: str = dspy.InputField(desc="message to analyze")
    assigned_strategy: str = dspy.InputField(desc="strategy name from STRATEGIES")
    role: str = dspy.InputField(desc="buyer or seller")
    context: Optional[Dict] = dspy.InputField(desc="additional context", default=None)

    adherence_score: float = dspy.OutputField(desc="strategy adherence score (0-1)")
    analysis: str = dspy.OutputField(desc="explanation of scoring")
    detected_tactics: List[str] = dspy.OutputField(desc="identified negotiation tactics")

class TacticDetection(dspy.Signature):
    """
    Detects specific negotiation tactics used in messages.
    Helps track and analyze negotiation behaviors.
    """
    message: str = dspy.InputField(desc="message to analyze")
    strategy_context: Optional[str] = dspy.InputField(desc="assigned strategy", default=None)
    
    tactics: List[str] = dspy.OutputField(desc="detected negotiation tactics")
    confidence: List[float] = dspy.OutputField(desc="confidence scores for detections")
    explanation: str = dspy.OutputField(desc="reasoning for detected tactics")

class LanguageAnalysis(dspy.Signature):
    """
    Analyzes language characteristics of negotiation messages.
    Tracks sophistication, emotional content, and persuasion attempts.
    """
    text: str = dspy.InputField(desc="text to analyze")
    context: Dict = dspy.InputField(desc="relevant context info")
    
    complexity_score: float = dspy.OutputField(desc="language complexity score (0-1)")
    emotional_content: Dict[str, float] = dspy.OutputField(desc="emotion scores")
    persuasion_techniques: List[str] = dspy.OutputField(desc="identified techniques")
    coherence_score: float = dspy.OutputField(desc="response coherence score (0-1)")


class BaseAgent:
    """
    Base Agent for the AgreeMate baseline negotiation system.
    Defines core functionality and abstract methods that both buyer and seller child agents will implement.
    """

    def __init__(
        self,
        strategy_name: str,
        target_price: float,
        category: str,
        is_buyer: bool,
        lm: dspy.LM
    ):
        """
        Initialize negotiation agent.

        Args:
            strategy_name: Name of strategy from STRATEGIES
            target_price: Target price for this agent
            category: Item category (electronics, vehicles, etc)
            is_buyer: Whether this is a buyer (True) or seller (False)
            lm: DSPy language model for response generation
        """
        if strategy_name not in STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        self.strategy = STRATEGIES[strategy_name]
        self.category_context = CATEGORY_CONTEXT[category]
        self.target_price = target_price
        self.category = category
        self.is_buyer = is_buyer
        self.role = "buyer" if is_buyer else "seller"

        # state tracking
        self.conversation_history = []
        self.price_history = []
        self.roles_sequence = []
        self.last_action = None
        self.current_price = None
        self.num_turns = 0

        # set up predictor modules
        self.state_predictor = dspy.ChainOfThought(NegotiationState)
        self.response_predictor = dspy.ChainOfThought(NegotiationResponse)
        self.state_extractor = dspy.ChainOfThought(StateExtractor)

        # configure DSPy to use provided language model across all modules
        dspy.settings.configure(lm=lm)

    def update_state(self, message: Dict[str, str]):
        """
        Updates negotiation state using LLM extraction.
        Uses StateExtractor to get structured information from messages.

        Args:
            message: Dict containing 'role' and 'content' of message
        """
        if not isinstance(message, dict) or 'role' not in message or 'content' not in message:
            raise ValueError("Invalid message format")

        # extract structured info using LLM
        extraction = self.state_extractor(
            message_content=message['content'],
            is_buyer=self.is_buyer
        )

        # update conversation state
        self.conversation_history.append(message)
        self.roles_sequence.append(message['role'])
        self.num_turns += 1

        # update price state if new price detected
        if extraction.extracted_price is not None:
            self.current_price = extraction.extracted_price
            self.price_history.append(extraction.extracted_price)

        # update action state
        self.last_action = extraction.detected_action

        # add extraction reasoning to debug info if needed
        if hasattr(self, 'extraction_history'):
            self.extraction_history.append(extraction.reasoning)

    def _get_prediction_context(self) -> Dict:
        """Get context for predictions."""
        return {
            "conversation_history": self.conversation_history,
            "target_price": self.target_price,
            "current_price": self.current_price,
            "strategy_name": self.strategy["name"],
            "category": self.category,
            "is_buyer": self.is_buyer,
            "num_turns": self.num_turns
        }

    def predict_action(self) -> Dict:
        """
        Predict next action in negotiation.

        Returns:
            Dict containing action prediction with rationale
        """
        prediction = self.state_predictor(**self._get_prediction_context())
        return {
            "rationale": prediction.rationale,
            "action": prediction.action,
            "counter_price": prediction.counter_price
        }

    def analyze_strategy_adherence(self, message: Dict[str, str]) -> Dict:
        """
        Analyze how well a message adheres to assigned strategy.

        Args:
            message: Message dictionary with role and content

        Returns:
            Dictionary containing adherence analysis
        """
        analysis = self.state_analyzer(
            message=message['content'],
            assigned_strategy=self.strategy['name'],
            role=self.role,
            context={
                'category': self.category,
                'target_price': self.target_price,
                'current_price': self.current_price
            }
        )

        return {
            'adherence_score': analysis.adherence_score,
            'analysis': analysis.analysis,
            'tactics': analysis.detected_tactics
        }

    def analyze_language(self, message: str) -> Dict:
        """
        Analyze language characteristics of a message.

        Args:
            message: Message text to analyze

        Returns:
            Dictionary containing language analysis
        """
        analysis = self.language_analyzer(
            text=message,
            context={
                'category': self.category,
                'strategy': self.strategy['name'],
                'role': self.role
            }
        )

        return {
            'complexity': analysis.complexity_score,
            'emotions': analysis.emotional_content,
            'techniques': analysis.persuasion_techniques,
            'coherence': analysis.coherence_score
        }

    def generate_response(self, action: str, price: Optional[float] = None) -> str:
        """Generate natural language response."""
        from utils.model_loader import MODEL_CONFIGS

        context = self._get_prediction_context()

        # format conversation history for prompt
        history_text = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in self.conversation_history
        ])

        # get prompt template and fill
        model_name = self.lm.model_name.split('/')[-1] # extract base name
        template = MODEL_CONFIGS[model_name].prompt_template
        prompt = template.format(
            role=self.role,
            strategy=self.strategy['description'],
            history=history_text,
            target_price=self.target_price,
            item=context.get('item', {'title': 'the item'})['title']
        )

        # add strategy-specific guidance
        prompt += f"\n\nYour negotiation approach: {self.strategy['initial_approach']}"
        prompt += f"\nCommunication style: {self.strategy['communication_style']}"
        prompt += f"\nCategory context: {self.category_context['market_dynamics']}"

        context.update({
            "complete_prompt": prompt,
            "action": action,
            "price": price
        })

        prediction = self.response_predictor(**context)
        return prediction.response

    def step(self) -> Dict[str, str]:
        """
        Take a negotiation step: predict action and generate response.

        Returns:
            Dict with role and content for response message
        """
        # predict next action
        prediction = self.predict_action()

        # generate natural language response
        response = self.generate_response(
            prediction["action"], 
            prediction["counter_price"]
        )

        # create message
        message = {
            "role": self.role,
            "content": response
        }

        # update own state
        self.update_state(message)

        return message


def test_base_agent():
    """Test BaseAgent functionality."""
    baseline_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    agreemate_dir = os.path.dirname(baseline_dir)
    pretrained_dir = os.path.join(agreemate_dir, "models", "pretrained")
    test_lm = dspy.LM(
        model="meta-llama/Llama-3.1-8B-Instruct",
        api_base="http://localhost:11434", # local Llama
        api_key="", # local Llama
        cache_dir=pretrained_dir
    )

    agent = BaseAgent(
        strategy_name="cooperative",
        target_price=100.0,
        category="electronics",
        is_buyer=True,
        lm=test_lm
    )
    assert agent.role == "buyer"
    assert agent.strategy["name"] == "cooperative"

    # test state updates
    message = {
        "role": "seller",
        "content": "I can offer it for $150"
    }
    agent.update_state(message)
    assert agent.current_price == 150.0
    assert len(agent.conversation_history) == 1
    assert agent.num_turns == 1

    # test step
    response = agent.step()
    assert "role" in response
    assert "content" in response
    assert response["role"] == "buyer"

    print("✓ All base agent tests passed")
    return agent

if __name__ == "__main__":
    agent = test_base_agent()