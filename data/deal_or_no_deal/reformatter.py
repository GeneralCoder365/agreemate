# reformatter.py
import os, json, logging, re, random
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
from statistics import mean, median
import pandas as pd

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@dataclass
class DialogueAction:
    """
    Represents a structured negotiation action with associated metadata.
    
    Attributes:
        action_type (str): Core action (offer, counter, accept, reject)
        price (Optional[float]): Associated price value if present
        items_mentioned (Set[str]): Mentioned items in the action
        is_cooperative (bool): Whether the action is cooperative or aggressive
    """
    action_type: str  
    price: Optional[float] = None
    items_mentioned: Set[str] = field(default_factory=set)
    is_cooperative: bool = True

@dataclass
class DialogueTurn:
    """
    Represents a complete turn in the negotiation with strategic elements.
    
    Attributes:
        speaker (str): Role of the speaker (buyer/seller)
        text (str): Raw utterance text
        context (List[str]): Previous conversation history
        thought (str): Generated strategic reasoning
        action (DialogueAction): Structured action representation
        values (Dict): Speaker's item values
        partner_values (Dict): Partner's item values
        turn_number (int): Position in dialogue
        turns_remaining (int): Remaining turns in dialogue
    """
    speaker: str
    text: str
    context: List[str]
    thought: str
    action: DialogueAction
    values: Dict
    partner_values: Dict
    turn_number: int
    turns_remaining: int

@dataclass
class NegotiationMetrics:
    """
    Tracks negotiation-specific metrics for a role or overall dataset.
    
    Attributes:
        turns (int): Total number of turns
        total_length (int): Total text length
        successful_deals (int): Number of successful negotiations
        failed_deals (int): Number of failed negotiations
        strategy_metrics (Dict): Metrics about strategic choices
        price_metrics (Dict): Statistics about price negotiations
        cooperative_metrics (Dict): Metrics about cooperation levels
    """
    turns: int = 0
    total_length: int = 0
    successful_deals: int = 0
    failed_deals: int = 0
    strategy_metrics: Dict = field(default_factory=lambda: {
        "thought_quality": 0.0,
        "strategic_adherence": 0.0,
        "price_targeting": 0.0
    })
    price_metrics: Dict = field(default_factory=lambda: {
        "min": float('inf'),
        "max": 0.0,
        "mean": 0.0,
        "values": []
    })
    cooperative_metrics: Dict = field(default_factory=lambda: {
        "cooperative_actions": 0,
        "aggressive_actions": 0,
        "cooperation_ratio": 0.0
    })

    def update(self, turn: DialogueTurn, success: bool = True):
        """Update metrics based on a dialogue turn."""
        self.turns += 1
        self.total_length += len(turn.text)

        if success:
            self.successful_deals += 1
        else:
            self.failed_deals += 1

        # update price metrics if price present
        if turn.action.price is not None:
            self.price_metrics["values"].append(turn.action.price)
            self.price_metrics["min"] = min(self.price_metrics["min"], turn.action.price)
            self.price_metrics["max"] = max(self.price_metrics["max"], turn.action.price)

        # update cooperation metrics
        if turn.action.is_cooperative:
            self.cooperative_metrics["cooperative_actions"] += 1
        else:
            self.cooperative_metrics["aggressive_actions"] += 1

    def finalize(self):
        """Calculate final derived metrics."""
        # calculate price statistics
        if self.price_metrics["values"]:
            self.price_metrics["mean"] = mean(self.price_metrics["values"])
        del self.price_metrics["values"]  # Clean up raw values

        # calculate cooperation ratio
        total_actions = (self.cooperative_metrics["cooperative_actions"] + 
                        self.cooperative_metrics["aggressive_actions"])
        if total_actions > 0:
            self.cooperative_metrics["cooperation_ratio"] = (
                self.cooperative_metrics["cooperative_actions"] / total_actions
            )

@dataclass
class DatasetStatistics:
    """
    Comprehensive statistics tracking for dataset processing.
    
    Attributes:
        record_counts (Dict): Basic dataset size metrics
        by_role (Dict[str, NegotiationMetrics]): Role-specific metrics
        dialogue_metrics (Dict): Dialogue-level statistics
        item_values (Dict): Statistics about item values
    """
    record_counts: Dict = field(default_factory=lambda: {
        "raw": 0,
        "processed": 0,
        "by_split": defaultdict(int)
    })
    by_role: Dict[str, NegotiationMetrics] = field(default_factory=lambda: {
        "buyer": NegotiationMetrics(),
        "seller": NegotiationMetrics()
    })
    dialogue_metrics: Dict = field(default_factory=lambda: {
        "turns_per_dialogue": {
            "min": float('inf'),
            "max": 0,
            "mean": 0.0,
            "median": 0.0,
            "values": []
        },
        "thought_metrics": {
            "min_length": float('inf'),
            "max_length": 0,
            "mean_length": 0.0
        }
    })
    item_values: Dict = field(default_factory=lambda: {
        item: {"min": float('inf'), "max": 0, "mean": 0.0, "values": []}
        for item in ["books", "hats", "balls"]
    })

    def update_from_dialogue(self, dialogue: Dict):
        """Update statistics with data from a complete dialogue."""
        self.record_counts["raw"] += 1

        # update turns per dialogue
        num_turns = len(dialogue["turns"])
        self.dialogue_metrics["turns_per_dialogue"]["values"].append(num_turns)
        self.dialogue_metrics["turns_per_dialogue"]["min"] = min(
            self.dialogue_metrics["turns_per_dialogue"]["min"],
            num_turns
        )
        self.dialogue_metrics["turns_per_dialogue"]["max"] = max(
            self.dialogue_metrics["turns_per_dialogue"]["max"],
            num_turns
        )

        # update item value distributions
        input_values = dialogue["input_values"]
        for item_type in ["book", "hat", "ball"]:
            value = input_values[item_type]["value"]
            self.item_values[item_type + "s"]["values"].append(value)
            self.item_values[item_type + "s"]["min"] = min(
                self.item_values[item_type + "s"]["min"], 
                value
            )
            self.item_values[item_type + "s"]["max"] = max(
                self.item_values[item_type + "s"]["max"],
                value
            )

    def update_from_turn(self, turn: DialogueTurn, success: bool):
        """Update statistics with data from a single turn."""
        self.by_role[turn.speaker].update(turn, success)

        # update thought length metrics
        thought_length = len(turn.thought.split())
        self.dialogue_metrics["thought_metrics"]["min_length"] = min(
            self.dialogue_metrics["thought_metrics"]["min_length"],
            thought_length
        )
        self.dialogue_metrics["thought_metrics"]["max_length"] = max(
            self.dialogue_metrics["thought_metrics"]["max_length"],
            thought_length
        )

    def finalize(self):
        """Calculate final statistics and clean up intermediate data."""
        # dialogue metrics
        turns_data = self.dialogue_metrics["turns_per_dialogue"]
        if turns_data["values"]:
            turns_data["mean"] = mean(turns_data["values"])
            turns_data["median"] = median(turns_data["values"])
        del turns_data["values"]

        # thought metrics
        thought_metrics = self.dialogue_metrics["thought_metrics"]
        if thought_metrics["max_length"] > 0:
            thought_metrics["mean_length"] = (
                thought_metrics["min_length"] + thought_metrics["max_length"]
            ) / 2

        # item value metrics
        for item_metrics in self.item_values.values():
            if item_metrics["values"]:
                item_metrics["mean"] = mean(item_metrics["values"])
            del item_metrics["values"]

        # role-specific metrics
        for metrics in self.by_role.values():
            metrics.finalize()

    def to_dict(self) -> Dict:
        """Convert statistics to JSON-serializable dictionary."""
        return {
            "record_counts": self.record_counts,
            "dialogue_metrics": self.dialogue_metrics,
            "by_role": {role: vars(metrics) for role, metrics in self.by_role.items()},
            "item_values": self.item_values
        }


class DialogueReformatter:
    """
    Transforms deal-or-no-deal dialogues into strategically-enhanced training data
    suitable for fine-tuning negotiation models.
    """

    def __init__(self, input_dir: str, output_dir: str):
        """
        Initialize reformatter with input and output paths.
        Sets up statistics tracking and strategic templates.

        Args:
            input_dir (str): Path to raw data directory
            output_dir (str): Path for processed output
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.statistics = DatasetStatistics()
        self._init_templates()

    def _init_templates(self):
        """Initialize strategic reasoning templates."""
        self.buyer_templates = {
            'initial': [
                "Given my target price of {target}, starting at {price} leaves room for negotiation",
                "Items are worth {value} to me, should start below {target} to allow compromise"
            ],
            'response': [
                "Counter offer of {price} would still achieve {value}% of target value",
                "Partner's offer of {price} is {evaluation}, could suggest {counter}"
            ],
            'final': [
                "Final price of {price} achieves {value}% of target value",
                "Deal at {price} is {evaluation} given my target of {target}"
            ]
        }

        self.seller_templates = {
            'initial': [
                "Starting at {price} given items worth {value} and target of {target}",
                "Opening with {price} maintains room above minimum of {min_price}"
            ],
            'response': [
                "Current offer {price} maintains {value}% of target value",
                "Can emphasize {features} to justify maintaining price above {min_price}"
            ],
            'final': [
                "Accepting {price} provides {value}% of target value",
                "Final price of {price} is {evaluation} given costs of {value}"
            ]
        }


    def _load_file(self, file_path: str) -> List[Dict]:
        """
        Load and parse dialogues from a file.

        Args:
            file_path: Path to dialogue file

        Returns:
            List of parsed dialogue dictionaries
        """
        dialogues = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    dialogue = self._parse_dialogue(line.strip())
                    if dialogue:
                        dialogues.append(dialogue)
                except Exception as e:
                    logger.warning(f"Error parsing dialogue: {str(e)}")
        return dialogues

    def process_all_files(self):
        """Process all dataset files and create training sets."""
        all_dialogues = []
        for filename in ["train.txt", "test.txt", "validation.txt"]:
            file_path = os.path.join(self.input_dir, filename)
            if os.path.exists(file_path):
                dialogues = self._load_file(file_path)
                all_dialogues.extend(dialogues)
                logger.info(f"Loaded {len(dialogues)} dialogues from {filename}")

        # create role-specific and generalist datasets
        for role in ['buyer', 'seller']:
            df = self._create_role_dataset(all_dialogues, role)
            self._save_dataset(df, f"{role}_training.csv")

        generalist_df = self._create_generalist_dataset(all_dialogues)
        self._save_dataset(generalist_df, "generalist_training.csv")

        # save statistics
        self.statistics.finalize()
        stats_path = os.path.join(self.output_dir, "dataset_info.json")
        with open(stats_path, 'w') as f:
            json.dump(self.statistics.to_dict(), f, indent=2)

        logger.info(f"Processed {len(all_dialogues)} dialogues into training datasets")


    def _parse_dialogue(self, line: str) -> Optional[Dict]:
        """Parse a single dialogue into structured format."""
        # extract main components using XML-style tags
        components = {}
        for tag in ['input', 'dialogue', 'output', 'partner_input']:
            start_tag = f"<{tag}>"
            end_tag = f"</{tag}>"
            start_idx = line.find(start_tag)
            if start_idx == -1: # skip if tag not found
                continue
            start_idx += len(start_tag)
            end_idx = line.find(end_tag, start_idx)
            if end_idx == -1: # skip if closing tag not found
                continue
            components[tag] = line[start_idx:end_idx].strip()

        if not all(tag in components for tag in ['input', 'dialogue', 'output']):
            return None

        # parse into structured format
        dialogue = {
            "input_values": self._parse_values(components['input']),
            "partner_values": self._parse_values(components['partner_input']) if 'partner_input' in components else {},
            "turns": self._parse_turns(components['dialogue']),
            "outcome": self._parse_outcome(components['output'])
        }

        self.statistics.update_from_dialogue(dialogue)
        return dialogue

    def _parse_values(self, value_str: str) -> Dict[str, Dict[str, int]]:
        """Parse item counts and values into structured format."""
        values = [int(x) for x in value_str.split()]
        if len(values) != 6:
            logger.warning(f"Expected 6 values, got {len(values)}: {value_str}")
            return {}
        return {
            "book": {"count": values[0], "value": values[1]},
            "hat": {"count": values[2], "value": values[3]},
            "ball": {"count": values[4], "value": values[5]}
        }

    def _parse_turns(self, dialogue: str) -> List[Dict]:
        """Parse dialogue text into structured turns."""
        turns = []
        context = []

        for turn in dialogue.split('<eos>'):
            turn = turn.strip()
            if not turn:
                continue

            speaker = "buyer" if turn.startswith("YOU:") else "seller"
            text = turn.replace("YOU:", "").replace("THEM:", "").strip()

            turns.append({
                "speaker": speaker,
                "text": text,
                "context": context.copy()
            })
            context.append(text)

        return turns

    def _parse_outcome(self, output: str) -> Dict:
        """
        Parse dialogue outcome into structured format.

        Args:
            output (str): Raw outcome string

        Returns:
            Dict containing success status and distribution
        """
        success = "<disagree>" not in output and "<no_agreement>" not in output
        return {
            "success": success,
            "distribution": output if success else None
        }

    def _extract_strategic_thought(self, turn_data: Dict, role: str, features: Set[str]) -> str:
        """
        Generate strategic reasoning based on dialogue context and values.

        Args:
            turn_data: Turn information including context and values
            role: Agent role (buyer/seller)
            features: Set of features mentioned in the turn

        Returns:
            Generated strategic thought
        """
        # calculate key metrics
        total_value = sum(item['value'] * item['count'] for item in turn_data['values'].values())
        target_price = (total_value * 0.8 if role == 'buyer' else total_value * 1.2)

        # select appropriate template based on context
        templates = self.buyer_templates if role == 'buyer' else self.seller_templates

        if not turn_data['context']: # initial offer
            selected_template = random.choice(templates['initial'])
            metrics = {
                'target': round(target_price, 2),
                'value': round(total_value, 2),
                'price': round(self._suggest_initial_price(role, total_value), 2)
            }

        else: # response or final offer
            selected_template = random.choice(templates['response'])
            prev_price = self._extract_last_price(turn_data['context'])
            metrics = {
                'price': round(prev_price, 2) if prev_price else None,
                'value': round((prev_price / target_price * 100), 2) if prev_price else 0,
                'evaluation': self._evaluate_price(prev_price, target_price),
                'counter': round(self._suggest_counter_price(prev_price, target_price, role), 2) if prev_price else round(self._suggest_counter_price(None, target_price, role), 2)
            }

        # add 'features' to metrics if the template requires it
        if '{features}' in selected_template:
            metrics['features'] = ', '.join(features) if features else 'key items'

        # add 'min_price' to metrics if the template requires it
        if '{min_price}' in selected_template: # define 'min_price' as 80% of 'target_price' for sellers
            metrics['min_price'] = round(target_price * 0.8, 2) if role == 'seller' else None

        return selected_template.format(**metrics)


    def _create_role_dataset(self, dialogues: List[Dict], role: str) -> pd.DataFrame:
        """
        Create specialized dataset for a specific role with enhanced strategic elements.

        Args:
            dialogues: List of dialogue dictionaries
            role: Target role (buyer/seller)

        Returns:
            DataFrame containing processed role-specific examples
        """
        processed_turns = []
        for dialogue in dialogues:
            dialogue_turns = []
            for turn in dialogue['turns']:
                if turn['speaker'] == role:
                    # create turn context
                    turn_data = {
                        'values': dialogue['input_values'],
                        'partner_values': dialogue['partner_values'],
                        'context': turn['context'],
                        'text': turn['text']
                    }

                    # extract action and price
                    action = self._extract_action(turn_data, role)
                    price = action.price

                    # generate strategic thought
                    thought = self._extract_strategic_thought(turn_data, role, action.items_mentioned)

                    # create processed turn
                    processed_turn = {
                        'role': role,
                        'context': ' '.join(turn['context']),
                        'thought': thought,
                        'action': action.action_type,
                        'utterance': turn['text'],
                        'price': price,
                        'values': json.dumps(dialogue['input_values']),
                        'partner_values': json.dumps(dialogue.get('partner_values', {})),
                        'outcome': json.dumps(dialogue['outcome'])
                    }

                    # update statistics
                    self.statistics.update_from_turn(
                        DialogueTurn(
                            speaker=role,
                            text=turn['text'],
                            context=turn['context'],
                            thought=thought,
                            action=action,
                            values=dialogue['input_values'],
                            partner_values=dialogue.get('partner_values', {}),
                            turn_number=len(dialogue_turns),
                            turns_remaining=len(dialogue['turns']) - len(dialogue_turns)
                        ),
                        dialogue['outcome']['success']
                    )

                    dialogue_turns.append(processed_turn)

            processed_turns.extend(dialogue_turns)

        return pd.DataFrame(processed_turns)

    def _create_generalist_dataset(self, dialogues: List[Dict], 
                                chunk_size: int = 10) -> pd.DataFrame:
        """
        Create combined dataset with balanced buyer/seller examples.

        Args:
            dialogues: List of dialogue dictionaries
            chunk_size: Number of consecutive examples per role

        Returns:
            DataFrame containing alternating chunks of buyer/seller examples
        """
        buyer_df = self._create_role_dataset(dialogues, "buyer")
        seller_df = self._create_role_dataset(dialogues, "seller")

        n_chunks = min(len(buyer_df) // chunk_size, 
                    len(seller_df) // chunk_size)

        combined_chunks = []
        for i in range(n_chunks):
            start = i * chunk_size
            end = start + chunk_size

            # add buyer chunk
            combined_chunks.append(buyer_df.iloc[start:end])

            # add seller chunk
            combined_chunks.append(seller_df.iloc[start:end])

        # add any remaining examples
        if len(buyer_df) > n_chunks * chunk_size:
            combined_chunks.append(buyer_df.iloc[n_chunks*chunk_size:])
        if len(seller_df) > n_chunks * chunk_size:
            combined_chunks.append(seller_df.iloc[n_chunks*chunk_size:])

        return pd.concat(combined_chunks, ignore_index=True)


    def _extract_action(self, turn_data: Dict, role: str) -> DialogueAction:
        """
        Extract structured action from turn data.

        Args:
            turn_data: Turn information including text and context
            role: Agent role (buyer/seller)

        Returns:
            DialogueAction containing action type and metadata
        """
        text = turn_data['text'].lower()

        # check for explicit accept/reject
        if any(word in text for word in ['accept', 'deal', 'agree']):
            return DialogueAction('accept', is_cooperative=True)
        if any(word in text for word in ['reject', 'no deal', 'cannot']):
            return DialogueAction('reject', is_cooperative=False)

        # extract price if present
        price = self._extract_price(turn_data)
        if price:
            # determine if counter or initial offer
            action_type = 'counter' if turn_data['context'] else 'offer'

            # analyze cooperativeness
            target_price = self._compute_target_price(turn_data, role)
            is_cooperative = self._is_price_cooperative(price, target_price)

            return DialogueAction(
                action_type=action_type,
                price=price,
                items_mentioned=self._extract_mentioned_items(text),
                is_cooperative=is_cooperative
            )

        # default to counter without explicit price
        return DialogueAction('counter', is_cooperative=True)

    def _extract_price(self, turn_data: Dict) -> Optional[float]:
        """
        Extract price value from turn text.

        Args:
            turn_data: Turn information including text

        Returns:
            Extracted price if present, None otherwise
        """
        text = turn_data['text']

        # try to extract explicit price
        if price_match := re.search(r'\$?(\d+(?:\.\d{2})?)', text):
            return float(price_match.group(1))

        # try to infer price from mentioned items
        mentioned_items = self._extract_mentioned_items(text)
        if mentioned_items:
            total_price = 0.0
            for item in mentioned_items: # ensure all mentioned items have corresponding values
                item_data = turn_data['values'].get(item)
                if item_data and 'value' in item_data and 'count' in item_data:
                    total_price += item_data['value'] * item_data['count']
                else:
                    logger.warning(f"Missing value or count for item '{item}' in turn data.")
            return total_price

        return None

    def _extract_mentioned_items(self, text: str) -> Set[str]:
        """Extract mentioned item types from text."""
        items = {'book', 'hat', 'ball'}
        return {item for item in items 
                if item in text.lower() or item + 's' in text.lower()}

    def _compute_target_price(self, turn_data: Dict, role: str) -> float:
        """Compute target price based on item values."""
        total_value = sum(
            item['value'] * item['count'] 
            for item in turn_data['values'].values()
        )
        if role == 'buyer':
            return total_value * 0.8 # buyers target 80% of total value
        else:
            return total_value * 1.2 # sellers target 120% of total value

    def _is_price_cooperative(self, price: float, target: float) -> bool:
        """Determine if price represents cooperative behavior."""
        return abs(price - target) / target <= 0.2 # within 20% of target

    def _suggest_initial_price(self, role: str, total_value: float) -> float:
        """Suggest initial price based on role and total value."""
        return (total_value * 0.7 if role == 'buyer' 
                else total_value * 1.3)

    def _suggest_counter_price(self, prev_price: Optional[float], target: float, 
                            role: str) -> float:
        """Suggest counter price based on previous price and target."""
        if prev_price is None: # define default counter price based on role
            if role == 'buyer':
                return target * 0.9 # slightly below target
            else:
                return target * 1.1 # slightly above target

        if role == 'buyer': # adjust counter price based on previous offer
            return min(prev_price * 1.1, target)

        return max(prev_price * 0.9, target) # seller: adjust counter price based on previous offer

    def _extract_last_price(self, context: List[str]) -> Optional[float]:
        """Extract most recent price from conversation context."""
        for turn in reversed(context):
            if price_match := re.search(r'\$?(\d+(?:\.\d{2})?)', turn):
                return float(price_match.group(1))
        return None

    def _evaluate_price(self, price: Optional[float], target: float) -> str:
        """Evaluate price relative to target."""
        if price is None:
            return "unknown"
        ratio = price / target
        if ratio > 1.2:
            return "significantly above target"
        if ratio > 1.05:
            return "slightly above target"
        if ratio > 0.95:
            return "close to target"
        if ratio > 0.8:
            return "slightly below target"
        return "significantly below target"


    def _save_dataset(self, df: pd.DataFrame, filename: str):
        """
        Save processed dataset to CSV file.
        
        Args:
            df: Processed DataFrame
            filename: Output filename
        """
        output_path = os.path.join(self.output_dir, filename)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} examples to {output_path}")


if __name__ == "__main__":
    data_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(data_dir, "raw")

    reformatter = DialogueReformatter(raw_dir, data_dir)
    reformatter.process_all_files()