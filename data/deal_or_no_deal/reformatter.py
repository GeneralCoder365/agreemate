# reformatter.py
import os, json, logging
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import defaultdict
from statistics import mean, median
import pandas as pd

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@dataclass
class DatasetStatistics:
    """Tracks comprehensive statistics about the dataset processing and content."""

    def __init__(self):
        self.record_counts = {
            "raw": 0,
            "processed": 0,
            "retention_rate": 0.0,
            "by_split": defaultdict(int)
        }

        self.dialogue_metrics = {
            "turns_per_dialogue": {
                "min": float('inf'),
                "max": 0,
                "mean": 0.0,
                "median": 0.0,
                "total_turns": 0,
                "all_counts": [] # store all counts for median calculation
            },
            "tokens_per_turn": {
                "min": float('inf'),
                "max": 0,
                "mean": 0.0,
                "total_tokens": 0,
                "total_turns": 0
            },
            "utterance_lengths": {
                "min": float('inf'),
                "max": 0,
                "mean": 0.0,
                "total_length": 0,
                "total_utterances": 0
            }
        }

        self.negotiation_metrics = {
            "successful_deals": 0,
            "failed_deals": 0,
            "success_rate": 0.0,
            "by_role": {
                "buyer": {
                    "turns": 0,
                    "avg_turn_length": 0.0,
                    "total_length": 0
                },
                "seller": {
                    "turns": 0,
                    "avg_turn_length": 0.0,
                    "total_length": 0
                }
            },
            "value_distribution": {
                "books": {"min": float('inf'), "max": 0, "mean": 0.0, "values": []},
                "hats": {"min": float('inf'), "max": 0, "mean": 0.0, "values": []},
                "balls": {"min": float('inf'), "max": 0, "mean": 0.0, "values": []}
            }
        }

    def update_from_dialogue(self, dialogue: Dict, split_name: str):
        """Update statistics based on a single dialogue."""
        self.record_counts["raw"] += 1
        self.record_counts["by_split"][split_name] += 1

        # dialogue-level metrics
        turns = dialogue["turns"]
        num_turns = len(turns)
        self.dialogue_metrics["turns_per_dialogue"]["all_counts"].append(num_turns)
        self.dialogue_metrics["turns_per_dialogue"]["min"] = min(
            self.dialogue_metrics["turns_per_dialogue"]["min"], 
            num_turns
        )
        self.dialogue_metrics["turns_per_dialogue"]["max"] = max(
            self.dialogue_metrics["turns_per_dialogue"]["max"], 
            num_turns
        )
        self.dialogue_metrics["turns_per_dialogue"]["total_turns"] += num_turns

        # process individual turns
        for turn in turns:
            text = turn["text"]
            tokens = text.split()

            # token metrics
            num_tokens = len(tokens)
            self.dialogue_metrics["tokens_per_turn"]["total_tokens"] += num_tokens
            self.dialogue_metrics["tokens_per_turn"]["total_turns"] += 1
            self.dialogue_metrics["tokens_per_turn"]["min"] = min(
                self.dialogue_metrics["tokens_per_turn"]["min"],
                num_tokens
            )
            self.dialogue_metrics["tokens_per_turn"]["max"] = max(
                self.dialogue_metrics["tokens_per_turn"]["max"],
                num_tokens
            )

            # length metrics
            length = len(text)
            self.dialogue_metrics["utterance_lengths"]["total_length"] += length
            self.dialogue_metrics["utterance_lengths"]["total_utterances"] += 1
            self.dialogue_metrics["utterance_lengths"]["min"] = min(
                self.dialogue_metrics["utterance_lengths"]["min"],
                length
            )
            self.dialogue_metrics["utterance_lengths"]["max"] = max(
                self.dialogue_metrics["utterance_lengths"]["max"],
                length
            )

            # role-specific metrics
            role = turn["speaker"]
            self.negotiation_metrics["by_role"][role]["turns"] += 1
            self.negotiation_metrics["by_role"][role]["total_length"] += length

        # negotiation outcome metrics
        if dialogue["outcome"]["success"]:
            self.negotiation_metrics["successful_deals"] += 1
        else:
            self.negotiation_metrics["failed_deals"] += 1

        # value distribution metrics
        input_values = dialogue["input_values"]
        for item_type in ["book", "hat", "ball"]:
            value = input_values[item_type]["value"]
            dist = self.negotiation_metrics["value_distribution"][item_type + "s"]
            dist["values"].append(value)
            dist["min"] = min(dist["min"], value)
            dist["max"] = max(dist["max"], value)

    def finalize(self):
        """Calculate final statistics and rates."""
        if self.record_counts["raw"] > 0:
            # record counts
            self.record_counts["retention_rate"] = (
                self.record_counts["processed"] / self.record_counts["raw"] * 100
            )

            # dialogue metrics
            turns_data = self.dialogue_metrics["turns_per_dialogue"]
            if turns_data["total_turns"] > 0:
                turns_data["mean"] = turns_data["total_turns"] / self.record_counts["raw"]
                turns_data["median"] = median(turns_data["all_counts"])
            del turns_data["all_counts"] # remove raw data before export

            tokens_data = self.dialogue_metrics["tokens_per_turn"]
            if tokens_data["total_turns"] > 0:
                tokens_data["mean"] = tokens_data["total_tokens"] / tokens_data["total_turns"]

            length_data = self.dialogue_metrics["utterance_lengths"]
            if length_data["total_utterances"] > 0:
                length_data["mean"] = length_data["total_length"] / length_data["total_utterances"]

            # negotiation metrics
            total_deals = self.negotiation_metrics["successful_deals"] + self.negotiation_metrics["failed_deals"]
            if total_deals > 0:
                self.negotiation_metrics["success_rate"] = (
                    self.negotiation_metrics["successful_deals"] / total_deals * 100
                )

            # role metrics
            for role in ["buyer", "seller"]:
                role_data = self.negotiation_metrics["by_role"][role]
                if role_data["turns"] > 0:
                    role_data["avg_turn_length"] = role_data["total_length"] / role_data["turns"]
                del role_data["total_length"] # clean up intermediate data

            # value distribution means
            for item_type in ["books", "hats", "balls"]:
                dist = self.negotiation_metrics["value_distribution"][item_type]
                if dist["values"]:
                    dist["mean"] = mean(dist["values"])
                del dist["values"] # clean up raw values

    def to_dict(self) -> Dict:
        """Convert statistics to dictionary format for JSON export."""
        return {
            "record_counts": self.record_counts,
            "dialogue_metrics": self.dialogue_metrics,
            "negotiation_metrics": self.negotiation_metrics
        }


class DialogueReformatter:
    """
    Transforms deal-or-no-deal dialogues into price-focused negotiation data 
    suitable for finetuning buyer/seller/generalist models.
    """

    def __init__(self, input_dir: str, output_dir: str):
        """Initialize reformatter with input and output directories."""
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.statistics = DatasetStatistics()

    def process_all_files(self):
        """Process all dataset files and create three training sets."""
        # load and combine all data
        all_dialogues = []
        for filename in ["train.txt", "test.txt", "validation.txt"]:
            file_path = self.input_dir / filename
            if file_path.exists():
                dialogues = self._load_file(file_path, split_name=filename.replace('.txt', ''))
                all_dialogues.extend(dialogues)
                logger.info(f"Loaded {len(dialogues)} dialogues from {filename}")

        # create specialized datasets
        buyer_data = self._create_role_dataset(all_dialogues, "buyer")
        seller_data = self._create_role_dataset(all_dialogues, "seller")
        generalist_data = self._create_generalist_dataset(all_dialogues)

        # save datasets
        self._save_dataset(buyer_data, "buyer_training.csv")
        self._save_dataset(seller_data, "seller_training.csv")
        self._save_dataset(generalist_data, "generalist_training.csv")

        # finalize and save statistics
        self.statistics.finalize()
        stats_path = self.output_dir / "dataset_info.json"
        with open(stats_path, 'w') as f:
            json.dump(self.statistics.to_dict(), f, indent=2)
        logger.info(f"Saved dataset statistics to {stats_path}")

    def _load_file(self, file_path: Path, split_name: str) -> List[Dict]:
        """Load and parse a dataset file into structured dialogues."""
        dialogues = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    dialogue = self._parse_dialogue(line.strip(), split_name)
                    if dialogue:
                        dialogues.append(dialogue)
                except Exception as e:
                    logger.warning(f"Error parsing dialogue: {str(e)}")
        return dialogues

    def _parse_dialogue(self, line: str, split_name: str) -> Optional[Dict]:
        """Parse a single dialogue line into structured format."""
        try:
            # extract main components using the dataset's XML-style tags
            parts = {}
            for tag in ['input', 'dialogue', 'output', 'partner_input']:
                start_tag = f"<{tag}>"
                end_tag = f"</{tag}>"
                start_idx = line.find(start_tag) + len(start_tag)
                end_idx = line.find(end_tag)
                if start_idx > -1 and end_idx > -1:
                    parts[tag] = line[start_idx:end_idx].strip()

            # parse values and create structured dialogue
            input_values = self._parse_values(parts['input'])
            partner_values = self._parse_values(parts['partner_input'])
            dialogue_turns = self._parse_turns(parts['dialogue'])
            outcome = self._parse_outcome(parts['output'])

            dialogue = {
                "input_values": input_values,
                "partner_values": partner_values,
                "turns": dialogue_turns,
                "outcome": outcome
            }

            # update statistics
            self.statistics.update_from_dialogue(dialogue, split_name)
            self.statistics.record_counts["processed"] += 1

            return dialogue

        except Exception as e:
            logger.warning(f"Failed to parse dialogue: {str(e)}")
            return None

    def _parse_values(self, value_str: str) -> Dict[str, Dict[str, int]]:
        """Parse item counts and values into a structured format."""
        values = [int(x) for x in value_str.split()]
        return {
            "book": {"count": values[0], "value": values[1]},
            "hat": {"count": values[2], "value": values[3]},
            "ball": {"count": values[4], "value": values[5]}
        }

    def _parse_turns(self, dialogue: str) -> List[Dict]:
        """Parse dialogue text into structured turns."""
        turns = []
        current_context = []

        for turn in dialogue.split('<eos>'):
            turn = turn.strip()
            if not turn:
                continue

            speaker = "buyer" if turn.startswith("YOU:") else "seller"
            text = turn.replace("YOU:", "").replace("THEM:", "").strip()

            turns.append({
                "speaker": speaker,
                "text": text,
                "context": current_context.copy()
            })
            current_context.append(text)

        return turns

    def _parse_outcome(self, output: str) -> Dict:
        """Parse dialogue outcome into structured format."""
        success = "<disagree>" not in output and "<no_agreement>" not in output
        return {
            "success": success,
            "distribution": output if success else None
        }

    def _create_role_dataset(self, dialogues: List[Dict], role: str) -> pd.DataFrame:
        """Create a specialized dataset for a specific role."""
        rows = []
        for dialogue in dialogues:
            turns = self._extract_role_turns(dialogue, role)
            rows.extend(turns)
        return pd.DataFrame(rows)

    def _create_generalist_dataset(self, dialogues: List[Dict], chunk_size: int = 10) -> pd.DataFrame:
        """
        Create a combined dataset for generalist training with alternating chunks of buyer/seller data.
        Alternate between buyer and seller examples to ensure balanced fine-tuning.

        Args:
            dialogues: List of dialogue dictionaries
            chunk_size: Number of examples per role before alternating
        """
        # get role-specific dataframes
        buyer_df = self._create_role_dataset(dialogues, "buyer")
        seller_df = self._create_role_dataset(dialogues, "seller")

        # calculate number of chunks needed
        n_chunks = min(len(buyer_df) // chunk_size, len(seller_df) // chunk_size)

        # create alternating chunks
        combined_chunks = []
        for i in range(n_chunks):
            buyer_start = i * chunk_size
            buyer_end = buyer_start + chunk_size
            seller_start = i * chunk_size
            seller_end = seller_start + chunk_size

            # add chunk of buyer examples
            buyer_chunk = buyer_df.iloc[buyer_start:buyer_end]
            combined_chunks.append(buyer_chunk)

            # add chunk of seller examples
            seller_chunk = seller_df.iloc[seller_start:seller_end]
            combined_chunks.append(seller_chunk)

        # add any remaining examples (to not lose data)
        remaining_buyer = buyer_df.iloc[n_chunks*chunk_size:]
        remaining_seller = seller_df.iloc[n_chunks*chunk_size:]
        if not remaining_buyer.empty:
            combined_chunks.append(remaining_buyer)
        if not remaining_seller.empty:
            combined_chunks.append(remaining_seller)

        # combine all chunks
        return pd.concat(combined_chunks, ignore_index=True)

    def _extract_role_turns(self, dialogue: Dict, role: str) -> List[Dict]:
        """Extract turns for a specific role with appropriate context."""
        rows = []
        for idx, turn in enumerate(dialogue["turns"]):
            if turn["speaker"] == role:
                row = {
                    "role": role,
                    "context": " ".join(turn["context"]),
                    "utterance": turn["text"],
                    "values": json.dumps(dialogue["input_values"]),
                    "partner_values": json.dumps(dialogue["partner_values"]),
                    "outcome": json.dumps(dialogue["outcome"]),
                }
                rows.append(row)
        return rows

    def _save_dataset(self, df: pd.DataFrame, filename: str):
        """Save processed dataset to CSV."""
        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} examples to {output_path}")


def main():
    """Main execution function."""
    data_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(data_dir, "raw")

    reformatter = DialogueReformatter(Path(raw_dir), Path(data_dir))
    reformatter.process_all_files()

if __name__ == "__main__":
    main()