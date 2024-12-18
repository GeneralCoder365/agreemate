# data_loader.py
import os, logging
from typing import Dict
import pandas as pd

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class NegotiationDialogueDataLoader:
    """
    Handles loading of negotiation dialogue datasets for finetuning.
    Provides easy access to buyer/seller/generalist training data.
    """

    def __init__(self):
        """Initialize paths to data files."""
        # setup paths
        self.finetuning_dir = os.path.dirname(os.path.abspath(__file__))
        self.agreemate_dir = os.path.dirname(self.finetuning_dir)
        self.data_dir = os.path.join(self.agreemate_dir, "data", "deal_or_no_deal")

        # dataset paths
        self.paths = {
            "buyer": os.path.join(self.data_dir, "buyer_training.csv"),
            "seller": os.path.join(self.data_dir, "seller_training.csv"),
            "generalist": os.path.join(self.data_dir, "generalist_training.csv")
        }

        # verify paths exist
        self._verify_data_paths()

        # cache for loaded datasets
        self._cache = {}

    def _verify_data_paths(self):
        """Verify all required data files exist."""
        for name, path in self.paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Required data file not found: {path}\n"
                    "Please run reformatter.py in deal_or_no_deal directory first."
                )

    def load_dataset(self, role: str) -> pd.DataFrame:
        """
        Load a specific dataset by role.

        Args:
            role: One of 'buyer', 'seller', or 'generalist'

        Returns:
            DataFrame containing the full dataset
        """
        if role not in self.paths:
            raise ValueError(f"Invalid role: {role}. Must be one of {list(self.paths.keys())}")

        # return cached if available
        if role in self._cache:
            logger.info(f"Using cached {role} dataset")
            return self._cache[role]

        # load from file
        logger.info(f"Loading {role} dataset from {self.paths[role]}")
        df = pd.read_csv(self.paths[role])

        # cache for future use
        self._cache[role] = df

        logger.info(f"Loaded {len(df)} examples")
        return df

    def prepare_for_training(self, df: pd.DataFrame) -> Dict[str, list]:
        """
        Prepare dataset for training by formatting with appropriate prompts.

        Args:
            df: DataFrame to prepare.

        Returns:
            Dict with 'input' and 'target' lists ready for tokenization.
        """
        prompts = []
        responses = []
        for _, row in df.iterrows():
            # validate required fields
            if pd.isna(row['role']) or pd.isna(row['values']) or pd.isna(row['partner_values']):
                logger.warning(f"Skipping row due to missing essential fields: {row}")
                continue

            # build the input structure
            prompt_data = {
                "values": row['values'],
                "partner_values": row['partner_values'],
                "context": row['context'].strip() if pd.notna(row['context']) else "",
                "role": row['role'],
                "thought": row['thought'] if pd.notna(row['thought']) else "",
                "utterance": row['utterance']
            }

            # add generated prompts and corresponding responses
            prompts.append(prompt_data)
            responses.append(row['utterance'])

        return {
            "input": prompts,
            "target": responses
        }


if __name__ == "__main__":
    loader = NegotiationDialogueDataLoader()

    # test loading each role
    for role in ['buyer', 'seller', 'generalist']:
        df = loader.load_dataset(role)
        print(f"\n{role.title()} Dataset:")
        print(f"Total examples: {len(df)}")
        print("\nColumns:", df.columns.tolist())
        print("\nSample row:")
        print(df.iloc[0])

        # test training preparation
        prepared = loader.prepare_for_training(df.head())
        print("\nSample prepared example:")
        print("Input:", prepared['input'][0][:100], "...")
        print("Target:", prepared['target'][0])