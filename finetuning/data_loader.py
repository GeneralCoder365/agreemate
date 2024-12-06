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

            # build role-specific system instructions
            if row['role'] == "buyer":
                role_instructions = (
                    "Your goal is to negotiate effectively. "
                    "As a buyer, aim to reach a deal as close as possible to maximizing your benefit.\n"
                )
            elif row['role'] == "seller":
                role_instructions = (
                    "Your goal is to negotiate effectively. "
                    "As a seller, aim to achieve a price above the average valuation while closing the deal.\n"
                )
            else:
                role_instructions = (
                    "Your goal is to negotiate effectively as either a buyer or a seller. "
                    "Adapt your strategy to the situation.\n"
                )

            # incorporate additional scenario details if available
            system_prompt = (
                f"You are a {row['role']} negotiating over items.\n"
                f"Item Description: <INJECTED DURING INFERENCE>.\n"
                f"Your Values: {row['values']}\n"
                f"Partner's Values: {row['partner_values']}\n"
            )
            if 'price' in row and pd.notna(row['price']):
                system_prompt += f"Offered Price: ${row['price']}\n"

            # combine system instructions and conversation history
            context = row['context'].strip() if pd.notna(row['context']) else ""
            if context:
                prompt = (
                    f"{role_instructions}\n"
                    f"{system_prompt}\n"
                    f"Previous Conversation:\n{context}\n"
                    f"Your Response:"
                )
            else:
                prompt = (
                    f"{role_instructions}\n"
                    f"{system_prompt}\n"
                    f"Your First Message:"
                )

            # add generated prompts and corresponding responses
            prompts.append(prompt)
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