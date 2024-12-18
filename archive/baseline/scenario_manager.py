# scenario_manager.py
"""
Manages negotiation scenarios from the Craigslist Bargains dataset for the AgreeMate system.

This module handles loading and preparation of negotiation scenarios from our CSV files,
where each scenario contains real buyer-seller negotiation data including:
- Item details (title, description, category)
- Price information (list price, buyer target, seller target)
- Quality metrics (completeness, confidence)

The manager ensures scenarios are properly loaded and validated for use in
negotiation experiments, maintaining the integrity of the original dataset
while making it easily accessible for experimentation.
"""
import logging
from typing import Dict, List, Optional
import pandas as pd

from utils.data_loader import DataLoader

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class NegotiationScenario:
    """
    Represents a complete negotiation scenario from the dataset.

    Contains all necessary information for both buyer and seller agents
    to conduct a negotiation, using the real scenario data from our CSVs.
    """

    def __init__(self, row: pd.Series):
        """
        Initialize scenario from dataset row.

        Args:
            row: pandas Series containing scenario data from CSV
        """
        # core identification
        self.scenario_id = row['scenario_id']
        self.split_type = row['split_type']
        self.category = row['category']

        # item information
        self.title = row['title']
        self.description = row['description']

        # price information
        self.list_price = float(row['list_price'])
        self.buyer_target = float(row['buyer_target'])
        self.seller_target = float(row['seller_target'])

        # quality metrics
        self.price_delta_pct = float(row['price_delta_pct'])
        self.relative_price = float(row['relative_price'])
        self.data_completeness = float(row['data_completeness'])
        self.price_confidence = bool(row['price_confidence'])
        self.has_images = bool(row['has_images'])

        # validate price relationships
        if not (self.buyer_target <= self.list_price and 
                self.seller_target <= self.list_price):
            raise ValueError(f"Invalid price relationships in scenario {self.scenario_id}")

    def get_buyer_context(self) -> Dict:
        """Get scenario context from buyer's perspective."""
        return {
            'role': 'buyer',
            'scenario_id': self.scenario_id,
            'category': self.category,
            'item': {
                'title': self.title,
                'description': self.description,
                'list_price': self.list_price
            },
            'target_price': self.buyer_target
        }

    def get_seller_context(self) -> Dict:
        """Get scenario context from seller's perspective."""
        return {
            'role': 'seller',
            'scenario_id': self.scenario_id,
            'category': self.category,
            'item': {
                'title': self.title,
                'description': self.description,
                'list_price': self.list_price
            },
            'target_price': self.seller_target
        }


class ScenarioManager:
    """
    Manages loading and selection of negotiation scenarios from the dataset.
    """

    def __init__(self, data_loader: DataLoader):
        """
        Initialize scenario manager.

        Args:
            data_loader: DataLoader instance for accessing CSVs
        """
        self.data_loader = data_loader

        # load all splits
        self.train_df, self.test_df, self.val_df = self.data_loader.load_splits()
        logger.info(f"Loaded {len(self.train_df)} training, {len(self.test_df)} test, "
                   f"and {len(self.val_df)} validation scenarios")

        # track created scenarios
        self.scenarios: Dict[str, NegotiationScenario] = {}

    def create_evaluation_batch(
        self,
        split: str = 'test',
        size: Optional[int] = None,
        balanced_categories: bool = True,
        category: Optional[str] = None
    ) -> List[NegotiationScenario]:
        """
        Create a batch of scenarios for evaluation.

        Args:
            split: Which dataset split to use ('train', 'test', 'val')
            size: Number of scenarios to return (default: all in split)
            balanced_categories: Whether to ensure category balance
            category: Optional specific category to use

        Returns:
            List of NegotiationScenario instances
        """
        # get appropriate dataframe
        if split == 'train':
            df = self.train_df
        elif split == 'test':
            df = self.test_df
        elif split == 'val':
            df = self.val_df
        else:
            raise ValueError(f"Unknown split: {split}")

        # filter by category if specified
        if category:
            df = df[df['category'] == category]
            if len(df) == 0:
                raise ValueError(f"No scenarios found for category: {category}")

        # handle balanced selection
        if balanced_categories and not category:
            scenarios = []
            categories = df['category'].unique()

            if size:
                per_category = size // len(categories)
                remainder = size % len(categories)
            else:
                # use all scenarios while maintaining balance
                min_per_cat = df['category'].value_counts().min()
                per_category = min_per_cat
                remainder = 0

            for cat in categories:
                cat_df = df[df['category'] == cat]
                count = per_category + (1 if remainder > 0 else 0)
                remainder -= 1

                # sample scenarios
                cat_scenarios = cat_df.sample(n=min(count, len(cat_df)))
                for _, row in cat_scenarios.iterrows():
                    scenario = NegotiationScenario(row)
                    self.scenarios[scenario.scenario_id] = scenario
                    scenarios.append(scenario)

            return scenarios

        else:
            # simple random sampling
            if size:
                df = df.sample(n=min(size, len(df)))

            scenarios = []
            for _, row in df.iterrows():
                scenario = NegotiationScenario(row)
                self.scenarios[scenario.scenario_id] = scenario
                scenarios.append(scenario)

            return scenarios

    def get_scenario(self, scenario_id: str) -> NegotiationScenario:
        """
        Get specific scenario by ID.

        Args:
            scenario_id: Scenario identifier

        Returns:
            NegotiationScenario instance
        """
        # return cached scenario if available
        if scenario_id in self.scenarios:
            return self.scenarios[scenario_id]

        # find scenario in dataframes
        for df in [self.train_df, self.test_df, self.val_df]:
            scenario_df = df[df['scenario_id'] == scenario_id]
            if len(scenario_df) == 1:
                scenario = NegotiationScenario(scenario_df.iloc[0])
                self.scenarios[scenario_id] = scenario
                return scenario

        raise ValueError(f"Scenario not found: {scenario_id}")


def test_scenario_manager():
    """Test ScenarioManager functionality."""
    # initialize with data loader
    data_loader = DataLoader()
    manager = ScenarioManager(data_loader)

    # test basic scenario creation
    test_batch = manager.create_evaluation_batch(
        split='test',
        size=10,
        balanced_categories=True
    )
    assert len(test_batch) == 10

    # verify category balance
    categories = set(s.category for s in test_batch)
    assert len(categories) > 1

    # test individual scenario loading
    scenario = manager.get_scenario(test_batch[0].scenario_id)
    assert scenario.scenario_id == test_batch[0].scenario_id
    assert scenario.buyer_target <= scenario.list_price
    assert scenario.seller_target <= scenario.list_price

    # test context generation
    buyer_context = scenario.get_buyer_context()
    seller_context = scenario.get_seller_context()
    assert buyer_context['role'] == 'buyer'
    assert seller_context['role'] == 'seller'
    assert buyer_context['target_price'] == scenario.buyer_target
    assert seller_context['target_price'] == scenario.seller_target

    print("✓ All scenario manager tests passed")
    return manager

if __name__ == "__main__":
    manager = test_scenario_manager()