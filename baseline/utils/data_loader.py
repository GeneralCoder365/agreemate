# data_loader.py
import os, warnings
from typing import Dict, List, Tuple
import pandas as pd


class DataLoader:
    """
    Data loading and processing utilities for the AgreeMate baseline system.
    Handles loading and preprocessing of negotiation datasets.
    """
    def __init__(self):
        """Initialize data loader with path to dataset directory."""
        baseline_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        agreemate_dir = os.path.dirname(baseline_dir)
        self.data_dir = os.path.join(agreemate_dir, "data", "craigslist_bargains")
        self.train_data = None
        self.test_data = None
        self.val_data = None
        self.metadata = None

        # load dataset info
        self._load_dataset_info()

    def _load_dataset_info(self):
        """Load dataset metadata from dataset_info.json."""
        try:
            info_path = os.path.join(self.data_dir, "dataset_info.json")
            self.metadata = pd.read_json(info_path)
        except Exception as e:
            print(f"Warning: Could not load dataset info: {str(e)}")
            self.metadata = None

    def load_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load all data splits (train, test, validation).

        Returns:
            Tuple of (train_df, test_df, val_df)
        """
        self.train_data = pd.read_csv(os.path.join(self.data_dir, "train.csv"))
        self.test_data = pd.read_csv(os.path.join(self.data_dir, "test.csv"))
        self.val_data = pd.read_csv(os.path.join(self.data_dir, "validation.csv"))

        return self.train_data, self.test_data, self.val_data

    def get_category_stats(self) -> Dict:
        """Get category distribution and price statistics."""
        if self.train_data is None:
            self.load_splits()

        stats = {}
        for category in self.train_data['category'].unique():
            cat_data = self.train_data[self.train_data['category'] == category]
            stats[category] = {
                'count': len(cat_data),
                'price_stats': {
                    'min': cat_data['list_price'].min(),
                    'max': cat_data['list_price'].max(),
                    'mean': cat_data['list_price'].mean(),
                    'median': cat_data['list_price'].median()
                },
                'avg_price_delta': cat_data['price_delta_pct'].mean()
            }
        return stats

    def create_negotiation_pair(
        self, 
        row: pd.Series
    ) -> Dict[str, Dict]:
        """
        Create buyer and seller info for a negotiation scenario.

        Args:
            row: DataFrame row containing scenario data

        Returns:
            Dict containing buyer and seller information
        """
        return {
            'scenario_id': row['scenario_id'],
            'category': row['category'],
            'item': {
                'title': row['title'],
                'description': row['description'],
                'list_price': row['list_price']
            },
            'buyer': {
                'target_price': row['buyer_target'],
                'relative_price': row['relative_price'],
            },
            'seller': {
                'target_price': row['seller_target'],
                'price_delta_pct': row['price_delta_pct']
            }
        }

    def get_batch(
        self,
        split: str = 'train',
        batch_size: int = 32,
        shuffle: bool = True
    ) -> List[Dict]:
        """
        Get a batch of negotiation scenarios.

        Args:
            split: Which dataset split to use ('train', 'test', 'val')
            batch_size: Number of scenarios to return
            shuffle: Whether to shuffle the data

        Returns:
            List of scenario dictionaries
        """
        # Get appropriate dataset
        if split == 'train':
            data = self.train_data
        elif split == 'test':
            data = self.test_data
        elif split == 'val':
            data = self.val_data
        else:
            raise ValueError(f"Unknown split: {split}")

        if data is None:
            self.load_splits()
            data = getattr(self, f"{split}_data")

        # shuffle if requested
        if shuffle:
            data = data.sample(frac=1).reset_index(drop=True)

        # get batch
        batch = data.head(batch_size)

        # convert to negotiation pairs
        scenarios = [self.create_negotiation_pair(row) for _, row in batch.iterrows()]
        
        return scenarios

    def get_category_price_bounds(self, category: str) -> Dict[str, float]:
        """Get price statistics for a specific category."""
        if self.metadata is not None:
            try:
                price_ranges = self.metadata['train']['categories']['price_ranges'][category]
                return {
                    'min': price_ranges['min'],
                    'max': price_ranges['max'],
                    'mean': price_ranges['mean'],
                    'median': price_ranges['median']
                }
            except KeyError:
                warnings.warn(f"No price range data found for category: {category}")

        # fallback to computing from data
        if self.train_data is None:
            self.load_splits()

        cat_data = self.train_data[self.train_data['category'] == category]
        return {
            'min': cat_data['list_price'].min(),
            'max': cat_data['list_price'].max(),
            'mean': cat_data['list_price'].mean(),
            'median': cat_data['list_price'].median()
        }


def test_data_loader():
    """Basic tests for data loader functionality."""
    loader = DataLoader()

    # test data loading
    train, test, val = loader.load_splits()
    assert len(train) > 0, "Train data empty"
    assert len(test) > 0, "Test data empty"
    assert len(val) > 0, "Validation data empty"

    # verify expected columns
    expected_columns = [
        'scenario_id', 'split_type', 'category', 'list_price',
        'buyer_target', 'seller_target', 'title', 'description',
        'price_delta_pct', 'relative_price', 'title_token_count',
        'description_length', 'data_completeness', 'price_confidence',
        'has_images'
    ]
    for col in expected_columns:
        assert col in train.columns, f"Missing column: {col}"

    # test batch creation
    batch = loader.get_batch(batch_size=2)
    assert len(batch) == 2, "Incorrect batch size"

    # test category stats
    stats = loader.get_category_stats()
    assert len(stats) > 0, "No category stats generated"

    print("✓ All data loader tests passed")
    return loader

if __name__ == "__main__":
    loader = test_data_loader()