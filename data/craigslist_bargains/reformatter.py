# reformatter.py
import os, re, json, logging, unicodedata, warnings
from typing import Dict, Optional, Union
from pathlib import Path
import pandas as pd

# configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# constants for data processing
CATEGORY_MAP = {
    'electronics': 'electronics',
    'phone': 'electronics',
    'furniture': 'furniture',
    'housing': 'housing',
    'car': 'vehicles',
    'bike': 'vehicles'
}
REQUIRED_COLUMNS = [
    'scenario_id',
    'split_type',
    'category', 
    'list_price',
    'buyer_target',
    'seller_target',
    'title',
    'description',
    'price_delta_pct',
    'relative_price',
    'title_token_count',
    'description_length',
    'data_completeness',
    'price_confidence',
    'has_images'
]


class DataProcessor:
    """
    Reformats Craigslist Bargains JSON data into standardized CSVs for inference.
    Handles data cleaning, feature engineering, and quality filtering.

    Output files: train.csv, test.csv, validation.csv in parent directory
    """
    def __init__(self, raw_dir: Path, output_dir: Path):
        """Initialize processor with input/output directories."""
        self.raw_dir = raw_dir
        self.output_dir = output_dir
        self.category_stats: Dict = {} # stores category-level statistics

    def clean_price(self, price: Union[str, float, int]) -> Optional[float]:
        """Clean and validate price values."""
        if pd.isna(price) or price == -1:
            return None

        if isinstance(price, str):
            price = str(price).replace('$', '').replace(',', '')

        try:
            price = float(price)
            if price <= 0 or price > 1000000: # basic sanity check
                return None
            return price
        except (ValueError, TypeError) as e:
            warnings.warn(f"Error cleaning price: {str(e)}")
            return None

    def clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if pd.isna(text):
            return ""

        # remove HTML tags
        text = re.sub(r'<[^>]+>', '', str(text))

        # normalize unicode characters
        text = unicodedata.normalize('NFKD', text)

        # remove URLs
        text = re.sub(r'http\S+', '', text)

        # remove multiple spaces
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def validate_price_logic(self, row: pd.Series) -> bool:
        """Validate price relationships within a row."""
        try:
            # basic presence check
            if pd.isna(row['buyer_target']) or pd.isna(row['seller_target']) or pd.isna(row['list_price']):
                return False

            # verify buyer target < seller target
            if row['buyer_target'] >= row['seller_target']:
                return False

            # verify targets within reasonable range of list price
            if not (0.1 * row['list_price'] <= row['buyer_target'] <= 2 * row['list_price']):
                return False

            if not (0.1 * row['list_price'] <= row['seller_target'] <= 2 * row['list_price']):
                return False

            return True

        except (KeyError, TypeError) as e:
            warnings.warn(f"Error validating price logic: {str(e)}")
            return False

    def extract_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract core features from the raw data."""
        df = df.copy()
        # get item info
        df.loc[:, 'category'] = df['items'].apply(lambda x: x.get('Category', [None])[0])
        df.loc[:, 'list_price'] = df['items'].apply(lambda x: x.get('Price', [None])[0])
        df.loc[:, 'title'] = df['items'].apply(lambda x: x.get('Title', [None])[0])
        df.loc[:, 'description'] = df['items'].apply(lambda x: x.get('Description', [None])[0])
        df.loc[:, 'has_images'] = df['items'].apply(lambda x: len(x.get('Images', [])) > 0)

        # get agent info
        df.loc[:, 'buyer_target'] = df['agent_info'].apply(lambda x: x.get('Target', [None, None])[0])
        df.loc[:, 'seller_target'] = df['agent_info'].apply(lambda x: x.get('Target', [None, None])[1])

        return df

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived features."""
        df = df.copy()
        # price-based features
        df.loc[:, 'price_delta_pct'] = (df['seller_target'] - df['buyer_target']) / df['list_price']
        df.loc[:, 'relative_price'] = df['list_price'] / df.groupby('category')['list_price'].transform('median')

        # text-based features
        df.loc[:, 'description_length'] = df['description'].str.len()
        df.loc[:, 'title_token_count'] = df['title'].apply(lambda x: len(str(x).split()))

        # quality scores
        cols_to_check = ['category', 'list_price', 'buyer_target', 'seller_target', 'title', 'description']
        df.loc[:, 'data_completeness'] = df[cols_to_check].notna().mean(axis=1)
        df.loc[:, 'price_confidence'] = df.apply(self.validate_price_logic, axis=1)

        return df

    def process_split(self, input_file: str, split_name: str) -> pd.DataFrame:
        """Process a single data split."""
        logger.info(f"Processing {split_name} split from {input_file}")

        # load data
        with open(input_file, 'r') as f:
            data = [json.loads(line) for line in f]
        df = pd.DataFrame(data).copy()
        df['split_type'] = split_name
        df['scenario_id'] = [f"{split_name}_{i:05d}" for i in range(len(df))]

        # extract base features
        df = self.extract_base_features(df)

        # clean prices
        price_cols = ['list_price', 'buyer_target', 'seller_target']
        for col in price_cols:
            df.loc[:, col] = df[col].apply(self.clean_price)

        # fill missing prices with category medians
        for col in price_cols:
            medians = df.groupby('category')[col].transform('median')
            df.loc[:, col] = df[col].fillna(medians)

        # clean text
        text_cols = ['title', 'description']
        for col in text_cols:
            df.loc[:, col] = df[col].apply(self.clean_text)

        # normalize categories
        df.loc[:, 'category'] = df['category'].map(CATEGORY_MAP)

        # calculate features
        df = self.calculate_features(df)

        # quality filtering
        df = df.loc[
            (df['data_completeness'] > 0.8) &
            (df['price_confidence']) & # implicit == True check
            (df['description_length'] > 20)
            ]

        return df[REQUIRED_COLUMNS]

    def save_stats(self) -> None:
        """Save comprehensive dataset statistics."""
        def calculate_price_tier_counts(df):
            """Helper calculates price tier counts for a given category."""
            bins = [0, 3000, 10000, float('inf')]
            labels = ['low', 'mid', 'high']
            temp_df = df.copy()
            temp_df['price_tier'] = pd.cut(temp_df['list_price'], bins=bins, labels=labels)
            return temp_df['price_tier'].value_counts().to_dict()

        stats = {}
        raw_counts = { # from raw json files
            'train': 5247,
            'test': 838,
            'validation': 597
        }

        # collect comprehensive statistics
        for split_name in ['train', 'test', 'validation']:
            output_file = os.path.join(self.output_dir, f"{split_name}.csv")
            df = pd.read_csv(output_file)
            total_processed = len(df)

            # basic split statistics
            stats[split_name] = {
                'record_counts': {
                    'raw': raw_counts[split_name],
                    'processed': total_processed,
                    'retention_rate': round(total_processed / raw_counts[split_name] * 100, 2)
                },
                'categories': {
                    'distribution': {
                        cat: {
                            'count': count,
                            'percentage': round(count / total_processed * 100, 2)
                        }
                        for cat, count in df['category'].value_counts().items()
                    },
                    'price_ranges': {
                        cat: {
                            'min': float(df[df['category'] == cat]['list_price'].min()),
                            'max': float(df[df['category'] == cat]['list_price'].max()),
                            'mean': float(df[df['category'] == cat]['list_price'].mean()),
                            'median': float(df[df['category'] == cat]['list_price'].median())
                        }
                        for cat in df['category'].unique()
                    },
                    'price_tiers': {
                        cat: calculate_price_tier_counts(df[df['category'] == cat])
                        for cat in df['category'].unique()
                    }
                },
                'quality_metrics': {
                    'description_lengths': {
                        'min': int(df['description_length'].min()),
                        'max': int(df['description_length'].max()),
                        'mean': round(float(df['description_length'].mean()), 2)
                    },
                    'price_confidence': {
                        'pass_rate': round(df['price_confidence'].mean() * 100, 2)
                    },
                    'data_completeness': {
                        'mean': round(float(df['data_completeness'].mean()) * 100, 2),
                        'complete_records': round(len(df[df['data_completeness'] == 1.0]) / len(df) * 100, 2)
                    }
                },
                'negotiation_metrics': {
                    'price_delta': {
                        'mean_pct': round(float(df['price_delta_pct'].mean()) * 100, 2),
                        'median_pct': round(float(df['price_delta_pct'].median()) * 100, 2),
                        'by_category': {
                            cat: round(float(df[df['category'] == cat]['price_delta_pct'].mean()) * 100, 2)
                            for cat in df['category'].unique()
                        }
                    }
                }
            }

        # add dataset summary
        total_raw = sum(raw_counts.values())
        total_processed = sum(len(pd.read_csv(os.path.join(self.output_dir, f"{split}.csv")))
                                                for split in ['train', 'test', 'validation'])

        stats['dataset_summary'] = {
            'total_records': {
                'raw': total_raw,
                'processed': total_processed,
                'overall_retention': round(total_processed / total_raw * 100, 2)
            },
            'split_sizes': {
                split: {
                    'count': stats[split]['record_counts']['processed'],
                    'percentage': round(stats[split]['record_counts']['processed'] / total_processed * 100, 2)
                }
                for split in ['train', 'test', 'validation']
            },
            'category_totals': {
                cat: sum(stats[split]['categories']['distribution'].get(cat, {}).get('count', 0) 
                        for split in ['train', 'test', 'validation'])
                for cat in ['electronics', 'furniture', 'housing', 'vehicles']
            }
        }

        # save statistics with new name
        stats_file = os.path.join(self.output_dir, 'dataset_info.json')
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved comprehensive dataset statistics to {stats_file}")

    def process_all(self) -> None:
        """Process all data splits."""
        try: # collect statistics for each split
            for split in ['train', 'test', 'validation']:
                input_file = os.path.join(self.raw_dir, f"{split}.json")
                output_file = os.path.join(self.output_dir, f"{split}.csv")

                # process split
                df = self.process_split(input_file, split)

                # save processed data
                df.to_csv(output_file, index=False)
                logger.info(f"Saved {len(df)} records to {output_file}")

                # collect statistics
                self.category_stats[split] = {
                    'total_records': len(df),
                    'by_category': df['category'].value_counts().to_dict(),
                    'price_ranges': df.groupby('category')['list_price'].agg(['min', 'max', 'mean']).to_dict()
                }

            # save statistics
            self.save_stats()
            logger.info("Processing complete!")

        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            raise


if __name__ == "__main__":
    # get directory paths
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(current_dir, 'raw')
    output_dir = current_dir

    logger.info(f"Processing data from {raw_dir}")
    logger.info(f"Saving results to {output_dir}")

    # initialize and run processor
    processor = DataProcessor(raw_dir, output_dir)
    processor.process_all()