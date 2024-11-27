# reformatter.py
import os, re, json, logging, unicodedata
from typing import Dict, List, Optional, Union
from pathlib import Path
import numpy as np
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
        self.category_stats: Dict = {}  # For storing category-level statistics

    def clean_price(self, price: Union[str, float, int]) -> Optional[float]:
        """Clean and validate price values."""
        if pd.isna(price) or price == -1:
            return None
            
        if isinstance(price, str):
            price = str(price).replace('$', '').replace(',', '')
        
        try:
            price = float(price)
            if price <= 0 or price > 1000000:  # Basic sanity check
                return None
            return price
        except (ValueError, TypeError):
            return None

    def clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if pd.isna(text):
            return ""
            
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', str(text))
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

    def validate_price_logic(self, row: pd.Series) -> bool:
        """Validate price relationships within a row."""
        try:
            # Basic presence check
            if pd.isna(row['buyer_target']) or pd.isna(row['seller_target']) or pd.isna(row['list_price']):
                return False
                
            # Verify buyer target < seller target
            if row['buyer_target'] >= row['seller_target']:
                return False
                
            # Verify targets within reasonable range of list price
            if not (0.1 * row['list_price'] <= row['buyer_target'] <= 2 * row['list_price']):
                return False
                
            if not (0.1 * row['list_price'] <= row['seller_target'] <= 2 * row['list_price']):
                return False
                
            return True
            
        except (KeyError, TypeError):
            return False

    def extract_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract core features from the raw data."""
        # Extract item info
        df['category'] = df['items'].apply(lambda x: x.get('Category', [None])[0])
        df['list_price'] = df['items'].apply(lambda x: x.get('Price', [None])[0])
        df['title'] = df['items'].apply(lambda x: x.get('Title', [None])[0])
        df['description'] = df['items'].apply(lambda x: x.get('Description', [None])[0])
        df['has_images'] = df['items'].apply(lambda x: len(x.get('Images', [])) > 0)
        
        # Extract agent info
        df['buyer_target'] = df['agent_info'].apply(lambda x: x.get('Target', [None, None])[0])
        df['seller_target'] = df['agent_info'].apply(lambda x: x.get('Target', [None, None])[1])
        
        return df

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived features."""
        # Price-based features
        df['price_delta_pct'] = (df['seller_target'] - df['buyer_target']) / df['list_price']
        df['relative_price'] = df['list_price'] / df.groupby('category')['list_price'].transform('median')
        
        # Text-based features
        df['description_length'] = df['description'].str.len()
        df['title_token_count'] = df['title'].apply(lambda x: len(str(x).split()))
        
        # Quality scores
        cols_to_check = ['category', 'list_price', 'buyer_target', 'seller_target', 'title', 'description']
        df['data_completeness'] = df[cols_to_check].notna().mean(axis=1)
        df['price_confidence'] = df.apply(self.validate_price_logic, axis=1)
        
        return df

    def process_split(self, input_file: str, split_name: str) -> pd.DataFrame:
        """Process a single data split."""
        logger.info(f"Processing {split_name} split from {input_file}")
        
        # Load data
        with open(input_file, 'r') as f:
            data = [json.loads(line) for line in f]
        df = pd.DataFrame(data)
        df['split_type'] = split_name
        df['scenario_id'] = [f"{split_name}_{i:05d}" for i in range(len(df))]
        
        # Extract base features
        df = self.extract_base_features(df)
        
        # Clean prices
        price_cols = ['list_price', 'buyer_target', 'seller_target']
        for col in price_cols:
            df[col] = df[col].apply(self.clean_price)
            
        # Fill missing prices with category medians
        for col in price_cols:
            medians = df.groupby('category')[col].transform('median')
            df[col].fillna(medians, inplace=True)
            
        # Clean text
        text_cols = ['title', 'description']
        for col in text_cols:
            df[col] = df[col].apply(self.clean_text)
            
        # Normalize categories
        df['category'] = df['category'].map(CATEGORY_MAP)
        
        # Calculate features
        df = self.calculate_features(df)
        
        # Quality filtering
        df = df[
            (df['data_completeness'] > 0.8) &
            (df['price_confidence'] == True) &
            (df['description_length'] > 20)
        ]
        
        return df[REQUIRED_COLUMNS]

    def save_stats(self) -> None:
        """Save dataset statistics."""
        stats_file = self.output_dir / 'dataset_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(self.category_stats, f, indent=2)

    def process_all(self) -> None:
        """Process all data splits."""
        try:
            # Process each split
            for split in ['train', 'test', 'validation']:
                input_file = self.raw_dir / f"{split}.json"
                output_file = self.output_dir / f"{split}.csv"
                
                # Process split
                df = self.process_split(input_file, split)
                
                # Save processed data
                df.to_csv(output_file, index=False)
                logger.info(f"Saved {len(df)} records to {output_file}")
                
                # Collect statistics
                self.category_stats[split] = {
                    'total_records': len(df),
                    'by_category': df['category'].value_counts().to_dict(),
                    'price_ranges': df.groupby('category')['list_price'].agg(['min', 'max', 'mean']).to_dict()
                }
                
            # Save statistics
            self.save_stats()
            logger.info("Processing complete!")
            
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            raise

def main():
    """Main execution function."""
    # Get directory paths
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = current_dir
    output_dir = current_dir.parent
    
    logger.info(f"Processing data from {raw_dir}")
    logger.info(f"Saving results to {output_dir}")
    
    # Initialize and run processor
    processor = DataProcessor(raw_dir, output_dir)
    processor.process_all()

if __name__ == "__main__":
    main()