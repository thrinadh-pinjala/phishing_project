import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, List, Tuple, Optional
import pickle
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

class DataLoader:
    """
    Utility class for loading and managing phishing detection datasets
    """
    
    def __init__(self, base_path: str = "data/"):
        self.base_path = base_path
        self.raw_path = os.path.join(base_path, "raw")
        self.processed_path = os.path.join(base_path, "processed")
        
        # Create directories if they don't exist
        os.makedirs(self.raw_path, exist_ok=True)
        os.makedirs(self.processed_path, exist_ok=True)
    
    def load_raw_data(self, filename: str = None) -> Dict[str, pd.DataFrame]:
        """
        Load raw CSV files from the data/raw directory
        
        Args:
            filename: Specific file to load, if None loads all CSV files
            
        Returns:
            Dictionary of DataFrames with filename as key
        """
        datasets = {}
        
        if filename:
            file_path = os.path.join(self.raw_path, filename)
            if os.path.exists(file_path):
                datasets[filename] = pd.read_csv(file_path)
                logger.info(f"Loaded {filename}: {datasets[filename].shape}")
        else:
            # Load all CSV files
            for file in os.listdir(self.raw_path):
                if file.endswith('.csv'):
                    try:
                        file_path = os.path.join(self.raw_path, file)
                        df = pd.read_csv(file_path)
                        datasets[file] = df
                        logger.info(f"Loaded {file}: {df.shape}")
                    except Exception as e:
                        logger.error(f"Error loading {file}: {str(e)}")
        
        return datasets
    
    def load_processed_data(self, filename: str = "processed_phishing_data.csv") -> pd.DataFrame:
        """
        Load processed data from data/processed directory
        
        Args:
            filename: Name of the processed file
            
        Returns:
            Processed DataFrame
        """
        file_path = os.path.join(self.processed_path, filename)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Processed file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        logger.info(f"Loaded processed data: {df.shape}")
        return df
    
    def get_train_test_split(self, df: pd.DataFrame, 
                           target_column: str = 'label',
                           test_size: float = 0.2,
                           stratify: bool = True,
                           random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and testing sets
        
        Args:
            df: Input DataFrame
            target_column: Name of target column
            test_size: Proportion of test set
            stratify: Whether to stratify split
            random_state: Random seed
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        # Separate features and target
        feature_cols = [col for col in df.columns if col not in [target_column, 'url', 'source']]
        X = df[feature_cols]
        y = df[target_column]
        
        # Perform split
        stratify_param = y if stratify else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
        )
        
        logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def save_data_splits(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                        y_train: pd.Series, y_test: pd.Series,
                        prefix: str = "split") -> None:
        """
        Save train/test splits to processed directory
        
        Args:
            X_train, X_test, y_train, y_test: Data splits
            prefix: Prefix for saved files
        """
        # Save as CSV
        X_train.to_csv(os.path.join(self.processed_path, f"{prefix}_X_train.csv"), index=False)
        X_test.to_csv(os.path.join(self.processed_path, f"{prefix}_X_test.csv"), index=False)
        y_train.to_csv(os.path.join(self.processed_path, f"{prefix}_y_train.csv"), index=False)
        y_test.to_csv(os.path.join(self.processed_path, f"{prefix}_y_test.csv"), index=False)
        
        # Save as pickle for faster loading
        with open(os.path.join(self.processed_path, f"{prefix}_data_splits.pkl"), 'wb') as f:
            pickle.dump((X_train, X_test, y_train, y_test), f)
        
        logger.info(f"Data splits saved with prefix '{prefix}'")
    
    def load_data_splits(self, prefix: str = "split") -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Load previously saved train/test splits
        
        Args:
            prefix: Prefix of saved files
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        pickle_path = os.path.join(self.processed_path, f"{prefix}_data_splits.pkl")
        
        if os.path.exists(pickle_path):
            # Load from pickle (faster)
            with open(pickle_path, 'rb') as f:
                X_train, X_test, y_train, y_test = pickle.load(f)
        else:
            # Load from CSV files
            X_train = pd.read_csv(os.path.join(self.processed_path, f"{prefix}_X_train.csv"))
            X_test = pd.read_csv(os.path.join(self.processed_path, f"{prefix}_X_test.csv"))
            y_train = pd.read_csv(os.path.join(self.processed_path, f"{prefix}_y_train.csv")).squeeze()
            y_test = pd.read_csv(os.path.join(self.processed_path, f"{prefix}_y_test.csv")).squeeze()
        
        logger.info(f"Loaded data splits: Train {X_train.shape}, Test {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def get_data_info(self, df: pd.DataFrame) -> Dict:
        """
        Get comprehensive information about the dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with dataset information
        """
        info = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist()
        }
        
        # Add target distribution if label column exists
        if 'label' in df.columns:
            info['label_distribution'] = df['label'].value_counts().to_dict()
            info['class_balance'] = df['label'].value_counts(normalize=True).to_dict()
        
        return info
    
    def validate_data(self, df: pd.DataFrame, required_columns: List[str] = None) -> bool:
        """
        Validate dataset structure and content
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            
        Returns:
            True if validation passes
        """
        validation_errors = []
        
        # Check if DataFrame is empty
        if df.empty:
            validation_errors.append("DataFrame is empty")
        
        # Check required columns
        if required_columns:
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                validation_errors.append(f"Missing required columns: {missing_cols}")
        
        # Check for excessive missing values
        missing_pct = df.isnull().sum() / len(df)
        high_missing = missing_pct[missing_pct > 0.5].index.tolist()
        if high_missing:
            validation_errors.append(f"Columns with >50% missing values: {high_missing}")
        
        # Log validation results
        if validation_errors:
            for error in validation_errors:
                logger.error(f"Validation error: {error}")
            return False
        else:
            logger.info("Data validation passed")
            return True
    
    def sample_data(self, df: pd.DataFrame, n_samples: int = 1000, 
                   stratify_column: str = None) -> pd.DataFrame:
        """
        Sample data for quick testing or exploration
        
        Args:
            df: Input DataFrame
            n_samples: Number of samples to extract
            stratify_column: Column to stratify sampling on
            
        Returns:
            Sampled DataFrame
        """
        if len(df) <= n_samples:
            return df.copy()
        
        if stratify_column and stratify_column in df.columns:
            # Stratified sampling
            sampled_df = df.groupby(stratify_column, group_keys=False).apply(
                lambda x: x.sample(min(len(x), n_samples // df[stratify_column].nunique()))
            )
        else:
            # Random sampling
            sampled_df = df.sample(n=n_samples, random_state=42)
        
        logger.info(f"Sampled {len(sampled_df)} rows from {len(df)} total rows")
        return sampled_df.reset_index(drop=True)

# Convenience functions for common operations
def quick_load(data_type: str = "processed", filename: str = None) -> pd.DataFrame:
    """
    Quick function to load data
    
    Args:
        data_type: 'raw' or 'processed'
        filename: Specific file to load
        
    Returns:
        Loaded DataFrame
    """
    loader = DataLoader()
    
    if data_type == "processed":
        return loader.load_processed_data(filename or "processed_phishing_data.csv")
    else:
        datasets = loader.load_raw_data(filename)
        if len(datasets) == 1:
            return list(datasets.values())[0]
        return datasets

def get_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Get a sample of processed data for quick testing
    
    Args:
        n_samples: Number of samples
        
    Returns:
        Sampled DataFrame
    """
    loader = DataLoader()
    df = loader.load_processed_data()
    return loader.sample_data(df, n_samples, 'label')

# Example usage
if __name__ == "__main__":
    # Initialize data loader
    loader = DataLoader()
    
    # Load all raw data
    raw_datasets = loader.load_raw_data()
    print(f"Loaded {len(raw_datasets)} raw datasets")
    
    # Try to load processed data
    try:
        processed_df = loader.load_processed_data()
        print(f"Processed data shape: {processed_df.shape}")
        
        # Get data information
        info = loader.get_data_info(processed_df)
        print(f"Dataset info: {info}")
        
        # Validate data
        is_valid = loader.validate_data(processed_df, ['label'])
        print(f"Data validation: {'Passed' if is_valid else 'Failed'}")
        
    except FileNotFoundError:
        print("No processed data found. Run preprocessing first.")