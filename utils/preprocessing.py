import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from urllib.parse import urlparse
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PhishingDataPreprocessor:
    def __init__(self, data_path="data/raw/"):
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.feature_columns = []
        
    def load_datasets(self):
        """Load all available datasets"""
        datasets = {}
        csv_files = [
            "Phishing_Legitimate_full.csv",
            "phishing_websites.csv", 
            "verified_online.csv"
        ]
        
        for file in csv_files:
            file_path = os.path.join(self.data_path, file)
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    datasets[file.replace('.csv', '')] = df
                    logger.info(f"Loaded {file}: {df.shape}")
                except Exception as e:
                    logger.error(f"Error loading {file}: {str(e)}")
            else:
                logger.warning(f"File not found: {file_path}")
        
        return datasets
    
    def extract_url_features(self, url):
        """Extract comprehensive features from a URL"""
        try:
            if pd.isna(url) or url is None:
                return self._get_default_url_features()
                
            url = str(url).strip()
            if not url:
                return self._get_default_url_features()
                
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            path = parsed.path
            query = parsed.query
            
            features = {
                # Length features
                'url_length': len(url),
                'domain_length': len(domain),
                'path_length': len(path),
                'query_length': len(query),
                
                # Character count features
                'num_dots': url.count('.'),
                'num_hyphens': url.count('-'),
                'num_underscores': url.count('_'),
                'num_slashes': url.count('/'),
                'num_params': len(query.split('&')) if query else 0,
                'num_digits': sum(c.isdigit() for c in url),
                
                # Security features
                'has_https': 1 if parsed.scheme == 'https' else 0,
                'has_ip': 1 if self._is_ip_address(domain) else 0,
                'has_port': 1 if ':' in domain and not domain.startswith('[') else 0,
                
                # Domain features
                'subdomain_count': max(0, len(domain.split('.')) - 2) if domain else 0,
                'domain_has_digits': 1 if any(c.isdigit() for c in domain) else 0,
                
                # Suspicious content features
                'suspicious_words': self._count_suspicious_words(url.lower()),
                'suspicious_tld': self._check_suspicious_tld(domain),
                
                # Complexity features
                'entropy': self._calculate_entropy(url),
                'digit_ratio': sum(c.isdigit() for c in url) / len(url) if url else 0,
                'special_char_ratio': sum(not c.isalnum() and c not in './-_' for c in url) / len(url) if url else 0,
                'vowel_ratio': sum(c.lower() in 'aeiou' for c in url) / len(url) if url else 0,
                
                # URL structure features
                'has_www': 1 if 'www.' in url.lower() else 0,
                'has_shortener': 1 if self._is_url_shortener(domain) else 0,
                'double_slash_count': url.count('//') - 1,  # Subtract 1 for protocol
                'at_symbol_count': url.count('@'),
                'percent_encoding': url.count('%'),
                
                # Path features
                'path_depth': len([p for p in path.split('/') if p]) if path else 0,
                'has_extension': 1 if (path and '.' in path.split('/')[-1]) else 0,
                'suspicious_extension': self._check_suspicious_extension(path)
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features from URL {url}: {str(e)}")
            return self._get_default_url_features()
    
    def _is_ip_address(self, domain):
        """Check if domain is an IP address"""
        ip_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
        return bool(re.match(ip_pattern, domain))
    
    def _count_suspicious_words(self, url):
        """Count suspicious words in URL"""
        suspicious_words = [
            'secure', 'account', 'update', 'confirm', 'login', 'signin',
            'banking', 'paypal', 'ebay', 'amazon', 'microsoft', 'apple',
            'verify', 'suspended', 'limited', 'security', 'notification',
            'urgent', 'immediate', 'expires', 'click', 'winner', 'prize'
        ]
        return sum(word in url for word in suspicious_words)
    
    def _check_suspicious_tld(self, domain):
        """Check for suspicious top-level domains"""
        if not domain:
            return 0
        suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.online', '.click', '.download']
        return 1 if any(domain.endswith(tld) for tld in suspicious_tlds) else 0
    
    def _is_url_shortener(self, domain):
        """Check if domain is a URL shortener"""
        shorteners = ['bit.ly', 'tinyurl.com', 't.co', 'goo.gl', 'ow.ly', 'short.link']
        return any(shortener in domain for shortener in shorteners)
    
    def _check_suspicious_extension(self, path):
        """Check for suspicious file extensions"""
        if not path:
            return 0
        suspicious_exts = ['.exe', '.zip', '.rar', '.bat', '.cmd', '.scr', '.jar']
        return 1 if any(path.lower().endswith(ext) for ext in suspicious_exts) else 0
    
    def _calculate_entropy(self, text):
        """Calculate Shannon entropy of text"""
        if not text:
            return 0
        
        # Count character frequencies
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        # Calculate entropy
        entropy = 0
        text_len = len(text)
        for count in char_counts.values():
            if count > 0:
                freq = count / text_len
                entropy -= freq * np.log2(freq)
        
        return entropy
    
    def _get_default_url_features(self):
        """Get default feature values for missing/invalid URLs"""
        return {
            'url_length': 0, 'domain_length': 0, 'path_length': 0, 'query_length': 0,
            'num_dots': 0, 'num_hyphens': 0, 'num_underscores': 0, 'num_slashes': 0,
            'num_params': 0, 'num_digits': 0, 'has_https': 0, 'has_ip': 0, 'has_port': 0,
            'subdomain_count': 0, 'domain_has_digits': 0, 'suspicious_words': 0,
            'suspicious_tld': 0, 'entropy': 0, 'digit_ratio': 0, 'special_char_ratio': 0,
            'vowel_ratio': 0, 'has_www': 0, 'has_shortener': 0, 'double_slash_count': 0,
            'at_symbol_count': 0, 'percent_encoding': 0, 'path_depth': 0,
            'has_extension': 0, 'suspicious_extension': 0
        }
    
    def preprocess_dataset(self, df, url_column='url', label_column='label'):
        """Preprocess a dataset with URL extraction"""
        logger.info(f"Preprocessing dataset with shape: {df.shape}")
        
        # Clean the dataset
        df = df.dropna(subset=[url_column])
        
        # Extract URL features
        logger.info("Extracting URL features...")
        url_features_list = []
        
        for idx, url in enumerate(df[url_column]):
            if idx % 1000 == 0:
                logger.info(f"Processed {idx}/{len(df)} URLs")
            features = self.extract_url_features(url)
            url_features_list.append(features)
        
        # Create features dataframe
        url_features_df = pd.DataFrame(url_features_list)
        
        # Combine with original data
        processed_df = pd.concat([df.reset_index(drop=True), url_features_df], axis=1)
        
        # Process labels
        if label_column and label_column in processed_df.columns:
            processed_df = self._standardize_labels(processed_df, label_column)
        else:
            logger.warning(f"No label column found, creating default labels")
            processed_df['label'] = 0
        
        logger.info(f"Preprocessed dataset shape: {processed_df.shape}")
        return processed_df
    
    def process_feature_dataset(self, df, label_column):
        """Process datasets that already have features"""
        logger.info(f"Processing feature-based dataset with shape: {df.shape}")
        
        processed_df = df.copy()
        
        # Standardize labels
        processed_df = self._standardize_labels(processed_df, label_column)
        
        # Add missing URL features with defaults
        url_feature_defaults = self._get_default_url_features()
        for feature, default_value in url_feature_defaults.items():
            if feature not in processed_df.columns:
                processed_df[feature] = default_value
        
        return processed_df
    
    def _standardize_labels(self, df, label_column):
        """Standardize labels to 0 (legitimate) and 1 (phishing)"""
        if label_column not in df.columns:
            df['label'] = 0
            return df
        
        # Create a copy to avoid modifying original
        df = df.copy()
        
        # Handle different label formats
        if label_column == 'CLASS_LABEL':
            df['label'] = df[label_column]
        elif label_column == 'result':
            df['label'] = df[label_column].map({-1: 0, 1: 1})
        else:
            # General mapping
            label_mapping = {
                'legitimate': 0, 'phishing': 1,
                'good': 0, 'bad': 1,
                'benign': 0, 'malicious': 1,
                'safe': 0, 'unsafe': 1,
                0: 0, 1: 1, -1: 0,
                '0': 0, '1': 1, '-1': 0,
                True: 1, False: 0,
                'yes': 1, 'no': 0,
                'ham': 0, 'spam': 1
            }
            df['label'] = df[label_column].map(label_mapping)
        
        # Fill missing labels with 0 (legitimate)
        df['label'] = df['label'].fillna(0)
        
        # Ensure binary labels
        df['label'] = df['label'].astype(int)
        
        return df
    
    def combine_datasets(self, datasets):
        """Combine multiple datasets into one"""
        combined_data = []
        
        for name, df in datasets.items():
            logger.info(f"\nProcessing dataset: {name}")
            logger.info(f"Shape: {df.shape}")
            logger.info(f"Columns: {list(df.columns)}")
            
            try:
                if name == "verified_online":
                    # This is a URL-only dataset - all are phishing
                    url_col = self._identify_url_column(df)
                    if url_col:
                        processed_df = self.preprocess_dataset(df, url_col, None)
                        processed_df['label'] = 1  # All are phishing
                        processed_df['source'] = name
                        combined_data.append(processed_df)
                        logger.info(f"Processed {name}: {processed_df.shape}")
                
                elif name in ["Phishing_Legitimate_full", "phishing_websites"]:
                    # These are feature-based datasets
                    label_col = self._identify_label_column(df)
                    if label_col:
                        processed_df = self.process_feature_dataset(df, label_col)
                        processed_df['source'] = name
                        combined_data.append(processed_df)
                        logger.info(f"Processed {name}: {processed_df.shape}")
                    else:
                        logger.warning(f"No label column found in {name}")
                
                else:
                    # Generic processing
                    url_col = self._identify_url_column(df)
                    label_col = self._identify_label_column(df)
                    
                    if url_col:
                        processed_df = self.preprocess_dataset(df, url_col, label_col)
                        processed_df['source'] = name
                        combined_data.append(processed_df)
                        logger.info(f"Processed {name}: {processed_df.shape}")
                    else:
                        logger.warning(f"No URL column found in {name}")
                        
            except Exception as e:
                logger.error(f"Error processing {name}: {str(e)}")
                continue
        
        if not combined_data:
            raise ValueError("No valid datasets found to combine")
        
        # Align all datasets to have the same columns
        all_columns = set()
        for df in combined_data:
            all_columns.update(df.columns)
        
        # Add missing columns to each dataset
        for i, df in enumerate(combined_data):
            for col in all_columns:
                if col not in df.columns:
                    if col in self._get_default_url_features():
                        df[col] = self._get_default_url_features()[col]
                    elif col == 'label':
                        df[col] = 0
                    elif col == 'source':
                        df[col] = 'unknown'
                    else:
                        df[col] = 0
            combined_data[i] = df
        
        # Combine all datasets
        final_df = pd.concat(combined_data, ignore_index=True, sort=False)
        
        logger.info(f"Combined dataset shape: {final_df.shape}")
        logger.info(f"Label distribution: {final_df['label'].value_counts().to_dict()}")
        
        return final_df
    
    def _identify_url_column(self, df):
        """Identify the URL column in a dataset"""
        url_indicators = ['url', 'URL', 'website', 'link', 'address', 'domain']
        
        for col in df.columns:
            if col.lower() in [indicator.lower() for indicator in url_indicators]:
                return col
        
        # Check for columns that look like URLs
        for col in df.columns:
            if df[col].dtype == 'object':
                sample_values = df[col].dropna().head(10).astype(str)
                if sample_values.str.contains(r'http|www|\.com|\.org|\.net', case=False).any():
                    return col
        
        return None
    
    def _identify_label_column(self, df):
        """Identify the label column in a dataset"""
        label_indicators = [
            'CLASS_LABEL', 'result', 'verified', 'label', 'class', 
            'target', 'phishing', 'legitimate', 'type', 'category'
        ]
        
        # Direct match
        for col in df.columns:
            if col in label_indicators:
                return col
        
        # Look for binary/categorical columns
        for col in df.columns:
            if df[col].dtype in ['object', 'int64', 'float64']:
                unique_vals = df[col].dropna().unique()
                if len(unique_vals) <= 10:  # Likely categorical
                    unique_str = [str(val).lower() for val in unique_vals]
                    binary_indicators = [
                        '0', '1', 'true', 'false', 'phishing', 'legitimate',
                        'good', 'bad', 'yes', 'no', 'ham', 'spam'
                    ]
                    if any(val in binary_indicators for val in unique_str):
                        return col
        
        return None
    
    def feature_engineering(self, df):
        """Create additional engineered features"""
        logger.info("Starting feature engineering...")
        
        # Ensure all base features exist
        url_feature_defaults = self._get_default_url_features()
        for feature, default_value in url_feature_defaults.items():
            if feature not in df.columns:
                df[feature] = default_value
        
        # Create interaction features
        df['length_entropy_ratio'] = df['url_length'] / (df['entropy'] + 1)
        df['suspicious_ratio'] = df['suspicious_words'] / (df['url_length'] + 1)
        df['complexity_score'] = (df['num_dots'] + df['num_slashes'] + df['num_params']) / (df['url_length'] + 1)
        df['digit_density'] = df['num_digits'] / (df['url_length'] + 1)
        df['special_char_density'] = (df['num_hyphens'] + df['num_underscores'] + df['at_symbol_count']) / (df['url_length'] + 1)
        
        # Create categorical features
        try:
            df['domain_length_category'] = pd.cut(
                df['domain_length'], 
                bins=[0, 10, 20, 50, float('inf')], 
                labels=[0, 1, 2, 3]
            ).fillna(0).astype(int)
            
            df['url_length_category'] = pd.cut(
                df['url_length'], 
                bins=[0, 30, 75, 150, float('inf')], 
                labels=[0, 1, 2, 3]
            ).fillna(0).astype(int)
            
            df['entropy_category'] = pd.cut(
                df['entropy'], 
                bins=[0, 2, 4, 6, float('inf')], 
                labels=[0, 1, 2, 3]
            ).fillna(0).astype(int)
            
        except Exception as e:
            logger.warning(f"Error creating categorical features: {e}")
            df['domain_length_category'] = 0
            df['url_length_category'] = 0
            df['entropy_category'] = 0
        
        # Fill any remaining NaN values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            df[col] = df[col].fillna(0)
        
        logger.info(f"Feature engineering completed. Final shape: {df.shape}")
        return df
    
    def prepare_for_training(self, df, target_column='label', test_size=0.2):
        """Prepare data for machine learning training"""
        logger.info("Preparing data for training...")
        
        # Get feature columns (numeric columns except target and metadata)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        exclude_columns = [target_column, 'source']
        feature_cols = [col for col in numeric_columns if col not in exclude_columns]
        
        logger.info(f"Found {len(feature_cols)} feature columns")
        
        if len(feature_cols) == 0:
            raise ValueError("No feature columns found for training!")
        
        # Encode categorical columns if any
        categorical_cols = df[feature_cols].select_dtypes(include=['object', 'category']).columns
        df_processed = df.copy()
        for col in categorical_cols:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        
        # Prepare features and target
        X = df_processed[feature_cols]
        y = df_processed[target_column] if target_column in df_processed.columns else None
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        # Store feature column names
        self.feature_columns = X.columns.tolist()
        
        logger.info(f"Feature columns: {self.feature_columns[:10]}...")  # Show first 10
        
        # Split data if we have labels
        if y is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=42, 
                stratify=y if len(y.unique()) > 1 else None
            )
            return X_train, X_test, y_train, y_test
        else:
            return X_scaled, None, None, None
    
    def save_processed_data(self, df, filename='processed_phishing_data.csv', output_dir='data/processed/'):
        """Save processed data to CSV"""
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        
        # Ensure we have the required columns
        if 'label' not in df.columns:
            df['label'] = 0
        
        # Get feature columns for reporting
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_columns if col not in ['label', 'source']]
        
        logger.info(f"Saving dataset with {len(df)} rows and {len(feature_cols)} features")
        logger.info(f"Label distribution: {df['label'].value_counts().to_dict()}")
        
        df.to_csv(output_path, index=False)
        logger.info(f"Data saved to: {output_path}")
        
        return output_path

def main():
    """Main preprocessing pipeline"""
    logger.info("Starting phishing data preprocessing pipeline...")
    
    # Initialize preprocessor
    preprocessor = PhishingDataPreprocessor()
    
    # Load datasets
    datasets = preprocessor.load_datasets()
    if not datasets:
        logger.error("No datasets loaded. Please check your data directory.")
        return None, None, None
    
    # Print dataset information
    print("\n" + "="*60)
    print("DATASET INFORMATION")
    print("="*60)
    
    for name, df in datasets.items():
        print(f"\n{name}:")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        
        url_col = preprocessor._identify_url_column(df)
        label_col = preprocessor._identify_label_column(df)
        print(f"  URL column: {url_col}")
        print(f"  Label column: {label_col}")
        
        if label_col and label_col in df.columns:
            print(f"  Label distribution: {df[label_col].value_counts().to_dict()}")
    
    # Process datasets
    try:
        combined_df = preprocessor.combine_datasets(datasets)
        enhanced_df = preprocessor.feature_engineering(combined_df)
        
        # Prepare for training
        X_train, X_test, y_train, y_test = preprocessor.prepare_for_training(enhanced_df)
        
        # Save processed data
        output_path = preprocessor.save_processed_data(enhanced_df)
        
        # Print summary
        print("\n" + "="*60)
        print("PREPROCESSING SUMMARY")
        print("="*60)
        print(f"Total samples: {len(enhanced_df):,}")
        print(f"Number of features: {len(preprocessor.feature_columns)}")
        print(f"Training samples: {len(X_train):,}")
        print(f"Test samples: {len(X_test):,}")
        
        if y_train is not None:
            print(f"Training - Phishing: {sum(y_train == 1):,}, Legitimate: {sum(y_train == 0):,}")
            print(f"Test - Phishing: {sum(y_test == 1):,}, Legitimate: {sum(y_test == 0):,}")
        
        print(f"\nTop 15 features:")
        for i, col in enumerate(preprocessor.feature_columns[:15], 1):
            print(f"  {i:2d}. {col}")
        
        if len(preprocessor.feature_columns) > 15:
            print(f"  ... and {len(preprocessor.feature_columns) - 15} more features")
        
        print(f"\nProcessed data saved to: {output_path}")
        
        return preprocessor, enhanced_df, (X_train, X_test, y_train, y_test)
        
    except Exception as e:
        logger.error(f"Error in preprocessing pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    main()