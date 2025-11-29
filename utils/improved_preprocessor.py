import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from urllib.parse import urlparse
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedPhishingDataPreprocessor:
    """
    Improved preprocessing class for phishing detection datasets with data cleaning
    """
    
    def __init__(self, data_path="data/raw/"):
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.feature_columns = []
        self.feature_mapping = self._create_feature_mapping()
        
    def _create_feature_mapping(self):
        """Create mapping for consolidating similar features from different datasets"""
        return {
            # URL Length features
            'url_length': ['UrlLength', 'url_length'],
            'domain_length': ['HostnameLength', 'domain_length'],
            'path_length': ['PathLength', 'path_length'],
            'query_length': ['QueryLength', 'query_length'],
            
            # Dot and subdomain features
            'num_dots': ['NumDots', 'num_dots'],
            'subdomain_level': ['SubdomainLevel', 'subdomain_count', 'having_sub_domain'],
            
            # Dash and hyphen features
            'num_hyphens': ['NumDash', 'NumDashInHostname', 'num_hyphens'],
            
            # Special character features
            'num_underscores': ['NumUnderscore', 'num_underscores'],
            'num_slashes': ['PathLevel', 'num_slashes'],
            'at_symbol': ['AtSymbol', 'having_at_symbol'],
            'tilde_symbol': ['TildeSymbol'],
            'num_percent': ['NumPercent'],
            'num_ampersand': ['NumAmpersand'],
            'num_hash': ['NumHash'],
            'double_slash': ['DoubleSlashInPath', 'double_slash_redirecting'],
            
            # Security features
            'has_https': ['NoHttps', 'has_https', 'https_token'],  # Note: NoHttps is inverted
            'has_ip': ['IpAddress', 'has_ip', 'having_ip_address'],
            'has_port': ['has_port', 'port'],
            'ssl_state': ['sslfinal_state'],
            
            # Content features
            'suspicious_words': ['NumSensitiveWords', 'suspicious_words'],
            'numeric_chars': ['NumNumericChars'],
            'random_string': ['RandomString'],
            'embedded_brand': ['EmbeddedBrandName'],
            
            # Web page features
            'external_links': ['PctExtHyperlinks'],
            'external_resources': ['PctExtResourceUrls', 'PctExtResourceUrlsRT'],
            'external_favicon': ['ExtFavicon', 'favicon'],
            'insecure_forms': ['InsecureForms'],
            'form_action': ['RelativeFormAction', 'ExtFormAction', 'AbnormalFormAction', 'AbnormalExtFormActionR'],
            'submit_to_email': ['SubmitInfoToEmail', 'submitting_to_email'],
            'iframe': ['IframeOrFrame', 'iframe'],
            'popup_window': ['PopUpWindow', 'popupwindow'],
            'right_click_disabled': ['RightClickDisabled', 'rightclick'],
            'missing_title': ['MissingTitle'],
            
            # Domain features
            'domain_registration_length': ['domain_registration_length'],
            'age_of_domain': ['age_of_domain'],
            'dns_record': ['dnsrecord'],
            'web_traffic': ['web_traffic'],
            'page_rank': ['page_rank'],
            'google_index': ['google_index'],
            'links_pointing': ['links_pointing_to_page'],
            
            # URL features
            'shortening_service': ['shortining_service'],  # Note: typo in original
            'prefix_suffix': ['prefix_suffix'],
            'abnormal_url': ['abnormal_url'],
            'redirect': ['redirect'],
            'on_mouseover': ['on_mouseover'],
            'statistical_report': ['statistical_report'],
            
            # Query and parameter features
            'num_query_components': ['NumQueryComponents', 'num_params'],
            'request_url': ['request_url'],
            'url_of_anchor': ['url_of_anchor'],
            'links_in_tags': ['links_in_tags'],
            'sfh': ['sfh'],
            
            # Computed features (keep as is)
            'entropy': ['entropy'],
            'digit_ratio': ['digit_ratio'],
            'special_char_ratio': ['special_char_ratio'],
            'length_entropy_ratio': ['length_entropy_ratio'],
            'suspicious_ratio': ['suspicious_ratio'],
            'complexity_score': ['complexity_score']
        }
    
    def consolidate_features(self, df):
        """Consolidate similar features from different datasets"""
        logger.info("Consolidating features from different datasets...")
        
        consolidated_df = pd.DataFrame()
        
        # Copy non-feature columns
        info_columns = ['source', 'url', 'phish_id', 'target', 'submission_time', 'verification_time']
        for col in info_columns:
            if col in df.columns:
                consolidated_df[col] = df[col]
        
        # Consolidate mapped features
        for new_feature, old_features in self.feature_mapping.items():
            values = None
            for old_feature in old_features:
                if old_feature in df.columns:
                    current_values = df[old_feature]
                    
                    # Handle special cases
                    if old_feature == 'NoHttps':  # Invert NoHttps to has_https
                        current_values = 1 - current_values
                    elif old_feature in ['having_sub_domain', 'having_ip_address', 'having_at_symbol']:
                        # Convert -1,1 to 0,1
                        current_values = current_values.map({-1: 0, 1: 1, 0: 0}).fillna(0)
                    elif old_feature == 'port':
                        # Convert port to binary has_port
                        current_values = (current_values != 0).astype(int)
                    
                    # Use first non-zero/non-null source, or combine if needed
                    if values is None:
                        values = current_values
                    else:
                        # For binary features, use OR logic
                        if new_feature.startswith('has_') or new_feature.startswith('num_'):
                            values = np.maximum(values, current_values.fillna(0))
                        else:
                            # For other features, use first non-zero value
                            mask = (values == 0) | values.isna()
                            values = values.where(~mask, current_values)
            
            if values is not None:
                consolidated_df[new_feature] = values.fillna(0)
            else:
                # Create default feature if not found
                consolidated_df[new_feature] = 0
        
        # Handle label consolidation
        label_columns = ['CLASS_LABEL', 'result', 'label', 'verified']
        final_label = None
        
        for label_col in label_columns:
            if label_col in df.columns:
                current_label = df[label_col]
                
                # Standardize labels
                if label_col == 'CLASS_LABEL':
                    # 1 = phishing, 0 = legitimate
                    standardized_label = current_label
                elif label_col == 'result':
                    # -1 = legitimate, 1 = phishing
                    standardized_label = current_label.map({-1: 0, 1: 1}).fillna(0)
                elif label_col == 'verified':
                    # 'yes' = phishing (since this is phishing dataset)
                    standardized_label = current_label.map({'yes': 1, 'no': 0}).fillna(1)
                else:
                    standardized_label = current_label
                
                if final_label is None:
                    final_label = standardized_label
                else:
                    # Use first non-null label
                    mask = final_label.isna()
                    final_label = final_label.where(~mask, standardized_label)
        
        if final_label is not None:
            consolidated_df['label'] = final_label.fillna(0).astype(int)
        else:
            consolidated_df['label'] = 0
        
        logger.info(f"Consolidated dataset shape: {consolidated_df.shape}")
        return consolidated_df
    
    def clean_and_validate_data(self, df):
        """Clean and validate the consolidated data"""
        logger.info("Cleaning and validating data...")
        
        # Remove rows with all zero features (likely from misaligned data)
        feature_cols = [col for col in df.columns if col not in ['source', 'url', 'phish_id', 'target', 'submission_time', 'verification_time', 'label']]
        
        # Calculate row sums to identify potentially invalid rows
        row_sums = df[feature_cols].abs().sum(axis=1)
        valid_rows = row_sums > 0
        
        logger.info(f"Removing {sum(~valid_rows)} rows with all zero features")
        df_clean = df[valid_rows].copy()
        
        # Handle outliers (cap extreme values)
        for col in feature_cols:
            if df_clean[col].dtype in ['int64', 'float64']:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                
                # Cap outliers
                df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
        
        # Ensure binary features are truly binary
        binary_features = [col for col in feature_cols if col.startswith('has_') or 
                          col in ['at_symbol', 'tilde_symbol', 'random_string', 'embedded_brand', 
                                'insecure_forms', 'submit_to_email', 'iframe', 'popup_window', 
                                'right_click_disabled', 'missing_title']]
        
        for col in binary_features:
            if col in df_clean.columns:
                df_clean[col] = (df_clean[col] > 0).astype(int)
        
        logger.info(f"Cleaned dataset shape: {df_clean.shape}")
        return df_clean
    
    def create_additional_features(self, df):
        """Create additional engineered features"""
        logger.info("Creating additional engineered features...")
        
        # URL complexity features
        df['url_complexity'] = (df['num_dots'] + df['num_hyphens'] + df['num_underscores'] + 
                               df['num_slashes'] + df['at_symbol'])
        
        # Security risk score
        df['security_risk_score'] = (df['has_ip'] + (1 - df['has_https']) + df['has_port'] + 
                                   df['shortening_service'] + df['suspicious_words'])
        
        # Domain trust score
        df['domain_trust_score'] = (df['age_of_domain'] + df['dns_record'] + df['web_traffic'] + 
                                  df['page_rank'] + df['google_index']) / 5
        
        # Content risk score
        df['content_risk_score'] = (df['external_links'] + df['external_resources'] + 
                                  df['submit_to_email'] + df['iframe'] + df['popup_window']) / 5
        
        # Length ratios
        df['path_to_url_ratio'] = df['path_length'] / (df['url_length'] + 1)
        df['query_to_url_ratio'] = df['query_length'] / (df['url_length'] + 1)
        df['domain_to_url_ratio'] = df['domain_length'] / (df['url_length'] + 1)
        
        return df
    
    def prepare_final_dataset(self, df, target_column='label', test_size=0.2):
        """Prepare the final clean dataset for machine learning"""
        logger.info("Preparing final dataset for machine learning...")
        
        # Select final feature columns (exclude metadata)
        exclude_cols = ['source', 'url', 'phish_id', 'target', 'submission_time', 
                       'verification_time', target_column]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        y = df[target_column] if target_column in df.columns else None
        
        # Handle any remaining missing values
        X = X.fillna(0)
        
        # Convert any remaining categorical columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Store feature names
        self.feature_columns = X.columns.tolist()
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X), 
            columns=X.columns,
            index=X.index
        )
        
        if y is not None and len(y.unique()) > 1:
            # Split data if we have labels and variation
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=42, 
                stratify=y if len(y.unique()) > 1 else None
            )
            return X_train, X_test, y_train, y_test, feature_cols
        else:
            return X_scaled, None, None, None, feature_cols
    
    def process_existing_data(self, input_file='data/processed/processed_phishing_data.csv'):
        """Process already combined but messy data"""
        logger.info(f"Loading existing processed data from: {input_file}")
        
        try:
            df = pd.read_csv(input_file)
            logger.info(f"Loaded data with shape: {df.shape}")
        except FileNotFoundError:
            logger.error(f"File not found: {input_file}")
            return None
        
        # Consolidate features
        consolidated_df = self.consolidate_features(df)
        
        # Clean and validate data
        clean_df = self.clean_and_validate_data(consolidated_df)
        
        # Create additional features
        enhanced_df = self.create_additional_features(clean_df)
        
        return enhanced_df
    
    def save_cleaned_data(self, df, filename='cleaned_phishing_data.csv', output_dir='data/processed/'):
        """Save cleaned data"""
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        df.to_csv(output_path, index=False)
        logger.info(f"Cleaned data saved to: {output_path}")
        return output_path

def main():
    """Main function to clean existing processed data"""
    processor = ImprovedPhishingDataPreprocessor()
    
    # Process existing messy data
    enhanced_df = processor.process_existing_data()
    
    if enhanced_df is None:
        logger.error("Failed to process data")
        return
    
    # Prepare for machine learning
    X_train, X_test, y_train, y_test, feature_names = processor.prepare_final_dataset(enhanced_df)
    
    # Save cleaned data
    processor.save_cleaned_data(enhanced_df)
    
    # Print summary
    print("\n" + "="*60)
    print("CLEANED DATA SUMMARY")
    print("="*60)
    print(f"Total samples: {len(enhanced_df)}")
    print(f"Total features: {len(feature_names)}")
    
    if y_train is not None:
        print(f"Training samples: {len(X_train)}")
        if X_test is not None:
            print(f"Test samples: {len(X_test)}")
        
        # Label distribution
        total_phishing = sum(enhanced_df['label'] == 1)
        total_legitimate = sum(enhanced_df['label'] == 0)
        print(f"Phishing samples: {total_phishing} ({total_phishing/len(enhanced_df)*100:.1f}%)")
        print(f"Legitimate samples: {total_legitimate} ({total_legitimate/len(enhanced_df)*100:.1f}%)")
    
    print(f"\nSource distribution:")
    if 'source' in enhanced_df.columns:
        source_counts = enhanced_df['source'].value_counts()
        for source, count in source_counts.items():
            print(f"  {source}: {count} samples")
    
    print(f"\nTop 15 features:")
    for i, feature in enumerate(feature_names[:15]):
        print(f"  {i+1:2d}. {feature}")
    
    if len(feature_names) > 15:
        print(f"  ... and {len(feature_names) - 15} more features")
    
    print(f"\nCleaned data saved to: data/processed/cleaned_phishing_data.csv")
    
    return processor, enhanced_df, (X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()