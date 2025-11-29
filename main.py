
#!/usr/bin/env python3
"""
Updated Phishing Detection Project - Main Execution Script
=========================================================

This script orchestrates the complete phishing detection pipeline with improved
error handling and fallback mechanisms:
1. Data preprocessing and cleaning
2. Model training and evaluation
3. Ensemble model creation
4. Results comparison and output

Author: Updated Version
Date: June 2025
"""

import os
import sys
import logging
import time
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.mixture import GaussianMixture
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression

# Project configuration
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def setup_logging():
    """Setup comprehensive logging configuration"""
    log_dir = project_root / "outputs"
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"phishing_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

class EnhancedPhishingPreprocessor:
    """Enhanced preprocessing class with robust error handling"""
    
    def __init__(self, data_path=None):
        self.data_path = Path(data_path) if data_path else project_root / "data" / "raw"
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.processed_data_path = project_root / "data" / "processed"
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
    
    def find_data_files(self):
        """Find all CSV files in the data directory"""
        csv_files = []
        
        # Check multiple possible locations
        possible_paths = [
            self.data_path,
            project_root / "data",
            project_root,
            Path.cwd()
        ]
        
        for path in possible_paths:
            if path.exists():
                files = list(path.glob("*.csv"))
                if files:
                    csv_files.extend(files)
                    break
        
        return csv_files
    
    def load_datasets(self):
        """Load all available CSV datasets"""
        csv_files = self.find_data_files()
        
        if not csv_files:
            print("❌ No CSV files found in any expected location")
            print("Expected locations:")
            print(f"  - {self.data_path}")
            print(f"  - {project_root / 'data'}")
            print(f"  - {project_root}")
            return []
        
        datasets = []
        print(f"📂 Found {len(csv_files)} CSV files:")
        
        for file_path in csv_files:
            try:
                print(f"  Loading: {file_path.name}")
                df = pd.read_csv(file_path)
                datasets.append({
                    'name': file_path.stem,
                    'path': file_path,
                    'data': df,
                    'shape': df.shape
                })
                print(f"    ✅ Loaded: {df.shape}")
            except Exception as e:
                print(f"    ❌ Failed to load {file_path.name}: {str(e)}")
        
        return datasets
    
    def standardize_column_names(self, df):
        """Standardize column names across different datasets"""
        # Common column name mappings
        column_mappings = {
            # Target columns
            'class': 'label',
            'class_label': 'label',
            'result': 'label',
            'phishing': 'label',
            'target': 'label',
            'classification': 'label',
            
            # URL columns
            'website': 'url',
            'domain': 'url',
            'site': 'url',
            'link': 'url',
        }
        
        # Apply mappings
        df_renamed = df.copy()
        df_renamed.columns = df_renamed.columns.str.lower().str.strip()
        
        for old_name, new_name in column_mappings.items():
            if old_name in df_renamed.columns:
                df_renamed = df_renamed.rename(columns={old_name: new_name})
        
        return df_renamed
    
    def clean_and_convert_data(self, df):
        """Comprehensive data cleaning and type conversion"""
        print(f"🧹 Cleaning dataset with shape: {df.shape}")
        
        df_clean = df.copy()
        
        # Standardize column names
        df_clean = self.standardize_column_names(df_clean)
        
        # Identify and handle different column types
        exclude_cols = ['url', 'domain', 'website', 'source', 'link']
        
        for col in df_clean.columns:
            if col.lower() in exclude_cols:
                continue
            
            if df_clean[col].dtype == 'object':
                print(f"  Processing column: {col}")
                
                # Try numeric conversion first
                try:
                    # Clean string values
                    df_clean[col] = df_clean[col].astype(str).str.strip()
                    
                    # Handle common problematic values
                    replacements = {
                        'nan': np.nan, 'NaN': np.nan, 'null': np.nan,
                        'NULL': np.nan, 'None': np.nan, '': np.nan,
                        'inf': np.inf, '-inf': -np.inf, 'infinity': np.inf
                    }
                    df_clean[col] = df_clean[col].replace(replacements)
                    
                    # Convert to numeric
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                    print(f"    ✅ Converted to numeric")
                    
                except Exception as e:
                    # Try label encoding for categorical data
                    try:
                        unique_vals = df_clean[col].nunique()
                        if unique_vals <= 50:  # Reasonable threshold for categorical
                            le = LabelEncoder()
                            df_clean[col] = le.fit_transform(df_clean[col].fillna('unknown'))
                            self.label_encoders[col] = le
                            print(f"    ✅ Label encoded ({unique_vals} categories)")
                        else:
                            print(f"    ⚠️ Too many unique values ({unique_vals}), dropping")
                            df_clean = df_clean.drop(columns=[col])
                    except Exception as e:
                        print(f"    ❌ Could not process, dropping column: {str(e)}")
                        df_clean = df_clean.drop(columns=[col])
        
        return df_clean
    
    def handle_missing_and_infinite(self, df):
        """Handle missing values and infinite values"""
        print(f"🔧 Handling missing values...")
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # Handle infinite values
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            
            # Fill missing values
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                fill_value = df[col].median()
                df[col] = df[col].fillna(fill_value)
                print(f"  Filled {missing_count} missing values in {col}")
        
        return df
    
    def combine_datasets(self, datasets):
        """Combine multiple datasets using union of all columns, filling missing columns with default values"""
        if not datasets:
            raise ValueError("No datasets provided")
        if len(datasets) == 1:
            print("📋 Single dataset found, processing...")
            return self.clean_and_convert_data(datasets[0]['data'])

        print(f"🔗 Combining {len(datasets)} datasets (union of all columns)...")

        # Clean each dataset
        cleaned_datasets = []
        for dataset in datasets:
            try:
                cleaned_df = self.clean_and_convert_data(dataset['data'])
                cleaned_datasets.append({
                    'name': dataset['name'],
                    'data': cleaned_df
                })
                print(f"  ✅ Cleaned {dataset['name']}: {cleaned_df.shape}")
            except Exception as e:
                print(f"  ❌ Failed to clean {dataset['name']}: {str(e)}")

        if not cleaned_datasets:
            raise ValueError("No datasets could be cleaned successfully")

        # Get union of all columns
        all_columns = set()
        for ds in cleaned_datasets:
            all_columns.update(ds['data'].columns)

        # Reindex each DataFrame to have all columns, filling missing with 0
        aligned_dfs = []
        for ds in cleaned_datasets:
            aligned_df = ds['data'].reindex(columns=all_columns, fill_value=0)
            aligned_dfs.append(aligned_df)

        combined_df = pd.concat(aligned_dfs, ignore_index=True, sort=False)
        print(f"✅ Combined dataset shape: {combined_df.shape}")
        return combined_df
    
    def feature_engineering(self, df):
        """Apply feature engineering techniques"""
        print(f"⚙️ Applying feature engineering...")
        
        df_enhanced = df.copy()
        
        # Handle missing values and infinite values
        df_enhanced = self.handle_missing_and_infinite(df_enhanced)
        
        # Basic feature engineering
        numeric_cols = df_enhanced.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target column from feature engineering
        target_candidates = ['label', 'class_label', 'result', 'target', 'phishing']
        for candidate in target_candidates:
            if candidate in numeric_cols:
                numeric_cols.remove(candidate)
        
        # Add some basic derived features if we have enough columns
        if len(numeric_cols) >= 3:
            try:
                # Add feature combinations (only if we have room)
                if len(df_enhanced.columns) < 100:  # Avoid creating too many features
                    feature_pairs = list(zip(numeric_cols[:3], numeric_cols[1:4]))
                    for col1, col2 in feature_pairs:
                        try:
                            df_enhanced[f'{col1}_{col2}_ratio'] = df_enhanced[col1] / (df_enhanced[col2] + 1e-8)
                            df_enhanced[f'{col1}_{col2}_diff'] = df_enhanced[col1] - df_enhanced[col2]
                        except:
                            continue
                    print(f"  ✅ Added feature combinations")
            except Exception as e:
                print(f"  ⚠️ Feature engineering partially failed: {str(e)}")
        
        # Final cleanup
        df_enhanced = self.handle_missing_and_infinite(df_enhanced)
        
        print(f"✅ Enhanced dataset shape: {df_enhanced.shape}")
        return df_enhanced

    def correlation_feature_selection(self, df, threshold=0.8):
        """Remove features with high correlation above the threshold"""
        print(f"🔍 Performing correlation-based feature selection with threshold {threshold}...")
        corr_matrix = df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        print(f"  Dropping {len(to_drop)} features due to high correlation: {to_drop}")
        df_reduced = df.drop(columns=to_drop)
        print(f"✅ Dataset shape after correlation feature selection: {df_reduced.shape}")
        return df_reduced
    
    def save_processed_data(self, df, filename='processed_phishing_data.csv'):
        """Save processed data to file"""
        output_path = self.processed_data_path / filename
        df.to_csv(output_path, index=False)
        print(f"💾 Processed data saved to: {output_path}")
        return str(output_path)

class ComprehensiveModelTrainer:
    """Enhanced model trainer with multiple algorithms"""
    
    def __init__(self):
        self.models = {
            'Naive Bayes': GaussianNB(),
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42, max_depth=5),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        }
        self.results = {}
        self.best_model = None
        self.best_model_name = None
    
    def prepare_data(self, df):
        """Prepare features and target from dataframe"""
        print(f"🎯 Preparing data for training...")
        
        # Find target column
        target_candidates = ['label', 'class_label', 'result', 'target', 'phishing']
        target_col = None
        
        for candidate in target_candidates:
            if candidate in df.columns:
                target_col = candidate
                break
        
        if target_col is None:
            raise ValueError(f"No target column found. Available columns: {list(df.columns)}")
        
        # Prepare target
        y = df[target_col].copy()
        
        # Standardize target to 0/1
        unique_vals = sorted(y.unique())
        print(f"📊 Found target column '{target_col}' with {len(unique_vals)} unique values: {unique_vals}")
        
        if len(unique_vals) == 2:
            if set(unique_vals) == {0, 1}:
                pass  # Already binary
            elif set(unique_vals) == {-1, 1}:
                y = y.map({-1: 0, 1: 1})
                print("✅ Mapped {-1: 0, 1: 1}")
            else:
                y = y.map({unique_vals[0]: 0, unique_vals[1]: 1})
                print(f"✅ Mapped {{{unique_vals[0]}: 0, {unique_vals[1]}: 1}}")
        elif len(unique_vals) == 3:
            print(f"📊 Target column has 3 values: {unique_vals}")
            print(f"📊 Value counts:\n{y.value_counts()}")
            # Convert 3-class to binary (map first value to 0, others to 1)
            y = y.map({unique_vals[0]: 0, unique_vals[1]: 1, unique_vals[2]: 1})
            print(f"✅ Converted 3-class to binary: {{{unique_vals[0]}: 0, {unique_vals[1]}: 1, {unique_vals[2]}: 1}}")
        else:
            raise ValueError(f"Target column has {len(unique_vals)} unique values, expected 2 or 3")
        
        # Prepare features
        exclude_cols = [target_col]
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['url', 'domain', 'website', 'link', 'source']):
                exclude_cols.append(col)

        print(f"🔍 Dynamically excluded columns: {exclude_cols}")
        print(f"🔍 Total columns: {len(df.columns)}")
        print(f"🔍 All columns: {list(df.columns)}")
        print(f"🔍 Target column: {target_col}")
        print(f"🔍 Exclude columns: {exclude_cols}")

        feature_cols = [col for col in df.columns if col not in exclude_cols]
        print(f"🔍 Feature columns after exclusion: {len(feature_cols)} columns")
        print(f"🔍 Feature columns: {feature_cols[:10]}...")

        if not feature_cols:
            raise ValueError("No feature columns remaining after exclusion!")

        X = df[feature_cols].copy()
        print(f"🔍 X shape before numeric filtering: {X.shape}")
        print(f"🔍 X dtypes:\n{X.dtypes.value_counts()}")
        
        # Ensure all features are numeric
        non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric_cols:
            print(f"⚠️ Dropping non-numeric columns: {non_numeric_cols}")
            X = X.drop(columns=non_numeric_cols)
        
        # Handle any remaining missing values
        if X.isnull().any().any():
            print("🔧 Handling missing values in features...")
            X = X.fillna(X.median())
        
        print(f"✅ Prepared data: {X.shape[1]} features, {len(y)} samples")
        print(f"✅ Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def train_and_evaluate_models(self, X, y):
        """Train and evaluate all models with extended metrics"""
        print(f"\n🤖 Training and evaluating models...")
        print("=" * 60)

        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        scaler = StandardScaler()
        if X_train.empty or X_train.shape[1] == 0:
            raise ValueError(f"X_train is empty or has no features. Shape: {X_train.shape}")

        print(f"📊 Training data shape: {X_train.shape}")
        print(f"📊 Training data dtypes:\n{X_train.dtypes}")

        # Check for any remaining non-numeric columns
        non_numeric_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric_cols:
            print(f"⚠️ Found non-numeric columns in training data: {non_numeric_cols}")
            X_train = X_train.select_dtypes(include=[np.number])
            X_test = X_test.select_dtypes(include=[np.number])
            print(f"📊 After removing non-numeric: {X_train.shape}")

        # Scale the features
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print(f"📊 Training set: {X_train_scaled.shape}")
        print(f"📊 Testing set: {X_test_scaled.shape}")

        best_accuracy = 0

        for name, model in self.models.items():
            print(f"\n🔄 Training {name}...")
            start_time = time.time()

            try:
                # Train model
                model.fit(X_train_scaled, y_train)

                # Make predictions
                y_pred = model.predict(X_test_scaled)

                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                try:
                    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])
                except Exception:
                    # For models without predict_proba
                    try:
                        roc_auc = roc_auc_score(y_test, y_pred)
                    except Exception:
                        roc_auc = 0.0

                # Cross-validation
                try:
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
                    cv_mean = cv_scores.mean()
                    cv_std = cv_scores.std()
                except:
                    cv_mean = accuracy
                    cv_std = 0

                # Store results
                self.results[name] = {
                    'model': model,
                    'scaler': scaler,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'roc_auc': roc_auc,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std,
                    'predictions': y_pred,
                    'training_time': time.time() - start_time,
                    'status': 'SUCCESS'
                }

                # Track best model
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    self.best_model = model
                    self.best_model_name = name

                print(f"  ✅ Accuracy: {accuracy:.4f}")
                print(f"  ✅ Precision: {precision:.4f}")
                print(f"  ✅ Recall: {recall:.4f}")
                print(f"  ✅ F1 Score: {f1:.4f}")
                print(f"  ✅ ROC AUC: {roc_auc:.4f}")
                print(f"  ✅ CV Score: {cv_mean:.4f} (±{cv_std:.4f})")
                print(f"  ✅ Time: {time.time() - start_time:.2f}s")

            except Exception as e:
                print(f"  ❌ Error: {str(e)}")
                self.results[name] = {
                    'error': str(e),
                    'status': 'FAILED',
                    'training_time': time.time() - start_time
                }

        return X_test, y_test
    
    def print_detailed_results(self, y_test):
        """Print comprehensive results with five metrics"""
        print(f"\n📈 DETAILED RESULTS:")
        print("=" * 60)

        successful_models = []

        for name, result in self.results.items():
            if result['status'] == 'FAILED':
                print(f"\n❌ {name}: FAILED")
                print(f"   Error: {result['error']}")
                continue

            successful_models.append((name, result['accuracy']))

            print(f"\n🎯 {name.upper()}:")
            print(f"   Accuracy: {result['accuracy']:.4f}")
            print(f"   Precision: {result['precision']:.4f}")
            print(f"   Recall: {result['recall']:.4f}")
            print(f"   F1 Score: {result['f1']:.4f}")
            print(f"   ROC AUC: {result['roc_auc']:.4f}")
            print(f"   CV Score: {result['cv_mean']:.4f} (±{result['cv_std']:.4f})")
            print(f"   Training Time: {result['training_time']:.2f}s")

        # Best model summary
        if successful_models:
            successful_models.sort(key=lambda x: x[1], reverse=True)
            print(f"\n🏆 MODEL RANKING:")
            for i, (name, accuracy) in enumerate(successful_models, 1):
                print(f"   {i}. {name}: {accuracy:.4f}")

def create_project_structure():
    """Create necessary project directories"""
    directories = [
        "data/raw",
        "data/processed", 
        "models",
        "utils",
        "outputs"
    ]
    
    for dir_path in directories:
        (project_root / dir_path).mkdir(parents=True, exist_ok=True)

def generate_comprehensive_report(preprocessor, trainer, logger):
    """Generate detailed execution report"""
    try:
        output_dir = project_root / "outputs"
        report_file = output_dir / f"comprehensive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(report_file, 'w') as f:
            f.write("COMPREHENSIVE PHISHING DETECTION REPORT\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Data Processing Summary
            f.write("DATA PROCESSING SUMMARY\n")
            f.write("-" * 25 + "\n")
            csv_files = preprocessor.find_data_files()
            f.write(f"Data Files Found: {len(csv_files)}\n")
            for file_path in csv_files:
                f.write(f"  - {file_path.name}\n")
            f.write(f"Feature Encoders Used: {len(preprocessor.label_encoders)}\n\n")
            
            # Model Results
            f.write("MODEL EVALUATION RESULTS\n")
            f.write("-" * 30 + "\n")
            
            successful = 0
            failed = 0
            
            for name, result in trainer.results.items():
                if result['status'] == 'SUCCESS':
                    successful += 1
                    f.write(f"{name}:\n")
                    f.write(f"  Accuracy: {result['accuracy']:.4f}\n")
                    f.write(f"  CV Score: {result['cv_mean']:.4f} (±{result['cv_std']:.4f})\n")
                    f.write(f"  Training Time: {result['training_time']:.2f}s\n\n")
                else:
                    failed += 1
                    f.write(f"{name}: FAILED - {result['error']}\n\n")
            
            # Summary
            f.write("EXECUTION SUMMARY\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Models: {len(trainer.results)}\n")
            f.write(f"Successful: {successful}\n")
            f.write(f"Failed: {failed}\n")
            f.write(f"Success Rate: {(successful/len(trainer.results)*100) if trainer.results else 0:.1f}%\n")
            
            if trainer.best_model_name:
                f.write(f"Best Model: {trainer.best_model_name}\n")
                best_result = trainer.results[trainer.best_model_name]
                f.write(f"Best Accuracy: {best_result['accuracy']:.4f}\n")
        
        logger.info(f"Comprehensive report saved: {report_file}")
        
    except Exception as e:
        logger.error(f"Failed to generate report: {str(e)}")

def main():
    """Enhanced main execution function with longevity analysis integration"""
    print("ENHANCED PHISHING DETECTION PIPELINE")
    print("=" * 55)

    # Setup
    logger = setup_logging()
    create_project_structure()

    logger.info("Starting enhanced phishing detection pipeline")

    try:
        # Initialize components
        print("\nInitializing components...")
        preprocessor = EnhancedPhishingPreprocessor()
        trainer = ComprehensiveModelTrainer()

        # Step 1: Load and preprocess data
        print("\nStep 1: Data Loading and Preprocessing")
        print("-" * 45)

        datasets = preprocessor.load_datasets()
        if not datasets:
            logger.error("❌ No data files found. Please add CSV files to the data directory.")
            sys.exit(1)

        # Combine and process datasets
        combined_df = preprocessor.combine_datasets(datasets)
        enhanced_df = preprocessor.feature_engineering(combined_df)
        enhanced_df = preprocessor.correlation_feature_selection(enhanced_df, threshold=0.8)

        # Save processed data
        processed_path = preprocessor.save_processed_data(enhanced_df)

        # Step 2: Correlation Analysis
        print("\nStep 2: Correlation Analysis")
        print("-" * 45)
        # Compute correlation matrix for numeric features
        import numpy as np
        feature_df = enhanced_df.select_dtypes(include=[np.number])
        corr_methods = ['pearson', 'spearman', 'kendall']

        # --- Enhanced Correlation Visualization ---
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.patches import Rectangle

        for method in corr_methods:
            corr_matrix = feature_df.corr(method=method)
            print(f"\nFeature Correlation Matrix ({method.title()}, sample):")
            print(corr_matrix.head(10).iloc[:, :10].round(2))

            # Identify highly correlated pairs
            threshold = 0.8
            high_corr = []
            for i, col1 in enumerate(corr_matrix.columns):
                for j, col2 in enumerate(corr_matrix.columns):
                    if i < j:
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > threshold:
                            high_corr.append((col1, col2, corr_val))
            if high_corr:
                print(f"Highly correlated feature pairs (|corr| > {threshold}) [{method.title()}]:")
                for f1, f2, val in high_corr:
                    print(f"  {f1} <-> {f2}: {val:.2f}")
            else:
                print(f"No highly correlated feature pairs found above threshold for {method.title()}.")

            # Enhanced Visualization: Heatmap with highlights
            try:
                plt.figure(figsize=(12, 10))
                ax = sns.heatmap(
                    corr_matrix,
                    cmap='coolwarm',
                    center=0,
                    annot=True,
                    fmt='.2f',
                    cbar=True,
                    square=True,
                    linewidths=0.5,
                    linecolor='gray',
                    annot_kws={"size": 8}
                )
                plt.title(f'Feature Correlation Heatmap ({method.title()})', fontsize=16)
                plt.tight_layout()
                # Highlight highly correlated pairs
                for f1, f2, val in high_corr:
                    i = list(corr_matrix.columns).index(f1)
                    j = list(corr_matrix.columns).index(f2)
                    ax.add_patch(Rectangle((j, i), 1, 1, fill=False, edgecolor='yellow', lw=2))
                    ax.add_patch(Rectangle((i, j), 1, 1, fill=False, edgecolor='yellow', lw=2))
                out_path = f'outputs/feature_correlation_heatmap_{method}_enhanced.png'
                plt.savefig(out_path)
                plt.close()
                print(f'Enhanced feature correlation heatmap saved to {out_path}.')
            except Exception as e:
                print(f'Enhanced correlation heatmap visualization failed for {method.title()}: {e}')

        # Improved Bar Graph Visualization: Top absolute correlations (Pearson)
        try:
            pearson_corr = feature_df.corr(method='pearson')
            abs_corr = pearson_corr.abs()
            upper = abs_corr.where(np.triu(np.ones(abs_corr.shape), k=1).astype(bool))
            sorted_pairs = upper.unstack().dropna().sort_values(ascending=False)
            top_n = 15
            top_corr = sorted_pairs.head(top_n)
            # Prepare labels as 'Feature1 vs Feature2' for clarity
            bar_labels = [f'{idx[0]} vs {idx[1]}' for idx in top_corr.index]
            plt.figure(figsize=(12, 7))
            bars = plt.bar(bar_labels, top_corr.values, color='mediumseagreen', edgecolor='black')
            plt.title(f'Top {top_n} Absolute Feature Correlations (Pearson)', fontsize=15)
            plt.ylabel('Absolute Correlation', fontsize=12)
            plt.xlabel('Feature Pair', fontsize=12)
            plt.ylim(0, 1)
            plt.xticks(rotation=40, ha='right', fontsize=10)
            # Annotate bars with correlation values
            for bar, val in zip(bars, top_corr.values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.2f}', ha='center', va='bottom', fontsize=9)
            plt.tight_layout()
            bar_path = 'outputs/feature_correlation_bargraph_top.png'
            plt.savefig(bar_path)
            plt.close()
            print(f'Bar graph of top absolute feature correlations saved to {bar_path}.')
        except Exception as e:
            print(f'Bar graph correlation visualization failed: {e}')

        # === GRAPH MAP: Feature Correlation Network Graph ===
        try:
            import networkx as nx
            corr_matrix = feature_df.corr(method='pearson')
            threshold = 0.7  # Only show strong correlations
            G = nx.Graph()
            # Add nodes
            for col in corr_matrix.columns:
                G.add_node(col)
            # Add edges for high correlations
            for i, col1 in enumerate(corr_matrix.columns):
                for j, col2 in enumerate(corr_matrix.columns):
                    if i < j:
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > threshold:
                            G.add_edge(col1, col2, weight=abs(corr_val))
            if G.number_of_edges() > 0:
                plt.figure(figsize=(14, 10))
                pos = nx.spring_layout(G, seed=42)
                edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
                nx.draw_networkx_nodes(G, pos, node_size=700, node_color='skyblue')
                nx.draw_networkx_edges(G, pos, width=[2*w for w in edge_weights], edge_color='orange')
                nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
                plt.title('Feature Correlation Graph Map (|corr| > 0.7)', fontsize=16)
                plt.axis('off')
                plt.tight_layout()
                graph_map_path = 'outputs/feature_correlation_graph_map.png'
                plt.savefig(graph_map_path)
                print(f'Feature correlation graph map saved to {graph_map_path}.')
                plt.show()  # Show the graph interactively
                plt.close()
            else:
                print('No strong feature correlations to display in graph map.')
        except ImportError:
            print('NetworkX is not installed. Skipping graph map. To enable, install networkx.')
        except Exception as e:
            print(f'Graph map visualization failed: {e}')

        # === GRAPH MAP: Code Structure Network Graph ===
        try:
            import networkx as nx
            import inspect
            import types
            code_graph = nx.DiGraph()
            # Add main classes and functions as nodes
            code_graph.add_node('main', type='function')
            code_graph.add_node('EnhancedPhishingPreprocessor', type='class')
            code_graph.add_node('ComprehensiveModelTrainer', type='class')
            code_graph.add_node('setup_logging', type='function')
            code_graph.add_node('create_project_structure', type='function')
            code_graph.add_node('generate_comprehensive_report', type='function')
            # Add relationships (calls/uses)
            code_graph.add_edge('main', 'setup_logging', label='calls')
            code_graph.add_edge('main', 'create_project_structure', label='calls')
            code_graph.add_edge('main', 'EnhancedPhishingPreprocessor', label='instantiates')
            code_graph.add_edge('main', 'ComprehensiveModelTrainer', label='instantiates')
            code_graph.add_edge('main', 'generate_comprehensive_report', label='calls')
            code_graph.add_edge('main', 'feature_engineering', label='calls')
            code_graph.add_edge('main', 'train_and_evaluate_models', label='calls')
            code_graph.add_edge('main', 'print_detailed_results', label='calls')
            code_graph.add_edge('main', 'save_processed_data', label='calls')
            code_graph.add_edge('main', 'combine_datasets', label='calls')
            code_graph.add_edge('main', 'load_datasets', label='calls')
            code_graph.add_edge('main', 'find_data_files', label='calls')
            code_graph.add_edge('main', 'handle_missing_and_infinite', label='calls')
            code_graph.add_edge('main', 'standardize_column_names', label='calls')
            code_graph.add_edge('main', 'clean_and_convert_data', label='calls')
            code_graph.add_edge('main', 'prepare_data', label='calls')
            code_graph.add_edge('main', 'print_detailed_results', label='calls')
            code_graph.add_edge('main', 'train_and_evaluate_models', label='calls')
            # Visualize the code structure graph
            plt.figure(figsize=(14, 8))
            pos = nx.spring_layout(code_graph, seed=42)
            node_colors = ['lightgreen' if code_graph.nodes[n]['type']=='class' else 'lightblue' for n in code_graph.nodes]
            nx.draw_networkx_nodes(code_graph, pos, node_color=node_colors, node_size=1200)
            nx.draw_networkx_labels(code_graph, pos, font_size=10, font_weight='bold')
            nx.draw_networkx_edges(code_graph, pos, arrowstyle='-|>', arrowsize=20, edge_color='gray')
            edge_labels = nx.get_edge_attributes(code_graph, 'label')
            nx.draw_networkx_edge_labels(code_graph, pos, edge_labels=edge_labels, font_color='red', font_size=8)
            plt.title('Code Structure Graph Map (Main Classes & Functions)', fontsize=16)
            plt.axis('off')
            plt.tight_layout()
            code_graph_path = 'outputs/code_structure_graph_map.png'
            plt.savefig(code_graph_path)
            print(f'Code structure graph map saved to {code_graph_path}.')
            plt.show()
            plt.close()
        except Exception as e:
            print(f'Code structure graph map visualization failed: {e}')

        # Step 3: Model training and evaluation
        print("\nStep 3: Model Training and Evaluation")
        print("-" * 45)

        X, y = trainer.prepare_data(enhanced_df)
        X_test, y_test = trainer.train_and_evaluate_models(X, y)

        # === Longevity Analysis: Feature Drift Detection ===
        print("\nStep 3b: Longevity Analysis (Feature Drift Detection & Model Degradation Monitoring)")
        print("-" * 60)
        from utils.longevity_analysis import LongevityAnalyzer
        feature_names = list(X.columns)
        longevity = LongevityAnalyzer(baseline_df=X, feature_names=feature_names)

        # Simulate tracking on test set (in real use, use new/temporal data)
        kl_results, psi_results = longevity.track_feature_distributions(X_test)
        drifted_features = longevity.detect_drift(kl_threshold=0.1, psi_threshold=0.1)
        importances = longevity.track_feature_relevance(trainer.best_model)

        # KL Divergence: Measures distribution changes
        print("\n[Longevity] KL Divergence (sample, first 5 features):")
        for k, v in list(kl_results.items())[:5]:
            print(f"  {k}: {v:.4f}")

        # PSI: Population Stability Index
        print("[Longevity] PSI (sample, first 5 features):")
        for k, v in list(psi_results.items())[:5]:
            print(f"  {k}: {v:.4f}")

        # Feature Importance Tracking
        print("[Longevity] Feature Importance Tracking (sample, first 5 features):")
        for k, v in list(importances.items())[:5]:
            print(f"  {k}: {v:.4f}")

        # Automated Alerts: Flag significant drift
        if drifted_features:
            print(f"[Longevity] 🚨 ALERT: Features with significant drift detected: {drifted_features}")
        else:
            print("[Longevity] No significant drift detected.")

        # Step 3: Results analysis
        print("\nStep 3: Results Analysis")
        print("-" * 45)
        trainer.print_detailed_results(y_test)

        # Step 4: Visualization
        print("\nStep 4: Visualization")
        print("-" * 45)
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
            model_names = []
            metric_values = {m: [] for m in metrics}
            for name, result in trainer.results.items():
                if result.get('status') == 'SUCCESS':
                    model_names.append(name)
                    for m in metrics:
                        metric_values[m].append(result.get(m, 0))
            if model_names:
                # Bar graph for each metric (grouped by model)
                for m in metrics:
                    plt.figure(figsize=(8, 5))
                    bars = plt.bar(model_names, metric_values[m], color='cornflowerblue', edgecolor='black')
                    plt.title(f'{m.capitalize()} by Model', fontsize=14)
                    plt.xlabel('Model', fontsize=12)
                    plt.ylabel(m.capitalize(), fontsize=12)
                    plt.ylim(0, 1)
                    plt.xticks(rotation=20, ha='right', fontsize=10)
                    # Annotate bars
                    for bar, val in zip(bars, metric_values[m]):
                        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.2f}', ha='center', va='bottom', fontsize=9)
                    plt.tight_layout()
                    bar_path = f'outputs/model_{m}_bargraph.png'
                    plt.savefig(bar_path)
                    plt.close()
                    print(f'Bar graph for {m} saved to {bar_path}.')

                # Combined grouped bar graph for all metrics
                import numpy as np
                x = np.arange(len(model_names))
                width = 0.15
                plt.figure(figsize=(12, 6))
                for idx, m in enumerate(metrics):
                    plt.bar(x + idx*width, metric_values[m], width, label=m.capitalize())
                plt.xlabel('Model', fontsize=12)
                plt.ylabel('Score', fontsize=12)
                plt.title('Model Performance Metrics (Grouped Bar Chart)', fontsize=15)
                plt.xticks(x + width*2, model_names, rotation=20, ha='right', fontsize=10)
                plt.ylim(0, 1)
                plt.legend()
                plt.tight_layout()
                grouped_bar_path = 'outputs/model_performance_grouped_bargraph.png'
                plt.savefig(grouped_bar_path)
                plt.close()
                print(f'Grouped bar graph for all metrics saved to {grouped_bar_path}.')
            else:
                print('No successful models to visualize.')
        except Exception as e:
            print(f'Visualization failed: {e}')

        # Step 5: Generate report
        print("\nStep 5: Report Generation")
        print("-" * 45)
        generate_comprehensive_report(preprocessor, trainer, logger)

        # === INTERPRETABILITY FRAMEWORK ===
        print("\nStep 6: Interpretability Framework (SHAP, Correlation, Probability Path, Risk Decomposition)")
        print("-" * 60)
        import shap
        from collections import defaultdict

        # Helper: Correlation matrix for features
        corr_matrix = feature_df.corr(method='pearson')
        # Helper: Correlation with target (for LR scaling)
        y_corr = pd.Series({col: np.corrcoef(X[col], y)[0,1] if X[col].std() > 0 else 0 for col in X.columns})

        # --- SHAP Feature Importance (with correlation-weighted adjustment) ---
        def get_shap_values(model, X_sample, model_type):
            if model_type in ['Random Forest', 'Decision Tree', 'Gradient Boosting']:
                explainer = shap.TreeExplainer(model)
            elif model_type == 'Logistic Regression':
                explainer = shap.LinearExplainer(model, X_sample, feature_dependence="independent")
            else:
                explainer = shap.Explainer(model, X_sample)
            shap_values = explainer.shap_values(X_sample)
            # For binary, shap_values is a list
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            return shap_values

        # Use a sample for SHAP (speed)
        X_shap = X_test.copy()
        if len(X_shap) > 200:
            X_shap = X_shap.sample(200, random_state=42)

        interpret_results = {}
        for name, result in trainer.results.items():
            if result.get('status') != 'SUCCESS':
                continue
            model = result['model']
            scaler = result['scaler']
            X_scaled = scaler.transform(X_shap)
            X_shap_df = pd.DataFrame(X_scaled, columns=X_shap.columns)
            model_type = None
            if 'Random Forest' in name:
                model_type = 'Random Forest'
            elif 'Decision Tree' in name:
                model_type = 'Decision Tree'
            elif 'Gradient Boosting' in name:
                model_type = 'Gradient Boosting'
            elif 'Logistic Regression' in name:
                model_type = 'Logistic Regression'
            else:
                model_type = 'Other'

            # SHAP values
            try:
                shap_vals = get_shap_values(model, X_shap_df, model_type)
                mean_abs_shap = np.abs(shap_vals).mean(axis=0)
                # Correlation-weighted adjustment
                if model_type in ['Random Forest', 'Decision Tree', 'Gradient Boosting']:
                    # Penalize by mean absolute correlation with other features
                    corr_penalty = np.array([1 - np.mean(np.abs(corr_matrix[col].drop(col))) for col in X_shap.columns])
                    adj_importance = mean_abs_shap * corr_penalty
                elif model_type == 'Logistic Regression':
                    # Scale by correlation with target
                    coefs = np.abs(model.coef_[0])
                    target_corr = np.abs(y_corr[X_shap.columns].values)
                    adj_importance = coefs * target_corr
                else:
                    adj_importance = mean_abs_shap
                # Normalize
                adj_importance = adj_importance / (adj_importance.sum() + 1e-8)
                interpret_results[name] = {
                    'shap': dict(zip(X_shap.columns, mean_abs_shap)),
                    'adj_importance': dict(zip(X_shap.columns, adj_importance)),
                }
            except Exception as e:
                interpret_results[name] = {'error': str(e)}

        # --- Probability Path Visualization (terminal output) ---
        def probability_path(model, X_row, feature_names):
            # For tree-based: use SHAP, for LR: use coef*value, else fallback
            try:
                if hasattr(model, 'predict_proba'):
                    base = model.predict_proba([X_row])[0][0]
                else:
                    base = model.decision_function([X_row])[0]
            except:
                base = 0.5
            contribs = []
            if hasattr(model, 'feature_importances_'):
                # Tree-based: use SHAP
                explainer = shap.TreeExplainer(model)
                # Ensure input is a 2D numpy array
                X_row_2d = np.array(X_row).reshape(1, -1)
                shap_vals = explainer.shap_values(X_row_2d)
                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
                for i, f in enumerate(feature_names):
                    contribs.append((f, shap_vals[0, i]))
            elif hasattr(model, 'coef_'):
                # LR: coef * value
                for i, f in enumerate(feature_names):
                    contribs.append((f, model.coef_[0][i] * X_row[i]))
            else:
                # Fallback: difference from mean
                for i, f in enumerate(feature_names):
                    contribs.append((f, X_row[i] - np.mean(X_row)))
            return base, contribs

        # --- Risk Factor Decomposition ---
        # Cluster features by prefix (e.g., url_, content_, network_)
        def cluster_features(features):
            clusters = defaultdict(list)
            for f in features:
                if '_' in f:
                    prefix = f.split('_')[0]
                else:
                    prefix = 'other'
                clusters[prefix].append(f)
            return clusters

        # Compute cluster weights from correlation matrix
        def cluster_weights(clusters, corr_matrix):
            weights = {}
            total = 0
            for cname, feats in clusters.items():
                # Mean absolute correlation within cluster
                if len(feats) == 1:
                    w = 1.0
                else:
                    w = 1 - np.mean([np.abs(corr_matrix.loc[f1, f2]) for f1 in feats for f2 in feats if f1 != f2])
                weights[cname] = max(w, 0.01)
                total += weights[cname]
            # Normalize
            for cname in weights:
                weights[cname] /= (total + 1e-8)
            return weights

        # Print interpretability results for best model with diagnostics and NaN handling
        if trainer.best_model_name and trainer.best_model_name in interpret_results:
            print(f"\n[INTERPRETABILITY] Best Model: {trainer.best_model_name}")
            res = interpret_results[trainer.best_model_name]
            if 'error' in res:
                print(f"  SHAP/importance error: {res['error']}")
            else:
                # Diagnostics: Check for NaNs in SHAP and adjusted importance
                mean_abs_shap = np.array(list(res['shap'].values()))
                adj_importance = np.array(list(res['adj_importance'].values()))
                if np.isnan(mean_abs_shap).any():
                    print("  [Diagnostics] Warning: NaN values found in mean_abs_shap.")
                    nan_feats = [f for f, v in res['shap'].items() if np.isnan(v)]
                    print(f"    Features with NaN SHAP: {nan_feats}")
                if np.isnan(adj_importance).any():
                    print("  [Diagnostics] Warning: NaN values found in adjusted importance.")
                    nan_feats = [f for f, v in res['adj_importance'].items() if np.isnan(v)]
                    print(f"    Features with NaN adjusted importance: {nan_feats}")

                # Top features by adjusted importance, skipping NaNs
                print("  Top 10 Features by Adjusted Importance (excluding NaN):")
                sorted_feats = [(f, v) for f, v in sorted(res['adj_importance'].items(), key=lambda x: x[1] if not np.isnan(x[1]) else -np.inf, reverse=True) if not np.isnan(v)][:10]
                if not sorted_feats:
                    print("    [Diagnostics] All top features have NaN importance.")
                for f, v in sorted_feats:
                    print(f"    {f}: {v:.4f}")

                # Probability path for a random test sample
                X_row = X_shap_df.iloc[0].values
                base, contribs = probability_path(trainer.best_model, X_row, X_shap_df.columns)
                print("\n  Probability Path (first test sample):")
                print(f"    Base probability: {base:.4f}")
                # Skip NaN contributions
                contribs_no_nan = [(f, c) for f, c in contribs if not np.isnan(c)]
                for f, c in sorted(contribs_no_nan, key=lambda x: -abs(x[1]))[:10]:
                    print(f"    {f}: contribution {c:+.4f}")

                # Risk factor decomposition with NaN handling
                clusters = cluster_features(X_shap_df.columns)
                weights = cluster_weights(clusters, corr_matrix)
                # For each cluster, sum adjusted importance (skip NaN)
                cluster_scores = {}
                for c, feats in clusters.items():
                    vals = [res['adj_importance'].get(f, 0) for f in feats]
                    vals_no_nan = [v for v in vals if not np.isnan(v)]
                    if vals_no_nan:
                        cluster_scores[c] = sum(vals_no_nan)
                    else:
                        cluster_scores[c] = float('nan')
                # Remove clusters with all-NaN scores from weighted sum
                valid_clusters = [c for c in clusters if not np.isnan(cluster_scores[c]) and not np.isnan(weights[c])]
                if valid_clusters:
                    r_total = sum([weights[c] * cluster_scores[c] for c in valid_clusters])
                else:
                    r_total = float('nan')
                print("\n  Risk Factor Decomposition:")
                for c in clusters:
                    w = weights[c]
                    s = cluster_scores[c]
                    if np.isnan(w) or np.isnan(s):
                        print(f"    Cluster '{c}': [Diagnostics] Skipped due to NaN weight or score.")
                    else:
                        print(f"    Cluster '{c}': weight={w:.3f}, score={s:.3f}")
                if np.isnan(r_total):
                    print("    [Diagnostics] Total risk score (weighted sum): NaN (all clusters invalid)")
                else:
                    print(f"    Total risk score (weighted sum): {r_total:.4f}")

        else:
            print("[INTERPRETABILITY] No interpretability results for best model.")

        # Final summary
        successful_models = sum(1 for result in trainer.results.values() 
                              if result['status'] == 'SUCCESS')

        print(f"\nPIPELINE COMPLETED SUCCESSFULLY!")
        print(f"   Dataset processed: {enhanced_df.shape}")
        print(f"   Models evaluated: {len(trainer.results)}")
        print(f"   Successful models: {successful_models}")
        print(f"   Best model: {trainer.best_model_name}")
        if trainer.best_model_name:
            best_acc = trainer.results[trainer.best_model_name]['accuracy']
            print(f"   Best accuracy: {best_acc:.4f}")
        print(f"   Outputs saved in: outputs/")

        logger.info("Pipeline completed successfully")

    except KeyboardInterrupt:
        print("\n⏹️ Pipeline interrupted by user")
        logger.info("Pipeline interrupted by user")
        sys.exit(0)

    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        logger.error(f"Pipeline failed: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()
