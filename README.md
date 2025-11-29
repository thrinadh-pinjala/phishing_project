# Phishing Detection System

A comprehensive machine learning pipeline for detecting phishing websites using advanced data preprocessing, correlation analysis, and multiple classification algorithms.

## Features

- **Robust Data Preprocessing**: Automatic CSV loading, column standardization, missing value imputation, and type conversion
- **Feature Engineering**: Intelligent feature combination and derived feature creation
- **Correlation Analysis**: Multiple correlation methods (Pearson, Spearman, Kendall) with visualization
- **Multi-Algorithm Training**: 5+ machine learning models with comprehensive evaluation
  - Naive Bayes
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - Logistic Regression
  
- **Model Evaluation**: Cross-validation, precision, recall, F1-score, ROC-AUC, and confusion matrices
- **Interpretability**: SHAP values, feature importance tracking, risk decomposition
- **Longevity Analysis**: Feature drift detection and model degradation monitoring
- **Comprehensive Visualization**: Heatmaps, bar charts, graph maps, and performance reports

## Project Structure

```
phishing_project/
├── data/
│   ├── raw/           # Input CSV files
│   └── processed/     # Cleaned and processed datasets
├── models/            # Trained model storage
├── outputs/           # Results, visualizations, and reports
├── static/            # Web interface files
├── utils/             # Helper modules
│   └── longevity_analysis.py
├── main.py            # Main pipeline orchestrator
├── app.py             # Web application (if applicable)
└── requirements.txt   # Python dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/phishing_project.git
cd phishing_project
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the complete pipeline:
```bash
python main.py
```

The pipeline will:
1. Load CSV files from `data/raw/`
2. Preprocess and clean the data
3. Perform correlation analysis
4. Train multiple ML models
5. Generate visualizations and reports
6. Save results to `outputs/`

## Output Files

- **Visualizations**: Heatmaps, bar charts, graph maps (PNG format)
- **Reports**: Comprehensive analysis reports (TXT format)
- **Logs**: Detailed execution logs
- **Processed Data**: Cleaned datasets (CSV format)

## Key Algorithms

### Naive Bayes
Probabilistic classifier based on Bayes' theorem. Assumes feature independence but works well for high-dimensional data.

### Decision Tree
Tree-based model making decisions by splitting data on feature values. Fast and interpretable.

### Random Forest
Ensemble of decision trees using bootstrap aggregation (bagging) for improved accuracy and robustness.

### Gradient Boosting
Sequential ensemble method that learns from previous errors, often achieving state-of-the-art results.

### Logistic Regression
Linear model for binary classification using the logistic function. Fast and provides probability estimates.

## Performance Metrics

- **Accuracy**: Overall correctness of predictions
- **Precision**: Proportion of positive predictions that were correct
- **Recall**: Proportion of actual positives correctly identified
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the receiver operating characteristic curve
- **Cross-Validation**: 5-fold validation for generalization assessment

## Correlation Analysis

The pipeline computes correlation matrices using three methods:
- **Pearson**: Linear correlation
- **Spearman**: Rank-based correlation
- **Kendall**: Rank-based correlation (robust to outliers)

Highly correlated features are identified and visualized through:
- Enhanced heatmaps with highlighted pairs
- Bar graphs of top correlations
- Network graph maps showing feature relationships

## Interpretability Framework

- **SHAP Values**: Explains individual predictions with game-theoretic approach
- **Feature Importance**: Correlation-weighted importance scores
- **Probability Path**: Traces how features contribute to predictions
- **Risk Decomposition**: Decomposes risk factors by feature clusters

## Longevity Analysis

Monitors model performance over time:
- **KL Divergence**: Measures feature distribution changes
- **PSI (Population Stability Index)**: Detects feature drift
- **Feature Relevance Tracking**: Monitors importance changes
- **Automated Alerts**: Flags significant drift

## Requirements

See `requirements.txt` for the complete list of dependencies:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- networkx
- shap

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Your Name / Organization

## Citation

If you use this project in your research, please cite:
```
@software{phishing_detection_2025,
  title={Comprehensive Phishing Detection System},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/phishing_project}
}
```

## Acknowledgments

- Built with scikit-learn, pandas, and matplotlib
- Inspired by cybersecurity best practices
- SHAP library for model interpretability

## Contact

For questions or support, please open an issue on GitHub.
