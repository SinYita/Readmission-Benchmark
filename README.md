# Hospital Readmission Prediction Benchmark

This project implements baseline models for predicting 30-day hospital readmissions using MIMIC-III data, with a focus on the LACE index and machine learning approaches.

## Project Structure

```
readmission_prediction/
├── data/
│   └── data_loader.py          # Data loading and preprocessing
├── models/
│   ├── lace_model.py          # LACE index implementation
│   ├── random_forest_model.py  # Random Forest baseline
│   └── base_model.py          # Base model interface
├── utils/
│   ├── feature_engineering.py # Feature extraction and engineering
│   ├── metrics.py             # Custom evaluation metrics
│   └── config.py              # Configuration settings
├── evaluation/
│   ├── evaluator.py           # Model evaluation pipeline
│   └── visualizations.py      # Plotting and visualization tools
└── main.py                    # Main execution script
results/
├── models/                    # Trained model artifacts
├── predictions/               # Model predictions
└── reports/                   # Analysis reports and plots
```

## Models Implemented

### 1. LACE Index Baseline (Logistic Regression)
- **L**ength of stay (0-7 points)
- **A**cuteness of admission (0-3 points)
- **C**omorbidities via Charlson Index (0-5 points)
- **E**mergency visits in prior 6 months (0-4 points)

### 2. Random Forest Baseline
- Ensemble method using LACE components and additional clinical features
- Handles non-linear relationships and feature interactions
- Built-in feature importance analysis

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run all baseline models
python main.py

# Run specific model
python main.py --model lace
python main.py --model random_forest
```

## Evaluation Metrics

- **AUC-ROC**: Area under the ROC curve
- **AUC-PR**: Area under the Precision-Recall curve (better for imbalanced data)
- **Accuracy**: Overall classification accuracy
- **Precision/Recall/F1**: Class-specific performance metrics
- **Sensitivity/Specificity**: Clinical interpretation metrics

## Dataset

This project uses MIMIC-III demo data containing:
- Patient demographics and admission details
- ICD-9 diagnosis codes for comorbidity calculation
- Length of stay and admission type information
- Historical emergency department visits

## Results

Results are automatically saved to the `results/` directory with timestamps and model configurations for reproducibility.

## Requirements

- Python 3.7+
- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.3.0
- seaborn >= 0.11.0