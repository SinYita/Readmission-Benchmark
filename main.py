"""
Main execution script for readmission prediction baseline models.
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Local imports
from readmission_prediction.data.data_loader import load_preprocessed_data
from readmission_prediction.utils.feature_engineering import prepare_modeling_features
from readmission_prediction.models.lace_model import LACEModel
from readmission_prediction.models.random_forest_model import RandomForestModel
from readmission_prediction.utils.metrics import ReadmissionEvaluator, evaluate_readmission_model
from readmission_prediction.evaluation.visualizations import ReadmissionVisualizer, create_comprehensive_report
from readmission_prediction.utils.config import MODEL_CONFIGS, EVALUATION_CONFIG, RESULTS_DIR

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(RESULTS_DIR / 'training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def train_lace_model(df: pd.DataFrame) -> tuple:
    """Train LACE index baseline model."""
    logger.info("Training LACE Index Model...")
    
    # Initialize model
    config = MODEL_CONFIGS['lace']
    model = LACEModel('LACE Index', config)
    
    # Split data
    X_train, X_test, y_train, y_test = model.split_data(
        df, test_size=EVALUATION_CONFIG['test_size'], 
        random_state=EVALUATION_CONFIG['random_state']
    )
    
    # Train model
    model.fit(X_train, y_train, scale_features=True)
    
    # Evaluate
    results = evaluate_readmission_model(model, X_test, y_test)
    
    # Save model
    model_path = RESULTS_DIR / 'models' / f'lace_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.joblib'
    model.save_model(str(model_path))
    
    return model, results, X_test, y_test


def train_random_forest_model(df: pd.DataFrame) -> tuple:
    """Train Random Forest baseline model."""
    logger.info("Training Random Forest Model...")
    
    # Initialize model
    config = MODEL_CONFIGS['random_forest']
    model = RandomForestModel('Random Forest', config)
    
    # Split data
    X_train, X_test, y_train, y_test = model.split_data(
        df, test_size=EVALUATION_CONFIG['test_size'], 
        random_state=EVALUATION_CONFIG['random_state']
    )
    
    # Train model
    model.fit(X_train, y_train, scale_features=False)  # RF doesn't need scaling
    
    # Evaluate
    results = evaluate_readmission_model(model, X_test, y_test)
    
    # Save model
    model_path = RESULTS_DIR / 'models' / f'random_forest_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.joblib'
    model.save_model(str(model_path))
    
    return model, results, X_test, y_test


def run_baseline_comparison(df: pd.DataFrame) -> None:
    """Run comprehensive baseline model comparison."""
    logger.info("Starting baseline model comparison...")
    
    # Train models
    lace_model, lace_results, X_test_lace, y_test_lace = train_lace_model(df)
    rf_model, rf_results, X_test_rf, y_test_rf = train_random_forest_model(df)
    
    # Create evaluator for reporting
    evaluator = ReadmissionEvaluator()
    
    # Print individual reports
    evaluator.print_evaluation_report(lace_results)
    evaluator.print_evaluation_report(rf_results)
    
    # Compare models
    print(f"\n{'='*60}")
    print("BASELINE MODELS COMPARISON")
    print("="*60)
    
    comparison_df = evaluator.compare_models([lace_results, rf_results])
    print(comparison_df.to_string(index=False))
    
    # Feature importance comparison
    print(f"\n{'='*40}")
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*40)
    
    # LACE coefficients
    print(f"\nLACE Model Coefficients:")
    lace_coefs = lace_model.get_coefficients()
    for feature, coef in lace_coefs.items():
        print(f"{feature:20s}: {coef:8.4f}")
    
    # Random Forest importance
    print(f"\nRandom Forest Feature Importance:")
    rf_importance = rf_model.get_feature_importance()
    sorted_rf_features = sorted(rf_importance.items(), key=lambda x: x[1], reverse=True)
    for feature, importance in sorted_rf_features[:10]:  # Top 10
        print(f"{feature:20s}: {importance:8.4f}")
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save comparison results
    comparison_df.to_csv(RESULTS_DIR / f'model_comparison_{timestamp}.csv', index=False)
    
    # Save predictions
    lace_predictions = pd.DataFrame({
        'actual': y_test_lace.values,
        'predicted_proba': lace_model.predict_proba(X_test_lace)[:, 1],
        'predicted': lace_model.predict(X_test_lace)
    })
    lace_predictions.to_csv(RESULTS_DIR / 'predictions' / f'lace_predictions_{timestamp}.csv', index=False)
    
    rf_predictions = pd.DataFrame({
        'actual': y_test_rf.values,
        'predicted_proba': rf_model.predict_proba(X_test_rf)[:, 1],
        'predicted': rf_model.predict(X_test_rf)
    })
    rf_predictions.to_csv(RESULTS_DIR / 'predictions' / f'rf_predictions_{timestamp}.csv', index=False)
    
    # Create comprehensive visualization report
    models_data = [
        {
            'name': 'LACE Index',
            'y_true': y_test_lace.values,
            'y_proba': lace_model.predict_proba(X_test_lace)
        },
        {
            'name': 'Random Forest',
            'y_true': y_test_rf.values,
            'y_proba': rf_model.predict_proba(X_test_rf)
        }
    ]
    
    create_comprehensive_report(
        df, [lace_results, rf_results], models_data,
        output_dir=RESULTS_DIR / 'reports'
    )
    
    # Feature importance plots
    visualizer = ReadmissionVisualizer()
    
    # LACE coefficients plot
    visualizer.plot_feature_importance(
        lace_coefs, 
        title="LACE Model Coefficients",
        save_path=RESULTS_DIR / 'reports' / 'lace_coefficients.png'
    )
    
    # Random Forest importance plot
    visualizer.plot_feature_importance(
        rf_importance, 
        title="Random Forest Feature Importance",
        save_path=RESULTS_DIR / 'reports' / 'rf_importance.png'
    )
    
    logger.info(f"Baseline comparison complete. Results saved to {RESULTS_DIR}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Hospital Readmission Prediction Baseline Models')
    parser.add_argument('--model', type=str, choices=['lace', 'random_forest', 'all'], 
                       default='all', help='Model to train')
    parser.add_argument('--data-path', type=str, default='demo', 
                       help='Path to MIMIC-III data directory')
    
    args = parser.parse_args()
    
    logger.info("Starting Hospital Readmission Prediction Pipeline")
    logger.info("="*60)
    
    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    df = load_preprocessed_data()
    
    # Feature engineering
    logger.info("Engineering features...")
    df_with_features = prepare_modeling_features(df, feature_set='extended')
    
    # Data summary
    logger.info(f"Dataset summary:")
    logger.info(f"Total admissions: {len(df_with_features)}")
    logger.info(f"Total patients: {df_with_features['subject_id'].nunique()}")
    logger.info(f"30-day readmissions: {df_with_features['readmission_30d'].sum()}")
    logger.info(f"Readmission rate: {df_with_features['readmission_30d'].mean():.3f}")
    logger.info(f"Mean LACE score: {df_with_features['LACE_score'].mean():.2f}")
    
    # Train models based on selection
    if args.model == 'lace':
        model, results, _, _ = train_lace_model(df_with_features)
        evaluator = ReadmissionEvaluator()
        evaluator.print_evaluation_report(results)
        
    elif args.model == 'random_forest':
        model, results, _, _ = train_random_forest_model(df_with_features)
        evaluator = ReadmissionEvaluator()
        evaluator.print_evaluation_report(results)
        
    else:  # args.model == 'all'
        run_baseline_comparison(df_with_features)
    
    logger.info("Pipeline execution completed successfully!")


if __name__ == "__main__":
    main()