"""
Custom evaluation metrics for readmission prediction.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class ReadmissionEvaluator:
    """Comprehensive evaluation metrics for readmission prediction."""
    
    @staticmethod
    def calculate_sensitivity_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
        """Calculate sensitivity (recall) and specificity."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Same as recall
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        return sensitivity, specificity
    
    @staticmethod
    def calculate_ppv_npv(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
        """Calculate positive predictive value (precision) and negative predictive value."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # Same as precision
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        
        return ppv, npv
    
    @staticmethod
    def calculate_likelihood_ratios(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
        """Calculate positive and negative likelihood ratios."""
        sensitivity, specificity = ReadmissionEvaluator.calculate_sensitivity_specificity(y_true, y_pred)
        
        # Positive likelihood ratio
        plr = sensitivity / (1 - specificity) if specificity != 1.0 else float('inf')
        
        # Negative likelihood ratio
        nlr = (1 - sensitivity) / specificity if specificity != 0.0 else float('inf')
        
        return plr, nlr
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      y_pred_proba: np.ndarray = None, model_name: str = "Model") -> Dict[str, Any]:
        """
        Comprehensive model evaluation.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels (binary)
            y_pred_proba: Predicted probabilities (optional)
            model_name: Name of the model for reporting
        
        Returns:
            Dictionary with all evaluation metrics
        """
        logger.info(f"Evaluating {model_name}...")
        
        results = {'model_name': model_name}
        
        # Basic classification metrics
        results['accuracy'] = accuracy_score(y_true, y_pred)
        results['precision'] = precision_score(y_true, y_pred, zero_division=0)
        results['recall'] = recall_score(y_true, y_pred, zero_division=0)
        results['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
        
        # Clinical interpretation metrics
        sensitivity, specificity = self.calculate_sensitivity_specificity(y_true, y_pred)
        results['sensitivity'] = sensitivity
        results['specificity'] = specificity
        
        ppv, npv = self.calculate_ppv_npv(y_true, y_pred)
        results['ppv'] = ppv
        results['npv'] = npv
        
        plr, nlr = self.calculate_likelihood_ratios(y_true, y_pred)
        results['positive_lr'] = plr
        results['negative_lr'] = nlr
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        results['confusion_matrix'] = cm
        results['tn'] = cm[0, 0]
        results['fp'] = cm[0, 1]
        results['fn'] = cm[1, 0]
        results['tp'] = cm[1, 1]
        
        # Probability-based metrics (if probabilities provided)
        if y_pred_proba is not None:
            try:
                # Handle different probability formats
                if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] == 2:
                    # Binary classification with [neg_prob, pos_prob]
                    proba_positive = y_pred_proba[:, 1]
                elif y_pred_proba.ndim == 2 and y_pred_proba.shape[1] == 1:
                    # Single column probability
                    proba_positive = y_pred_proba[:, 0]
                else:
                    # 1D array
                    proba_positive = y_pred_proba
                
                results['auc_roc'] = roc_auc_score(y_true, proba_positive)
                results['auc_pr'] = average_precision_score(y_true, proba_positive)
            except Exception as e:
                logger.warning(f"Could not calculate AUC metrics: {e}")
                results['auc_roc'] = None
                results['auc_pr'] = None
        else:
            results['auc_roc'] = None
            results['auc_pr'] = None
        
        # Class distribution
        results['n_samples'] = len(y_true)
        results['n_positive'] = np.sum(y_true)
        results['n_negative'] = len(y_true) - np.sum(y_true)
        results['positive_rate'] = np.mean(y_true)
        
        return results
    
    def compare_models(self, evaluations: list) -> pd.DataFrame:
        """Compare multiple model evaluations."""
        comparison_metrics = [
            'model_name', 'accuracy', 'precision', 'recall', 'f1_score',
            'sensitivity', 'specificity', 'auc_roc', 'auc_pr', 'ppv', 'npv'
        ]
        
        comparison_data = []
        for eval_result in evaluations:
            row = {metric: eval_result.get(metric, None) for metric in comparison_metrics}
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Round numerical columns
        numerical_cols = comparison_df.select_dtypes(include=[np.number]).columns
        comparison_df[numerical_cols] = comparison_df[numerical_cols].round(4)
        
        return comparison_df
    
    def print_evaluation_report(self, eval_result: Dict[str, Any]) -> None:
        """Print comprehensive evaluation report."""
        print(f"\n{eval_result['model_name']} Evaluation Report")
        print("=" * 50)
        
        print(f"Dataset: {eval_result['n_samples']} samples")
        print(f"Positive class: {eval_result['n_positive']} ({eval_result['positive_rate']:.3f})")
        print(f"Negative class: {eval_result['n_negative']} ({1-eval_result['positive_rate']:.3f})")
        
        print(f"\nClassification Metrics:")
        print(f"Accuracy:    {eval_result['accuracy']:.4f}")
        print(f"Precision:   {eval_result['precision']:.4f}")
        print(f"Recall:      {eval_result['recall']:.4f}")
        print(f"F1-Score:    {eval_result['f1_score']:.4f}")
        
        print(f"\nClinical Metrics:")
        print(f"Sensitivity: {eval_result['sensitivity']:.4f}")
        print(f"Specificity: {eval_result['specificity']:.4f}")
        print(f"PPV:         {eval_result['ppv']:.4f}")
        print(f"NPV:         {eval_result['npv']:.4f}")
        
        if eval_result['auc_roc'] is not None:
            print(f"\nProbabilistic Metrics:")
            print(f"AUC-ROC:     {eval_result['auc_roc']:.4f}")
            print(f"AUC-PR:      {eval_result['auc_pr']:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(f"              Predicted")
        print(f"             No   Yes")
        print(f"Actual No   {eval_result['tn']:4d} {eval_result['fp']:4d}")
        print(f"       Yes  {eval_result['fn']:4d} {eval_result['tp']:4d}")
        
        if eval_result['positive_lr'] != float('inf'):
            print(f"\nLikelihood Ratios:")
            print(f"Positive LR: {eval_result['positive_lr']:.2f}")
            print(f"Negative LR: {eval_result['negative_lr']:.2f}")


def evaluate_readmission_model(model, X_test: pd.DataFrame, y_test: pd.Series, 
                              threshold: float = 0.5) -> Dict[str, Any]:
    """
    Convenience function to evaluate a readmission prediction model.
    
    Args:
        model: Trained model with predict and predict_proba methods
        X_test: Test features
        y_test: Test labels
        threshold: Classification threshold
    
    Returns:
        Evaluation results dictionary
    """
    evaluator = ReadmissionEvaluator()
    
    # Get predictions
    y_pred_proba = model.predict_proba(X_test)
    y_pred = model.predict(X_test, threshold=threshold)
    
    # Evaluate
    results = evaluator.evaluate_model(
        y_test.values, y_pred, y_pred_proba, 
        model_name=getattr(model, 'model_name', 'Unknown')
    )
    
    return results