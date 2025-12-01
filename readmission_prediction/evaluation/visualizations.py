"""
Visualization tools for readmission prediction analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

# Set style
plt.style.use('default')
sns.set_palette("husl")


class ReadmissionVisualizer:
    """Visualization tools for readmission prediction analysis."""
    
    def __init__(self, figsize: tuple = (12, 8)):
        self.figsize = figsize
    
    def plot_lace_distribution(self, df: pd.DataFrame, save_path: str = None) -> None:
        """Plot LACE score distribution by readmission status."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. LACE Score Distribution
        axes[0, 0].hist([df[df['readmission_30d']==0]['LACE_score'], 
                        df[df['readmission_30d']==1]['LACE_score']], 
                       bins=15, alpha=0.7, label=['No Readmission', 'Readmission'])
        axes[0, 0].set_xlabel('LACE Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('LACE Score Distribution by Readmission Status')
        axes[0, 0].legend()
        
        # 2. LACE Components
        lace_components = ['L_score', 'A_score', 'C_score', 'E_score']
        mean_scores = []
        for component in lace_components:
            no_readmit = df[df['readmission_30d']==0][component].mean()
            readmit = df[df['readmission_30d']==1][component].mean()
            mean_scores.append([no_readmit, readmit])
        
        mean_scores = np.array(mean_scores).T
        x = np.arange(len(lace_components))
        width = 0.35
        
        axes[0, 1].bar(x - width/2, mean_scores[0], width, label='No Readmission', alpha=0.7)
        axes[0, 1].bar(x + width/2, mean_scores[1], width, label='Readmission', alpha=0.7)
        axes[0, 1].set_xlabel('LACE Components')
        axes[0, 1].set_ylabel('Mean Score')
        axes[0, 1].set_title('Mean LACE Component Scores')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(lace_components)
        axes[0, 1].legend()
        
        # 3. Length of Stay Distribution
        axes[1, 0].hist([df[df['readmission_30d']==0]['los_days'],
                        df[df['readmission_30d']==1]['los_days']],
                       bins=20, alpha=0.7, label=['No Readmission', 'Readmission'])
        axes[1, 0].set_xlabel('Length of Stay (days)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Length of Stay Distribution')
        axes[1, 0].legend()
        
        # 4. Risk Categories
        if 'risk_category' in df.columns:
            risk_crosstab = pd.crosstab(df['risk_category'], df['readmission_30d'], normalize='index')
            risk_crosstab.plot(kind='bar', ax=axes[1, 1], alpha=0.7)
            axes[1, 1].set_title('Readmission Rate by Risk Category')
            axes[1, 1].set_xlabel('Risk Category')
            axes[1, 1].set_ylabel('Rate')
            axes[1, 1].legend(['No Readmission', 'Readmission'])
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"LACE distribution plot saved to {save_path}")
        
        plt.show()
    
    def plot_model_performance(self, evaluations: List[Dict[str, Any]], save_path: str = None) -> None:
        """Plot model performance comparison."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Extract model names and metrics
        models = [eval_result['model_name'] for eval_result in evaluations]
        
        # 1. Main Classification Metrics
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        metric_values = {metric: [eval_result[metric] for eval_result in evaluations] for metric in metrics}
        
        x = np.arange(len(models))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            axes[0, 0].bar(x + i * width, metric_values[metric], width, 
                          label=metric.replace('_', ' ').title(), alpha=0.7)
        
        axes[0, 0].set_xlabel('Models')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Classification Metrics Comparison')
        axes[0, 0].set_xticks(x + width * 1.5)
        axes[0, 0].set_xticklabels(models, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].set_ylim(0, 1)
        
        # 2. Clinical Metrics
        clinical_metrics = ['sensitivity', 'specificity', 'ppv', 'npv']
        clinical_values = {metric: [eval_result[metric] for eval_result in evaluations] for metric in clinical_metrics}
        
        for i, metric in enumerate(clinical_metrics):
            axes[0, 1].bar(x + i * width, clinical_values[metric], width, 
                          label=metric.upper(), alpha=0.7)
        
        axes[0, 1].set_xlabel('Models')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_title('Clinical Metrics Comparison')
        axes[0, 1].set_xticks(x + width * 1.5)
        axes[0, 1].set_xticklabels(models, rotation=45)
        axes[0, 1].legend()
        axes[0, 1].set_ylim(0, 1)
        
        # 3. AUC Metrics
        auc_roc_values = [eval_result.get('auc_roc', 0) for eval_result in evaluations]
        auc_pr_values = [eval_result.get('auc_pr', 0) for eval_result in evaluations]
        
        axes[1, 0].bar(x - width/2, auc_roc_values, width, label='AUC-ROC', alpha=0.7)
        axes[1, 0].bar(x + width/2, auc_pr_values, width, label='AUC-PR', alpha=0.7)
        axes[1, 0].set_xlabel('Models')
        axes[1, 0].set_ylabel('AUC Score')
        axes[1, 0].set_title('AUC Metrics Comparison')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(models, rotation=45)
        axes[1, 0].legend()
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
        
        # 4. Confusion Matrix Heatmap (for first model as example)
        if evaluations:
            cm = evaluations[0]['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
            axes[1, 1].set_xlabel('Predicted')
            axes[1, 1].set_ylabel('Actual')
            axes[1, 1].set_title(f'Confusion Matrix - {evaluations[0]["model_name"]}')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model performance plot saved to {save_path}")
        
        plt.show()
    
    def plot_roc_pr_curves(self, models_data: List[Dict[str, Any]], save_path: str = None) -> None:
        """
        Plot ROC and Precision-Recall curves for multiple models.
        
        models_data: List of dicts with keys 'name', 'y_true', 'y_proba'
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # ROC Curve
        for model_data in models_data:
            y_true = model_data['y_true']
            y_proba = model_data['y_proba']
            name = model_data['name']
            
            # Handle probability format
            if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                proba_positive = y_proba[:, 1]
            else:
                proba_positive = y_proba.ravel()
            
            fpr, tpr, _ = roc_curve(y_true, proba_positive)
            auc_score = np.trapz(tpr, fpr)
            
            axes[0].plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})', linewidth=2)
        
        axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title('ROC Curves')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Precision-Recall Curve
        for model_data in models_data:
            y_true = model_data['y_true']
            y_proba = model_data['y_proba']
            name = model_data['name']
            
            # Handle probability format
            if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                proba_positive = y_proba[:, 1]
            else:
                proba_positive = y_proba.ravel()
            
            precision, recall, _ = precision_recall_curve(y_true, proba_positive)
            avg_precision = np.mean(precision)
            
            axes[1].plot(recall, precision, label=f'{name} (AP = {avg_precision:.3f})', linewidth=2)
        
        # Baseline (random classifier)
        baseline = np.mean([model_data['y_true'] for model_data in models_data][0])
        axes[1].axhline(y=baseline, color='k', linestyle='--', alpha=0.5, label=f'Baseline ({baseline:.3f})')
        
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].set_title('Precision-Recall Curves')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC-PR curves plot saved to {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, importance_dict: Dict[str, float], 
                               title: str = "Feature Importance", 
                               top_n: int = 15, save_path: str = None) -> None:
        """Plot feature importance."""
        if not importance_dict:
            logger.warning("No feature importance data available")
            return
        
        # Sort features by importance
        sorted_features = sorted(importance_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        
        # Take top N features
        top_features = sorted_features[:top_n]
        
        features, importances = zip(*top_features)
        
        plt.figure(figsize=(10, max(6, len(features) * 0.4)))
        
        # Create horizontal bar plot
        colors = ['red' if imp < 0 else 'blue' for imp in importances]
        bars = plt.barh(range(len(features)), importances, color=colors, alpha=0.7)
        
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.title(title)
        plt.yticks(range(len(features)), features)
        
        # Add value labels on bars
        for i, (feature, importance) in enumerate(top_features):
            plt.text(importance + (0.01 if importance >= 0 else -0.01), i, 
                    f'{importance:.3f}', 
                    ha='left' if importance >= 0 else 'right', 
                    va='center')
        
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.show()


def create_comprehensive_report(df: pd.DataFrame, evaluations: List[Dict[str, Any]], 
                              models_data: List[Dict[str, Any]] = None,
                              output_dir: str = "results/reports") -> None:
    """
    Create comprehensive visualization report.
    
    Args:
        df: Dataset with LACE scores and features
        evaluations: List of model evaluation results
        models_data: List of model prediction data for ROC/PR curves
        output_dir: Directory to save plots
    """
    from pathlib import Path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    visualizer = ReadmissionVisualizer()
    
    # 1. LACE Distribution Analysis
    visualizer.plot_lace_distribution(df, save_path=output_dir / "lace_distribution.png")
    
    # 2. Model Performance Comparison
    visualizer.plot_model_performance(evaluations, save_path=output_dir / "model_comparison.png")
    
    # 3. ROC and PR Curves (if model data available)
    if models_data:
        visualizer.plot_roc_pr_curves(models_data, save_path=output_dir / "roc_pr_curves.png")
    
    logger.info(f"Comprehensive visualization report saved to {output_dir}")