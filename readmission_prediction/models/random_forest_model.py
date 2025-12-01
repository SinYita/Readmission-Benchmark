"""
Random Forest model for readmission prediction.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from typing import Any, Dict

from .base_model import BaseReadmissionModel


class RandomForestModel(BaseReadmissionModel):
    """Random Forest model for readmission prediction."""
    
    def _build_model(self) -> RandomForestClassifier:
        """Build Random Forest classifier."""
        hyperparams = self.config.get('hyperparameters', {})
        return RandomForestClassifier(**hyperparams)
    
    def _fit_model(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the random forest model."""
        self.model.fit(X_train, y_train)
    
    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.model.predict_proba(X)
    
    def get_feature_importance_detailed(self) -> Dict[str, float]:
        """Get detailed feature importance with rankings."""
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
        
        importance_dict = dict(zip(self.feature_columns, self.model.feature_importances_))
        
        # Sort by importance
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'importance': dict(sorted_importance),
            'ranking': {feature: rank + 1 for rank, (feature, _) in enumerate(sorted_importance)}
        }
    
    def get_model_parameters(self) -> Dict[str, Any]:
        """Get model parameters and statistics."""
        if not self.is_trained:
            raise ValueError("Model must be trained to get parameters")
        
        return {
            'n_estimators': self.model.n_estimators,
            'max_depth': self.model.max_depth,
            'min_samples_split': self.model.min_samples_split,
            'min_samples_leaf': self.model.min_samples_leaf,
            'n_features': self.model.n_features_in_,
            'n_classes': self.model.n_classes_,
            'oob_score': getattr(self.model, 'oob_score_', None) if self.model.oob_score else None
        }