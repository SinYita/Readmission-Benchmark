"""
LACE Index model for readmission prediction using logistic regression.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from typing import Any

from .base_model import BaseReadmissionModel


class LACEModel(BaseReadmissionModel):
    """LACE Index model using Logistic Regression."""
    
    def _build_model(self) -> LogisticRegression:
        """Build Logistic Regression model."""
        # Default hyperparameters for logistic regression
        default_params = {
            'random_state': 42,
            'max_iter': 1000,
            'class_weight': 'balanced'
        }
        params = {**default_params, **self.config.get('hyperparameters', {})}
        return LogisticRegression(**params)
    
    def _fit_model(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the logistic regression model."""
        self.model.fit(X_train, y_train)
    
    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities using logistic regression.
        Returns proper probability estimates for binary classification.
        """
        return self.model.predict_proba(X)
    
    def get_coefficients(self) -> dict:
        """Get model coefficients."""
        if not self.is_trained:
            raise ValueError("Model must be trained to get coefficients")
        
        # Handle both 1D and 2D coefficient arrays
        if self.model.coef_.ndim == 2:
            # For binary classification, take the first (and only) row
            coefficients = self.model.coef_[0]
        else:
            coefficients = self.model.coef_
            
        coef_dict = dict(zip(self.feature_columns, coefficients))
        
        # Handle intercept similarly
        if hasattr(self.model.intercept_, '__len__') and len(self.model.intercept_) > 0:
            coef_dict['intercept'] = self.model.intercept_[0]
        else:
            coef_dict['intercept'] = self.model.intercept_
        
        return coef_dict