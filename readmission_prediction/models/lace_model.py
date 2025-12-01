"""
LACE Index model for readmission prediction using linear regression.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from typing import Any

from .base_model import BaseReadmissionModel


class LACEModel(BaseReadmissionModel):
    """LACE Index model using Linear Regression."""
    
    def _build_model(self) -> LinearRegression:
        """Build Linear Regression model."""
        return LinearRegression(**self.config.get('hyperparameters', {}))
    
    def _fit_model(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the linear regression model."""
        self.model.fit(X_train, y_train)
    
    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities using linear regression.
        Since LinearRegression doesn't have predict_proba, we use predict 
        and format as probabilities.
        """
        predictions = self.model.predict(X)
        
        # Clip predictions to [0, 1] range for probability interpretation
        predictions = np.clip(predictions, 0, 1)
        
        # Format as 2D array with [negative_prob, positive_prob]
        negative_proba = 1 - predictions
        positive_proba = predictions
        
        return np.column_stack([negative_proba, positive_proba])
    
    def get_coefficients(self) -> dict:
        """Get model coefficients."""
        if not self.is_trained:
            raise ValueError("Model must be trained to get coefficients")
        
        coef_dict = dict(zip(self.feature_columns, self.model.coef_))
        coef_dict['intercept'] = self.model.intercept_
        
        return coef_dict