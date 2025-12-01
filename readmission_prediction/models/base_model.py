"""
Base model interface for readmission prediction models.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import logging

logger = logging.getLogger(__name__)


class BaseReadmissionModel(ABC):
    """Abstract base class for readmission prediction models."""
    
    def __init__(self, model_name: str, config: Dict[str, Any]):
        self.model_name = model_name
        self.config = config
        self.model = None
        self.scaler = None
        self.feature_columns = config.get('features', [])
        self.is_trained = False
        
    @abstractmethod
    def _build_model(self) -> Any:
        """Build the specific model instance."""
        pass
    
    @abstractmethod
    def _fit_model(self, X_train: np.ndarray, y_train: np.ndarray) -> Any:
        """Train the model."""
        pass
    
    @abstractmethod
    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        pass
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for modeling."""
        # Select feature columns
        available_features = [col for col in self.feature_columns if col in df.columns]
        if len(available_features) != len(self.feature_columns):
            missing_features = set(self.feature_columns) - set(available_features)
            logger.warning(f"Missing features: {missing_features}")
        
        X = df[available_features].copy()
        
        # Fill missing values
        X = X.fillna(0)
        
        return X
    
    def split_data(self, df: pd.DataFrame, target_column: str = 'readmission_30d', 
                   test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train and test sets."""
        # Remove rows with missing target
        clean_df = df.dropna(subset=[target_column])
        
        # Prepare features and target
        X = self.prepare_features(clean_df)
        y = clean_df[target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, scale_features: bool = True) -> 'BaseReadmissionModel':
        """Train the model."""
        logger.info(f"Training {self.model_name}...")
        
        # Build model if not exists
        if self.model is None:
            self.model = self._build_model()
        
        # Scale features if requested
        if scale_features:
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
        else:
            X_train_scaled = X_train.values
        
        # Train model
        self._fit_model(X_train_scaled, y_train.values)
        self.is_trained = True
        
        logger.info(f"{self.model_name} training completed")
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Scale features if scaler exists
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values
        
        return self._predict_proba(X_scaled)
    
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Make binary predictions."""
        probabilities = self.predict_proba(X)
        
        # Handle both binary and multiclass outputs
        if probabilities.shape[1] == 2:
            # Binary classification - use positive class probability
            positive_proba = probabilities[:, 1]
        else:
            # Assume single column output
            positive_proba = probabilities[:, 0]
        
        return (positive_proba >= threshold).astype(int)
    
    def save_model(self, filepath: str) -> None:
        """Save trained model to disk."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'config': self.config,
            'feature_columns': self.feature_columns
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'BaseReadmissionModel':
        """Load trained model from disk."""
        model_data = joblib.load(filepath)
        
        # Create instance
        instance = cls(model_data['config']['name'], model_data['config'])
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.feature_columns = model_data['feature_columns']
        instance.is_trained = True
        
        logger.info(f"Model loaded from {filepath}")
        return instance
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance if available."""
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
        
        if hasattr(self.model, 'feature_importances_'):
            importance_dict = dict(zip(self.feature_columns, self.model.feature_importances_))
            return importance_dict
        elif hasattr(self.model, 'coef_'):
            # For linear models, use absolute coefficients
            importance_dict = dict(zip(self.feature_columns, np.abs(self.model.coef_)))
            return importance_dict
        else:
            logger.warning(f"Feature importance not available for {self.model_name}")
            return {}
    
    def __str__(self) -> str:
        return f"{self.model_name}(trained={self.is_trained}, features={len(self.feature_columns)})"