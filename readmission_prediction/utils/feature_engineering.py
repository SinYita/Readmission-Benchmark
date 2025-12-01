"""
Feature engineering module for readmission prediction.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
import logging

from .config import LACE_SCORING

logger = logging.getLogger(__name__)


class LACEFeatureEngine:
    """Calculate LACE index components and total score."""
    
    @staticmethod
    def calculate_length_of_stay_score(los_days: float) -> int:
        """
        Calculate Length of Stay component (0-7 points)
        Based on LACE index scoring system.
        """
        if pd.isna(los_days) or los_days < 1:
            return 0
        elif los_days == 1:
            return 1
        elif los_days == 2:
            return 2
        elif los_days == 3:
            return 3
        elif 4 <= los_days <= 6:
            return 4
        elif 7 <= los_days <= 13:
            return 5
        else:  # >= 14 days
            return 7
    
    @staticmethod
    def calculate_acuteness_score(admission_type: str) -> int:
        """
        Calculate Acuteness component (0-3 points)
        Based on admission type.
        """
        if pd.isna(admission_type):
            return 0
        
        admission_type = str(admission_type).upper()
        if 'EMERGENCY' in admission_type or 'URGENT' in admission_type:
            return 3
        elif 'ELECTIVE' in admission_type:
            return 0
        else:
            return 1
    
    @staticmethod
    def calculate_comorbidity_score(charlson_score: int) -> int:
        """
        Convert Charlson score to LACE Comorbidity component (0-5 points)
        """
        if charlson_score == 0:
            return 0
        elif charlson_score == 1:
            return 1
        elif charlson_score == 2:
            return 2
        elif charlson_score == 3:
            return 3
        elif charlson_score == 4:
            return 4
        else:  # >= 5
            return 5
    
    @staticmethod
    def calculate_emergency_visits_score(visit_count: int) -> int:
        """
        Calculate Emergency visits component (0-4 points)
        """
        if visit_count == 0:
            return 0
        elif visit_count == 1:
            return 1
        elif visit_count == 2:
            return 2
        elif visit_count == 3:
            return 3
        else:  # >= 4 visits
            return 4
    
    def calculate_lace_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all LACE components and total score."""
        logger.info("Calculating LACE scores...")
        
        result_df = df.copy()
        
        # Calculate individual LACE components
        result_df['L_score'] = df['los_days'].apply(self.calculate_length_of_stay_score)
        result_df['A_score'] = df['admission_type'].apply(self.calculate_acuteness_score)
        result_df['C_score'] = df['charlson_score'].apply(self.calculate_comorbidity_score)
        result_df['E_score'] = df['emergency_visits_6m'].apply(self.calculate_emergency_visits_score)
        
        # Calculate total LACE score
        result_df['LACE_score'] = (
            result_df['L_score'] + 
            result_df['A_score'] + 
            result_df['C_score'] + 
            result_df['E_score']
        )
        
        logger.info(f"LACE scores calculated. Mean LACE score: {result_df['LACE_score'].mean():.2f}")
        
        return result_df


class AdditionalFeatureEngine:
    """Generate additional clinical features for modeling."""
    
    @staticmethod
    def create_risk_categories(lace_scores: pd.Series) -> pd.Series:
        """Create risk categories based on LACE scores."""
        risk_bins = [0, 7, 10, 13, float('inf')]
        risk_labels = ['Low', 'Medium', 'High', 'Very High']
        return pd.cut(lace_scores, bins=risk_bins, labels=risk_labels, right=False)
    
    @staticmethod
    def create_age_groups(ages: pd.Series) -> pd.Series:
        """Create age group categories."""
        age_bins = [0, 18, 35, 50, 65, 80, float('inf')]
        age_labels = ['Pediatric', 'Young Adult', 'Adult', 'Middle Age', 'Elderly', 'Very Elderly']
        return pd.cut(ages, bins=age_bins, labels=age_labels, right=False)
    
    @staticmethod
    def create_los_categories(los_days: pd.Series) -> pd.Series:
        """Create length of stay categories."""
        los_bins = [0, 1, 3, 7, 14, float('inf')]
        los_labels = ['Very Short', 'Short', 'Medium', 'Long', 'Very Long']
        return pd.cut(los_days, bins=los_bins, labels=los_labels, right=False)
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer additional features for modeling."""
        logger.info("Engineering additional features...")
        
        result_df = df.copy()
        
        # Categorical features
        result_df['risk_category'] = self.create_risk_categories(df['LACE_score'])
        result_df['age_group'] = self.create_age_groups(df['age'])
        result_df['los_category'] = self.create_los_categories(df['los_days'])
        
        # Binary flags
        result_df['is_emergency'] = (df['admission_type'].str.contains('EMERGENCY', na=False)).astype(int)
        result_df['has_comorbidities'] = (df['charlson_score'] > 0).astype(int)
        result_df['has_prior_ed_visits'] = (df['emergency_visits_6m'] > 0).astype(int)
        result_df['long_stay'] = (df['los_days'] >= 7).astype(int)
        result_df['elderly'] = (df['age'] >= 65).astype(int)
        
        # Interaction features
        result_df['age_lace_interaction'] = df['age'] * df['LACE_score']
        result_df['los_charlson_interaction'] = df['los_days'] * df['charlson_score']
        result_df['emergency_age_interaction'] = result_df['is_emergency'] * df['age']
        
        logger.info(f"Feature engineering complete. Total features: {len(result_df.columns)}")
        
        return result_df


def prepare_modeling_features(df: pd.DataFrame, feature_set: str = 'extended') -> pd.DataFrame:
    """
    Prepare features for modeling.
    
    Args:
        df: Input dataframe
        feature_set: 'basic' for LACE only, 'extended' for all features
    
    Returns:
        DataFrame with engineered features
    """
    # Calculate LACE scores
    lace_engine = LACEFeatureEngine()
    df_with_lace = lace_engine.calculate_lace_scores(df)
    
    # Add additional features if requested
    if feature_set == 'extended':
        feature_engine = AdditionalFeatureEngine()
        df_with_features = feature_engine.engineer_features(df_with_lace)
    else:
        df_with_features = df_with_lace
    
    return df_with_features