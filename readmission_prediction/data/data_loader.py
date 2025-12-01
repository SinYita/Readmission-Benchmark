"""
Data loading and preprocessing module for MIMIC-III readmission prediction.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple, Dict, Any
import logging

from ..utils.config import DATA_DIR, CHARLSON_ICD9_MAPPING

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MIMICDataLoader:
    """Load and preprocess MIMIC-III data for readmission prediction."""
    
    def __init__(self, data_path: Path = DATA_DIR):
        self.data_path = Path(data_path)
        self.admissions = None
        self.patients = None
        self.diagnoses = None
        
    def load_raw_data(self) -> None:
        """Load raw MIMIC-III tables."""
        logger.info("Loading MIMIC-III data...")
        
        # Load main tables
        self.admissions = pd.read_csv(self.data_path / "ADMISSIONS.csv")
        self.patients = pd.read_csv(self.data_path / "PATIENTS.csv")
        self.diagnoses = pd.read_csv(self.data_path / "DIAGNOSES_ICD.csv")
        
        # Convert datetime columns
        datetime_cols = ['admittime', 'dischtime', 'deathtime', 'edregtime', 'edouttime']
        for col in datetime_cols:
            if col in self.admissions.columns:
                self.admissions[col] = pd.to_datetime(self.admissions[col])
        
        self.patients['dob'] = pd.to_datetime(self.patients['dob'])
        self.patients['dod'] = pd.to_datetime(self.patients['dod'])
        
        logger.info(f"Loaded {len(self.admissions)} admissions for {self.admissions['subject_id'].nunique()} patients")
    
    def calculate_age(self) -> pd.Series:
        """Calculate patient age at admission."""
        # Merge patients and admissions to get age
        merged = self.admissions.merge(self.patients[['subject_id', 'dob']], on='subject_id', how='left')
        
        # Handle MIMIC-III shifted dates by using a reasonable age calculation
        # MIMIC shifts dates for privacy, so we'll use a simpler approach
        ages = []
        for idx, row in merged.iterrows():
            try:
                if pd.notna(row['admittime']) and pd.notna(row['dob']):
                    # Calculate age, handling potential overflow
                    admit_year = row['admittime'].year
                    birth_year = row['dob'].year
                    age = admit_year - birth_year
                    
                    # Reasonable age bounds for adults
                    age = max(18, min(age, 120))
                else:
                    age = 65  # Default age if missing
            except (OverflowError, ValueError):
                age = 65  # Default age for problematic calculations
            
            ages.append(age)
        
        return pd.Series(ages, index=merged.index)
    
    def encode_categorical_features(self) -> Dict[str, pd.Series]:
        """Encode categorical variables for modeling."""
        encoded_features = {}
        
        # Gender encoding
        gender_map = {'M': 1, 'F': 0}
        patients_gender = self.patients.set_index('subject_id')['gender'].map(gender_map)
        encoded_features['gender_encoded'] = self.admissions['subject_id'].map(patients_gender).fillna(0)
        
        # Admission type encoding
        admission_type_map = {
            'EMERGENCY': 2,
            'URGENT': 1,
            'ELECTIVE': 0
        }
        encoded_features['admission_type_encoded'] = self.admissions['admission_type'].map(admission_type_map).fillna(1)
        
        # Insurance encoding
        insurance_map = {
            'Medicare': 2,
            'Medicaid': 1,
            'Private': 3,
            'Government': 1,
            'Self Pay': 0
        }
        encoded_features['insurance_encoded'] = self.admissions['insurance'].map(insurance_map).fillna(1)
        
        return encoded_features
    
    def calculate_charlson_score(self, subject_id: int, hadm_id: int) -> int:
        """Calculate Charlson Comorbidity Index for a patient admission."""
        # Get all diagnoses for this admission
        patient_diagnoses = self.diagnoses[
            (self.diagnoses['subject_id'] == subject_id) & 
            (self.diagnoses['hadm_id'] == hadm_id)
        ]['icd9_code'].tolist()
        
        if not patient_diagnoses:
            return 0
        
        charlson_score = 0
        found_conditions = set()
        
        # Check each diagnosis against Charlson categories
        for diagnosis in patient_diagnoses:
            if pd.isna(diagnosis):
                continue
                
            diagnosis = str(diagnosis).strip()
            
            for condition, info in CHARLSON_ICD9_MAPPING.items():
                if condition in found_conditions:
                    continue  # Already counted this condition
                    
                for code in info['codes']:
                    if diagnosis.startswith(code):
                        charlson_score += info['weight']
                        found_conditions.add(condition)
                        break
        
        return charlson_score
    
    def calculate_emergency_visits(self, subject_id: int, current_admittime: datetime) -> int:
        """Count emergency visits in 6 months prior to current admission."""
        if pd.isna(current_admittime):
            return 0
        
        # Define 6 months prior period
        six_months_ago = current_admittime - timedelta(days=180)
        
        # Count emergency admissions in prior 6 months
        prior_emergency_visits = self.admissions[
            (self.admissions['subject_id'] == subject_id) &
            (self.admissions['admittime'] >= six_months_ago) &
            (self.admissions['admittime'] < current_admittime) &
            (self.admissions['admission_type'].str.contains('EMERGENCY', na=False))
        ]
        
        return len(prior_emergency_visits)
    
    def create_readmission_labels(self, df: pd.DataFrame) -> pd.Series:
        """Create 30-day readmission labels."""
        logger.info("Creating 30-day readmission labels...")
        
        readmission_labels = []
        
        for idx, admission in df.iterrows():
            subject_id = admission['subject_id']
            discharge_time = admission['dischtime']
            
            if pd.isna(discharge_time):
                readmission_labels.append(0)
                continue
            
            # Check for readmissions within 30 days
            future_admissions = self.admissions[
                (self.admissions['subject_id'] == subject_id) &
                (self.admissions['admittime'] > discharge_time) &
                (self.admissions['admittime'] <= discharge_time + timedelta(days=30))
            ]
            
            # Label as 1 if readmitted within 30 days, 0 otherwise
            readmission_labels.append(1 if len(future_admissions) > 0 else 0)
        
        return pd.Series(readmission_labels, index=df.index)
    
    def preprocess_data(self) -> pd.DataFrame:
        """Complete data preprocessing pipeline."""
        if self.admissions is None:
            self.load_raw_data()
        
        logger.info("Starting data preprocessing...")
        
        # Filter admissions with valid discharge times
        valid_admissions = self.admissions.dropna(subset=['dischtime']).copy()
        
        # Calculate basic features
        valid_admissions['los_days'] = (
            valid_admissions['dischtime'] - valid_admissions['admittime']
        ).dt.days
        
        valid_admissions['age'] = self.calculate_age()
        
        # Add encoded categorical features
        encoded_features = self.encode_categorical_features()
        for feature_name, feature_values in encoded_features.items():
            valid_admissions[feature_name] = feature_values
        
        # Calculate Charlson scores
        charlson_scores = []
        for idx, row in valid_admissions.iterrows():
            score = self.calculate_charlson_score(row['subject_id'], row['hadm_id'])
            charlson_scores.append(score)
        valid_admissions['charlson_score'] = charlson_scores
        
        # Calculate emergency visits
        emergency_visits = []
        for idx, row in valid_admissions.iterrows():
            visits = self.calculate_emergency_visits(row['subject_id'], row['admittime'])
            emergency_visits.append(visits)
        valid_admissions['emergency_visits_6m'] = emergency_visits
        
        # Create readmission labels
        valid_admissions['readmission_30d'] = self.create_readmission_labels(valid_admissions)
        
        # Remove patients who died in hospital (can't have readmissions)
        final_data = valid_admissions[valid_admissions['hospital_expire_flag'] == 0].copy()
        
        logger.info(f"Preprocessing complete. Final dataset: {len(final_data)} admissions")
        logger.info(f"30-day readmission rate: {final_data['readmission_30d'].mean():.3f}")
        
        return final_data


def load_preprocessed_data() -> pd.DataFrame:
    """Convenience function to load and preprocess MIMIC-III data."""
    loader = MIMICDataLoader()
    return loader.preprocess_data()