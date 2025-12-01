"""
LACE Index Baseline Model for Hospital Readmission Prediction

This script implements the LACE index (Length of stay, Acuteness, Comorbidities, Emergency visits) 
as a baseline model for predicting hospital readmissions using MIMIC-III data.

LACE Index Components:
- L (Length of stay): 0-7 points based on days
- A (Acuteness): 0-3 points based on admission type  
- C (Comorbidities): 0-5 points based on Charlson Comorbidity Index
- E (Emergency visits): 0-4 points based on ED visits in prior 6 months

Total LACE score: 0-19 points (higher = higher readmission risk)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class LACEIndexCalculator:
    def __init__(self, data_path='demo/'):
        """Initialize with path to MIMIC-III data files"""
        self.data_path = data_path
        self.charlson_icd9_mapping = self._get_charlson_mapping()
        
    def _get_charlson_mapping(self):
        """
        Charlson Comorbidity Index ICD-9 code mappings
        Returns dictionary mapping condition categories to ICD-9 codes and weights
        """
        return {
            'myocardial_infarction': {
                'codes': ['410', '412'],
                'weight': 1
            },
            'congestive_heart_failure': {
                'codes': ['39891', '40201', '40211', '40291', '40401', '40403', '40411', '40413', '40491', '40493', '4254', '4255', '4257', '4258', '4259', '428'],
                'weight': 1
            },
            'peripheral_vascular_disease': {
                'codes': ['0930', '4373', '440', '441', '4431', '4432', '4438', '4439', '4471', '5571', '5579', 'V434'],
                'weight': 1
            },
            'cerebrovascular_disease': {
                'codes': ['36234', '430', '431', '432', '433', '434', '435', '436', '437', '438'],
                'weight': 1
            },
            'dementia': {
                'codes': ['290', '2941', '3312'],
                'weight': 1
            },
            'copd': {
                'codes': ['4168', '4169', '490', '491', '492', '493', '494', '495', '496', '500', '501', '502', '503', '504', '505'],
                'weight': 1
            },
            'rheumatic_disease': {
                'codes': ['4465', '7100', '7101', '7102', '7103', '7104', '7140', '7141', '7142', '725'],
                'weight': 1
            },
            'peptic_ulcer': {
                'codes': ['531', '532', '533', '534'],
                'weight': 1
            },
            'mild_liver_disease': {
                'codes': ['07022', '07023', '07032', '07033', '07044', '07054', '0706', '0709', '570', '571', '5733', '5734', '5738', '5739', 'V427'],
                'weight': 1
            },
            'diabetes_without_complications': {
                'codes': ['2500', '2501', '2502', '2503', '2508', '2509'],
                'weight': 1
            },
            'diabetes_with_complications': {
                'codes': ['2504', '2505', '2506', '2507'],
                'weight': 2
            },
            'hemiplegia': {
                'codes': ['3341', '342', '343', '3440', '3441', '3442', '3443', '3444', '3445', '3446', '3449'],
                'weight': 2
            },
            'renal_disease': {
                'codes': ['40301', '40311', '40391', '40402', '40403', '40412', '40413', '40492', '40493', '582', '5830', '5831', '5832', '5833', '5834', '5835', '5836', '5837', '585', '586', '5880', 'V420', 'V451'],
                'weight': 2
            },
            'cancer': {
                'codes': ['140', '141', '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154', '155', '156', '157', '158', '159', '160', '161', '162', '163', '164', '165', '170', '171', '172', '174', '175', '176', '179', '180', '181', '182', '183', '184', '185', '186', '187', '188', '189', '190', '191', '192', '193', '194', '195', '200', '201', '202', '203', '204', '205', '206', '207', '208', '2386'],
                'weight': 2
            },
            'moderate_severe_liver_disease': {
                'codes': ['4560', '4561', '4562', '5722', '5723', '5724', '5728'],
                'weight': 3
            },
            'metastatic_cancer': {
                'codes': ['196', '197', '198', '199'],
                'weight': 6
            },
            'aids': {
                'codes': ['042', '043', '044'],
                'weight': 6
            }
        }
    
    def load_data(self):
        """Load all necessary MIMIC-III tables"""
        print("Loading MIMIC-III data...")
        
        # Load main tables
        self.admissions = pd.read_csv(f'{self.data_path}/ADMISSIONS.csv')
        self.patients = pd.read_csv(f'{self.data_path}/PATIENTS.csv')
        self.diagnoses = pd.read_csv(f'{self.data_path}/DIAGNOSES_ICD.csv')
        
        # Convert datetime columns
        datetime_cols = ['admittime', 'dischtime', 'deathtime', 'edregtime', 'edouttime']
        for col in datetime_cols:
            if col in self.admissions.columns:
                self.admissions[col] = pd.to_datetime(self.admissions[col])
        
        self.patients['dob'] = pd.to_datetime(self.patients['dob'])
        self.patients['dod'] = pd.to_datetime(self.patients['dod'])
        
        print(f"Loaded {len(self.admissions)} admissions for {self.admissions['subject_id'].nunique()} patients")
        
    def calculate_length_of_stay_score(self, los_days):
        """
        Calculate Length of Stay component (0-7 points)
        Based on LACE index: 
        - < 1 day: 0 points
        - 1 day: 1 point  
        - 2 days: 2 points
        - 3 days: 3 points
        - 4-6 days: 4 points
        - 7-13 days: 5 points  
        - >= 14 days: 7 points
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
    
    def calculate_acuteness_score(self, admission_type):
        """
        Calculate Acuteness component (0-3 points)
        Based on admission type:
        - Emergency: 3 points
        - Urgent: 2 points  
        - Elective: 0 points
        """
        if pd.isna(admission_type):
            return 0
        
        admission_type = str(admission_type).upper()
        if 'EMERGENCY' in admission_type or 'URGENT' in admission_type:
            return 3
        elif 'ELECTIVE' in admission_type:
            return 0
        else:
            return 1  # Default for other types
    
    def calculate_charlson_score(self, subject_id, hadm_id):
        """
        Calculate Charlson Comorbidity Index from ICD-9 diagnoses
        Returns weighted sum of comorbidities
        """
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
            
            for condition, info in self.charlson_icd9_mapping.items():
                if condition in found_conditions:
                    continue  # Already counted this condition
                    
                for code in info['codes']:
                    if diagnosis.startswith(code):
                        charlson_score += info['weight']
                        found_conditions.add(condition)
                        break
        
        return charlson_score
    
    def calculate_comorbidity_score(self, charlson_score):
        """
        Convert Charlson score to LACE Comorbidity component (0-5 points)
        - 0: 0 points
        - 1: 1 point
        - 2: 2 points  
        - 3: 3 points
        - 4: 4 points
        - >= 5: 5 points
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
    
    def calculate_emergency_visits_score(self, subject_id, current_admittime):
        """
        Calculate Emergency visits component (0-4 points)
        Count emergency visits in 6 months prior to current admission
        """
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
        
        visit_count = len(prior_emergency_visits)
        
        # Convert to LACE score
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
    
    def calculate_lace_scores(self):
        """Calculate LACE scores for all admissions"""
        print("Calculating LACE scores...")
        
        lace_data = []
        
        for idx, admission in self.admissions.iterrows():
            if pd.isna(admission['dischtime']):
                continue  # Skip admissions without discharge time
                
            # Calculate length of stay in days
            if not pd.isna(admission['admittime']) and not pd.isna(admission['dischtime']):
                los_days = (admission['dischtime'] - admission['admittime']).days
            else:
                los_days = 0
            
            # Calculate each LACE component
            l_score = self.calculate_length_of_stay_score(los_days)
            a_score = self.calculate_acuteness_score(admission['admission_type'])
            
            charlson_score = self.calculate_charlson_score(admission['subject_id'], admission['hadm_id'])
            c_score = self.calculate_comorbidity_score(charlson_score)
            
            e_score = self.calculate_emergency_visits_score(admission['subject_id'], admission['admittime'])
            
            # Calculate total LACE score
            total_lace = l_score + a_score + c_score + e_score
            
            lace_data.append({
                'subject_id': admission['subject_id'],
                'hadm_id': admission['hadm_id'],
                'admittime': admission['admittime'],
                'dischtime': admission['dischtime'],
                'los_days': los_days,
                'admission_type': admission['admission_type'],
                'hospital_expire_flag': admission['hospital_expire_flag'],
                'L_score': l_score,
                'A_score': a_score,
                'C_score': c_score,
                'charlson_score': charlson_score,
                'E_score': e_score,
                'LACE_score': total_lace
            })
        
        self.lace_df = pd.DataFrame(lace_data)
        print(f"Calculated LACE scores for {len(self.lace_df)} admissions")
        
        return self.lace_df
    
    def create_readmission_labels(self):
        """
        Create 30-day readmission labels
        For each admission, check if patient was readmitted within 30 days
        """
        print("Creating 30-day readmission labels...")
        
        readmission_labels = []
        
        for idx, admission in self.lace_df.iterrows():
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
        
        self.lace_df['readmission_30d'] = readmission_labels
        
        readmission_rate = self.lace_df['readmission_30d'].mean()
        print(f"30-day readmission rate: {readmission_rate:.3f} ({self.lace_df['readmission_30d'].sum()} out of {len(self.lace_df)} admissions)")
        
        return self.lace_df

class LACEBaselineModel:
    def __init__(self, lace_df):
        self.lace_df = lace_df
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        
    def prepare_features(self):
        """Prepare features for modeling"""
        # Features: Individual LACE components + total score
        feature_columns = ['L_score', 'A_score', 'C_score', 'E_score', 'LACE_score', 'los_days', 'charlson_score']
        
        # Remove any rows with missing readmission labels
        clean_df = self.lace_df.dropna(subset=['readmission_30d'])
        
        X = clean_df[feature_columns].fillna(0)
        y = clean_df['readmission_30d']
        
        return X, y
    
    def train_model(self, test_size=0.2, random_state=42):
        """Train linear regression model"""
        print("\nTraining LACE baseline model...")
        
        X, y = self.prepare_features()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        # Store results
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        self.y_train_pred, self.y_test_pred = y_train_pred, y_test_pred
        
        return X_train, X_test, y_train, y_test, y_train_pred, y_test_pred
    
    def evaluate_model(self):
        """Evaluate model performance"""
        print("\nModel Evaluation:")
        print("="*50)
        
        # Regression metrics
        train_mse = mean_squared_error(self.y_train, self.y_train_pred)
        test_mse = mean_squared_error(self.y_test, self.y_test_pred)
        train_r2 = r2_score(self.y_train, self.y_train_pred)
        test_r2 = r2_score(self.y_test, self.y_test_pred)
        
        print(f"Regression Metrics:")
        print(f"Train MSE: {train_mse:.4f}")
        print(f"Test MSE:  {test_mse:.4f}")
        print(f"Train R²:  {train_r2:.4f}")
        print(f"Test R²:   {test_r2:.4f}")
        
        # Classification metrics (using 0.5 threshold)
        y_train_binary = (self.y_train_pred >= 0.5).astype(int)
        y_test_binary = (self.y_test_pred >= 0.5).astype(int)
        
        train_acc = accuracy_score(self.y_train, y_train_binary)
        test_acc = accuracy_score(self.y_test, y_test_binary)
        
        print(f"\nClassification Metrics (threshold=0.5):")
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy:  {test_acc:.4f}")
        
        # AUC if we have binary classification
        try:
            train_auc = roc_auc_score(self.y_train, self.y_train_pred)
            test_auc = roc_auc_score(self.y_test, self.y_test_pred)
            print(f"Train AUC: {train_auc:.4f}")
            print(f"Test AUC:  {test_auc:.4f}")
        except ValueError:
            print("AUC calculation failed (may need more diverse predictions)")
        
        # Feature importance
        print(f"\nFeature Importance (Coefficients):")
        feature_names = ['L_score', 'A_score', 'C_score', 'E_score', 'LACE_score', 'los_days', 'charlson_score']
        for feature, coef in zip(feature_names, self.model.coef_):
            print(f"{feature:15s}: {coef:8.4f}")
        
        print(f"Intercept: {self.model.intercept_:.4f}")
    
    def plot_results(self):
        """Plot model results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. LACE Score Distribution by Readmission Status
        axes[0,0].hist([self.lace_df[self.lace_df['readmission_30d']==0]['LACE_score'], 
                       self.lace_df[self.lace_df['readmission_30d']==1]['LACE_score']], 
                      bins=20, alpha=0.7, label=['No Readmission', 'Readmission'])
        axes[0,0].set_xlabel('LACE Score')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].set_title('LACE Score Distribution by Readmission Status')
        axes[0,0].legend()
        
        # 2. Predicted vs Actual
        axes[0,1].scatter(self.y_test, self.y_test_pred, alpha=0.6)
        axes[0,1].plot([0, 1], [0, 1], 'r--', lw=2)
        axes[0,1].set_xlabel('Actual Readmission')
        axes[0,1].set_ylabel('Predicted Readmission')
        axes[0,1].set_title('Predicted vs Actual (Test Set)')
        
        # 3. Residuals
        residuals = self.y_test - self.y_test_pred
        axes[1,0].scatter(self.y_test_pred, residuals, alpha=0.6)
        axes[1,0].axhline(y=0, color='r', linestyle='--')
        axes[1,0].set_xlabel('Predicted Readmission')
        axes[1,0].set_ylabel('Residuals')
        axes[1,0].set_title('Residual Plot')
        
        # 4. LACE Components by Readmission
        lace_components = ['L_score', 'A_score', 'C_score', 'E_score']
        mean_scores = []
        for component in lace_components:
            no_readmit = self.lace_df[self.lace_df['readmission_30d']==0][component].mean()
            readmit = self.lace_df[self.lace_df['readmission_30d']==1][component].mean()
            mean_scores.append([no_readmit, readmit])
        
        mean_scores = np.array(mean_scores).T
        x = np.arange(len(lace_components))
        width = 0.35
        
        axes[1,1].bar(x - width/2, mean_scores[0], width, label='No Readmission', alpha=0.7)
        axes[1,1].bar(x + width/2, mean_scores[1], width, label='Readmission', alpha=0.7)
        axes[1,1].set_xlabel('LACE Components')
        axes[1,1].set_ylabel('Mean Score')
        axes[1,1].set_title('Mean LACE Component Scores by Readmission Status')
        axes[1,1].set_xticks(x)
        axes[1,1].set_xticklabels(lace_components)
        axes[1,1].legend()
        
        plt.tight_layout()
        plt.show()
    
    def save_results(self, filename='lace_baseline_results.csv'):
        """Save LACE scores and predictions to CSV"""
        results_df = self.lace_df.copy()
        
        # Add predictions for the test set
        test_predictions = pd.Series(index=self.X_test.index, data=self.y_test_pred, name='predicted_readmission')
        results_df = results_df.join(test_predictions, how='left')
        
        results_df.to_csv(filename, index=False)
        print(f"\nResults saved to {filename}")

def main():
    """Main execution function"""
    print("LACE Index Baseline Model for Hospital Readmission Prediction")
    print("="*60)
    
    # Initialize calculator
    calculator = LACEIndexCalculator(data_path='demo')
    
    # Load data and calculate LACE scores
    calculator.load_data()
    lace_df = calculator.calculate_lace_scores()
    lace_df = calculator.create_readmission_labels()
    
    # Display summary statistics
    print(f"\nLACE Score Summary:")
    print(f"Mean LACE score: {lace_df['LACE_score'].mean():.2f}")
    print(f"LACE score range: {lace_df['LACE_score'].min()}-{lace_df['LACE_score'].max()}")
    print(f"Standard deviation: {lace_df['LACE_score'].std():.2f}")
    
    print(f"\nLACE Components Summary:")
    for component in ['L_score', 'A_score', 'C_score', 'E_score']:
        print(f"{component}: mean={lace_df[component].mean():.2f}, std={lace_df[component].std():.2f}")
    
    # Train baseline model
    baseline_model = LACEBaselineModel(lace_df)
    baseline_model.train_model()
    baseline_model.evaluate_model()
    
    # Plot results and save
    try:
        baseline_model.plot_results()
    except Exception as e:
        print(f"Could not generate plots: {e}")
    
    baseline_model.save_results()
    
    print(f"\nBaseline model training complete!")
    
if __name__ == "__main__":
    main()