"""
Configuration settings for the readmission prediction project.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "demo"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
REPORTS_DIR = RESULTS_DIR / "reports"

# Ensure directories exist
for dir_path in [RESULTS_DIR, MODELS_DIR, PREDICTIONS_DIR, REPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Model configurations
MODEL_CONFIGS = {
    'lace': {
        'name': 'LACE Index Logistic Regression',
        'algorithm': 'logistic_regression',
        'features': ['L_score', 'A_score', 'C_score', 'E_score', 'LACE_score', 'los_days', 'charlson_score'],
        'hyperparameters': {
            'C': 1.0,
            'solver': 'liblinear',
            'random_state': 42,
            'max_iter': 1000,
            'class_weight': 'balanced'
        }
    },
    'random_forest': {
        'name': 'Random Forest Classifier',
        'algorithm': 'random_forest',
        'features': ['L_score', 'A_score', 'C_score', 'E_score', 'LACE_score', 'los_days', 'charlson_score', 
                    'age', 'gender_encoded', 'admission_type_encoded'],
        'hyperparameters': {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'class_weight': 'balanced'
        }
    }
}

# Evaluation settings
EVALUATION_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'cross_validation_folds': 5,
    'metrics': ['auc_roc', 'auc_pr', 'accuracy', 'precision', 'recall', 'f1', 'sensitivity', 'specificity'],
    'classification_threshold': 0.5
}

# LACE index scoring parameters
LACE_SCORING = {
    'length_of_stay': {
        'thresholds': [1, 2, 3, 4, 7, 14],
        'scores': [0, 1, 2, 3, 4, 5, 7]
    },
    'acuteness': {
        'emergency_urgent': 3,
        'elective': 0,
        'other': 1
    },
    'comorbidity_cap': 5,
    'emergency_visits': {
        'thresholds': [0, 1, 2, 3, 4],
        'scores': [0, 1, 2, 3, 4]
    }
}

# Charlson Comorbidity Index mapping
CHARLSON_ICD9_MAPPING = {
    'myocardial_infarction': {'codes': ['410', '412'], 'weight': 1},
    'congestive_heart_failure': {'codes': ['39891', '40201', '40211', '40291', '40401', '40403', 
                                          '40411', '40413', '40491', '40493', '4254', '4255', 
                                          '4257', '4258', '4259', '428'], 'weight': 1},
    'peripheral_vascular_disease': {'codes': ['0930', '4373', '440', '441', '4431', '4432', 
                                             '4438', '4439', '4471', '5571', '5579', 'V434'], 'weight': 1},
    'cerebrovascular_disease': {'codes': ['36234', '430', '431', '432', '433', '434', '435', '436', '437', '438'], 'weight': 1},
    'dementia': {'codes': ['290', '2941', '3312'], 'weight': 1},
    'copd': {'codes': ['4168', '4169', '490', '491', '492', '493', '494', '495', '496', '500', '501', '502', '503', '504', '505'], 'weight': 1},
    'rheumatic_disease': {'codes': ['4465', '7100', '7101', '7102', '7103', '7104', '7140', '7141', '7142', '725'], 'weight': 1},
    'peptic_ulcer': {'codes': ['531', '532', '533', '534'], 'weight': 1},
    'mild_liver_disease': {'codes': ['07022', '07023', '07032', '07033', '07044', '07054', '0706', '0709', '570', '571', '5733', '5734', '5738', '5739', 'V427'], 'weight': 1},
    'diabetes_without_complications': {'codes': ['2500', '2501', '2502', '2503', '2508', '2509'], 'weight': 1},
    'diabetes_with_complications': {'codes': ['2504', '2505', '2506', '2507'], 'weight': 2},
    'hemiplegia': {'codes': ['3341', '342', '343', '3440', '3441', '3442', '3443', '3444', '3445', '3446', '3449'], 'weight': 2},
    'renal_disease': {'codes': ['40301', '40311', '40391', '40402', '40403', '40412', '40413', '40492', '40493', '582', '5830', '5831', '5832', '5833', '5834', '5835', '5836', '5837', '585', '586', '5880', 'V420', 'V451'], 'weight': 2},
    'cancer': {'codes': ['140', '141', '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154', '155', '156', '157', '158', '159', '160', '161', '162', '163', '164', '165', '170', '171', '172', '174', '175', '176', '179', '180', '181', '182', '183', '184', '185', '186', '187', '188', '189', '190', '191', '192', '193', '194', '195', '200', '201', '202', '203', '204', '205', '206', '207', '208', '2386'], 'weight': 2},
    'moderate_severe_liver_disease': {'codes': ['4560', '4561', '4562', '5722', '5723', '5724', '5728'], 'weight': 3},
    'metastatic_cancer': {'codes': ['196', '197', '198', '199'], 'weight': 6},
    'aids': {'codes': ['042', '043', '044'], 'weight': 6}
}