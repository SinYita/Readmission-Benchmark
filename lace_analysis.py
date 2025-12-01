"""
LACE Index Baseline Model - Analysis Report
==========================================

This report provides additional analysis and interpretation of the LACE baseline model results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def load_and_analyze_results():
    """Load results and provide detailed analysis"""
    
    # Load the results
    try:
        results_df = pd.read_csv('lace_baseline_results.csv')
        print("LACE Baseline Model - Detailed Analysis Report")
        print("=" * 50)
        
        # Basic statistics
        print(f"\nDataset Overview:")
        print(f"Total admissions: {len(results_df)}")
        print(f"Patients: {results_df['subject_id'].nunique()}")
        print(f"30-day readmissions: {results_df['readmission_30d'].sum()} ({results_df['readmission_30d'].mean()*100:.1f}%)")
        
        # LACE score distribution analysis
        print(f"\nLACE Score Analysis:")
        print(f"Mean LACE score: {results_df['LACE_score'].mean():.2f} ± {results_df['LACE_score'].std():.2f}")
        print(f"Range: {results_df['LACE_score'].min()}-{results_df['LACE_score'].max()}")
        
        # Risk stratification
        print(f"\nRisk Stratification by LACE Score:")
        risk_bins = [0, 7, 10, 13, float('inf')]
        risk_labels = ['Low (0-7)', 'Medium (8-10)', 'High (11-13)', 'Very High (14+)']
        results_df['risk_category'] = pd.cut(results_df['LACE_score'], bins=risk_bins, labels=risk_labels, right=False)
        
        risk_analysis = results_df.groupby('risk_category').agg({
            'readmission_30d': ['count', 'sum', 'mean']
        }).round(3)
        
        risk_analysis.columns = ['Total', 'Readmissions', 'Rate']
        print(risk_analysis)
        
        # Component analysis
        print(f"\nLACE Component Analysis by Readmission Status:")
        component_analysis = results_df.groupby('readmission_30d')[['L_score', 'A_score', 'C_score', 'E_score', 'LACE_score']].mean()
        component_analysis.index = ['No Readmission', 'Readmission']
        print(component_analysis.round(2))
        
        # Length of stay analysis
        print(f"\nLength of Stay Analysis:")
        print(f"Mean LOS (no readmission): {results_df[results_df['readmission_30d']==0]['los_days'].mean():.1f} days")
        print(f"Mean LOS (readmission): {results_df[results_df['readmission_30d']==1]['los_days'].mean():.1f} days")
        
        # Admission type analysis
        print(f"\nAdmission Type Distribution:")
        admission_analysis = pd.crosstab(results_df['admission_type'], results_df['readmission_30d'], margins=True, normalize='index')
        print(admission_analysis.round(3))
        
        # Model performance by risk category (if predictions available)
        if 'predicted_readmission' in results_df.columns:
            test_data = results_df.dropna(subset=['predicted_readmission'])
            if len(test_data) > 0:
                print(f"\nModel Performance by Risk Category:")
                test_data['pred_binary'] = (test_data['predicted_readmission'] >= 0.5).astype(int)
                
                for category in risk_labels:
                    if category in test_data['risk_category'].values:
                        cat_data = test_data[test_data['risk_category'] == category]
                        if len(cat_data) > 0:
                            accuracy = (cat_data['readmission_30d'] == cat_data['pred_binary']).mean()
                            print(f"{category}: n={len(cat_data)}, accuracy={accuracy:.3f}")
        
        # Generate visualizations
        generate_visualization_plots(results_df)
        
    except FileNotFoundError:
        print("Results file not found. Please run the main LACE model first.")
        return None
    
    return results_df

def generate_visualization_plots(results_df):
    """Generate additional visualization plots"""
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. LACE Score by Risk Category
    sns.boxplot(data=results_df, x='risk_category', y='LACE_score', ax=axes[0,0])
    axes[0,0].set_title('LACE Score Distribution by Risk Category')
    axes[0,0].set_xlabel('Risk Category')
    axes[0,0].set_ylabel('LACE Score')
    
    # 2. Readmission Rate by Risk Category
    risk_rates = results_df.groupby('risk_category')['readmission_30d'].mean()
    axes[0,1].bar(range(len(risk_rates)), risk_rates.values, alpha=0.7)
    axes[0,1].set_title('30-day Readmission Rate by Risk Category')
    axes[0,1].set_xlabel('Risk Category')
    axes[0,1].set_ylabel('Readmission Rate')
    axes[0,1].set_xticks(range(len(risk_rates)))
    axes[0,1].set_xticklabels(risk_rates.index, rotation=45)
    
    # 3. Length of Stay Distribution
    axes[0,2].hist([results_df[results_df['readmission_30d']==0]['los_days'],
                    results_df[results_df['readmission_30d']==1]['los_days']],
                   bins=15, alpha=0.7, label=['No Readmission', 'Readmission'])
    axes[0,2].set_title('Length of Stay Distribution')
    axes[0,2].set_xlabel('Length of Stay (days)')
    axes[0,2].set_ylabel('Frequency')
    axes[0,2].legend()
    
    # 4. LACE Components Heatmap
    component_corr = results_df[['L_score', 'A_score', 'C_score', 'E_score', 'LACE_score', 'readmission_30d']].corr()
    sns.heatmap(component_corr, annot=True, cmap='coolwarm', center=0, ax=axes[1,0])
    axes[1,0].set_title('Correlation Matrix: LACE Components')
    
    # 5. Admission Type vs Readmission
    admission_crosstab = pd.crosstab(results_df['admission_type'], results_df['readmission_30d'])
    admission_crosstab.plot(kind='bar', ax=axes[1,1], alpha=0.7)
    axes[1,1].set_title('Admission Type vs Readmission')
    axes[1,1].set_xlabel('Admission Type')
    axes[1,1].set_ylabel('Count')
    axes[1,1].legend(['No Readmission', 'Readmission'])
    axes[1,1].tick_params(axis='x', rotation=45)
    
    # 6. Charlson Score Distribution
    axes[1,2].hist([results_df[results_df['readmission_30d']==0]['charlson_score'],
                    results_df[results_df['readmission_30d']==1]['charlson_score']],
                   bins=10, alpha=0.7, label=['No Readmission', 'Readmission'])
    axes[1,2].set_title('Charlson Comorbidity Score Distribution')
    axes[1,2].set_xlabel('Charlson Score')
    axes[1,2].set_ylabel('Frequency')
    axes[1,2].legend()
    
    plt.tight_layout()
    plt.savefig('lace_analysis_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nVisualization plots saved as 'lace_analysis_plots.png'")

def generate_summary_report():
    """Generate a comprehensive summary report"""
    
    report = """
LACE Index Baseline Model - Summary Report
==========================================

METHODOLOGY:
The LACE index is a validated clinical prediction tool for hospital readmission risk.
It combines four key factors:

L - Length of stay (0-7 points)
  • <1 day: 0 points
  • 1 day: 1 point  
  • 2 days: 2 points
  • 3 days: 3 points
  • 4-6 days: 4 points
  • 7-13 days: 5 points
  • ≥14 days: 7 points

A - Acuteness of admission (0-3 points)
  • Emergency/Urgent: 3 points
  • Elective: 0 points

C - Comorbidities via Charlson Index (0-5 points)
  • Based on ICD-9 diagnoses
  • Weighted by condition severity
  • Capped at 5 points for LACE

E - Emergency visits in prior 6 months (0-4 points)
  • 0 visits: 0 points
  • 1 visit: 1 point
  • 2 visits: 2 points
  • 3 visits: 3 points
  • ≥4 visits: 4 points

RESULTS INTERPRETATION:
• The model shows low predictive performance (Test R² = -0.15, AUC = 0.125)
• This suggests readmission prediction is challenging with LACE alone
• The low readmission rate (8.5%) creates class imbalance issues
• LACE may work better for risk stratification than binary prediction

CLINICAL INSIGHTS:
• Mean LACE score: 10.0 ± 3.0 (moderate risk range)
• Emergency admissions comprise majority of cases
• Length of stay is primary contributor to LACE score
• Comorbidities show negative correlation (unexpected, may need review)

RECOMMENDATIONS FOR IMPROVEMENT:
1. Address class imbalance with sampling techniques
2. Add temporal features (time-to-readmission, seasonal effects)
3. Include medication, procedure, and vital sign data
4. Consider ensemble methods combining LACE with other predictors
5. Validate on larger, more diverse patient populations
6. Use appropriate evaluation metrics for imbalanced data (precision, recall, F1)

LIMITATIONS:
• Small sample size (129 admissions, 11 readmissions)
• MIMIC-III demo data may not be representative
• Simple linear regression may be insufficient for complex relationships
• Missing data handling could be improved
• No external validation performed

CONCLUSION:
While the LACE index provides a structured approach to readmission risk assessment,
this implementation shows that additional features and more sophisticated modeling
approaches are needed for accurate readmission prediction in this dataset.
"""
    
    # Save report to file
    with open('lace_model_report.txt', 'w') as f:
        f.write(report)
    
    print(report)
    print(f"\nComplete report saved to 'lace_model_report.txt'")

def main():
    """Run complete analysis"""
    results_df = load_and_analyze_results()
    if results_df is not None:
        generate_summary_report()
    
    print(f"\n" + "="*50)
    print("Analysis complete! Files generated:")
    print("• lace_baseline_results.csv - Raw results with LACE scores")
    print("• lace_analysis_plots.png - Visualization plots")  
    print("• lace_model_report.txt - Comprehensive summary report")
    print("="*50)

if __name__ == "__main__":
    main()