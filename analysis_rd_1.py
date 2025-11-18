"""
TGIF A/B Test Analysis - Complete Standalone Script

This script performs the complete analysis of TGIF frozen snack A/B test data.
It generates all reports, matrices, and summary tables.

Requirements:
- pandas
- numpy
- scipy

Usage:
    python tgif_complete_analysis.py

Outputs:
    - tgif_ab_test_results.md (detailed results for all tests)
    - complete_test_matrices.md (two matrices per test)
    - summary_table_with_segments.csv (summary with demographic slices)
    - summary_table_with_segments.md (markdown version of summary)
    - tgif_processed_data.csv (data with binary columns for further analysis)
"""

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

# Path to your data file
DATA_FILE = 'tgif_firstround_raw.csv'

# Test numbers to analyze
TEST_NUMBERS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 
                19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def convert_to_binary(value):
    """Convert 1-5 scale to binary (0 for 1-3, 1 for 4-5)"""
    if pd.isna(value) or value == '' or value == ' ':
        return np.nan
    try:
        score = int(value)
        if score >= 4:
            return 1
        elif score >= 1:
            return 0
        else:
            return np.nan
    except:
        return np.nan

def get_significance_marker(p_value):
    """Get significance marker for p-value"""
    if p_value < 0.05:
        return "✓"
    elif p_value < 0.10:
        return "~"
    elif p_value < 0.20:
        return "°"
    else:
        return "✗"

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_ab_test(test_num, df):
    """Analyze DV1 (purchase intent) for a single A/B test"""
    
    control_col = f'dv{test_num}_1_binary'
    test_assignment_col = f'test{test_num}'
    
    test_df = df[df[test_assignment_col].notna() & 
                 (df[test_assignment_col] != '') & 
                 (df[test_assignment_col] != ' ')].copy()
    
    if len(test_df) == 0:
        return None
    
    test_df['treatment'] = test_df[test_assignment_col].astype(str).str.extract(r'([A-Z])', expand=False)
    
    # Control data
    control_data = test_df[test_df['treatment'] == 'A'][control_col].dropna()
    
    if len(control_data) == 0:
        return None
    
    control_n = len(control_data)
    control_intent = control_data.sum()
    control_pct = (control_intent / control_n) * 100
    
    results = {
        'test_number': test_num,
        'control_n': control_n,
        'control_intent': control_intent,
        'control_pct': control_pct,
        'treatments': []
    }
    
    # Treatments
    treatments = sorted([t for t in test_df['treatment'].unique() 
                        if t not in ['A', None, np.nan, '']])
    
    for treatment_letter in treatments:
        treatment_data = test_df[test_df['treatment'] == treatment_letter][control_col].dropna()
        
        if len(treatment_data) == 0:
            continue
        
        treatment_n = len(treatment_data)
        treatment_intent = treatment_data.sum()
        treatment_pct = (treatment_intent / treatment_n) * 100
        
        lift = ((treatment_pct - control_pct) / control_pct * 100) if control_pct > 0 else 0
        absolute_lift = treatment_pct - control_pct
        
        contingency_table = np.array([
            [control_intent, control_n - control_intent],
            [treatment_intent, treatment_n - treatment_intent]
        ])
        
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        results['treatments'].append({
            'treatment_letter': treatment_letter,
            'treatment_n': treatment_n,
            'treatment_intent': treatment_intent,
            'treatment_pct': treatment_pct,
            'lift_pct': lift,
            'absolute_lift': absolute_lift,
            'chi2': chi2,
            'p_value': p_value,
            'is_significant': p_value < 0.05
        })
    
    results['treatments'] = sorted(results['treatments'], 
                                  key=lambda x: x['lift_pct'], reverse=True)
    
    return results


def analyze_ab_test_dv2(test_num, df):
    """Analyze DV2 (behavioral question) for a single A/B test"""
    
    dv2_col = f'dv{test_num}_2'
    test_assignment_col = f'test{test_num}'
    
    test_df = df[df[test_assignment_col].notna() & 
                 (df[test_assignment_col] != '') & 
                 (df[test_assignment_col] != ' ')].copy()
    
    if len(test_df) == 0:
        return None
    
    test_df['treatment'] = test_df[test_assignment_col].astype(str).str.extract(r'([A-Z])', expand=False)
    test_df['dv2_binary'] = test_df[dv2_col].apply(convert_to_binary)
    
    # Control data
    control_data = test_df[test_df['treatment'] == 'A']['dv2_binary'].dropna()
    
    if len(control_data) == 0:
        return None
    
    control_n = len(control_data)
    control_intent = control_data.sum()
    control_pct = (control_intent / control_n) * 100
    
    results = {
        'test_number': test_num,
        'control_n': control_n,
        'control_intent': control_intent,
        'control_pct': control_pct,
        'treatments': [],
        'is_dv2_analysis': True
    }
    
    treatments = sorted([t for t in test_df['treatment'].unique() 
                        if t not in ['A', None, np.nan, '']])
    
    for treatment_letter in treatments:
        treatment_data = test_df[test_df['treatment'] == treatment_letter]['dv2_binary'].dropna()
        
        if len(treatment_data) == 0:
            continue
        
        treatment_n = len(treatment_data)
        treatment_intent = treatment_data.sum()
        treatment_pct = (treatment_intent / treatment_n) * 100
        
        lift = ((treatment_pct - control_pct) / control_pct * 100) if control_pct > 0 else 0
        absolute_lift = treatment_pct - control_pct
        
        contingency_table = np.array([
            [control_intent, control_n - control_intent],
            [treatment_intent, treatment_n - treatment_intent]
        ])
        
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        results['treatments'].append({
            'treatment_letter': treatment_letter,
            'treatment_n': treatment_n,
            'treatment_intent': treatment_intent,
            'treatment_pct': treatment_pct,
            'lift_pct': lift,
            'absolute_lift': absolute_lift,
            'chi2': chi2,
            'p_value': p_value,
            'is_significant': p_value < 0.05
        })
    
    results['treatments'] = sorted(results['treatments'], 
                                  key=lambda x: x['lift_pct'], reverse=True)
    
    return results


def get_full_test_matrices(test_num, df):
    """Get both Matrix 1 (DV1) and Matrix 2 (DV2) for a test"""
    
    test_assignment_col = f'test{test_num}'
    
    test_df = df[df[test_assignment_col].notna() & 
                 (df[test_assignment_col] != '') & 
                 (df[test_assignment_col] != ' ')].copy()
    
    if len(test_df) == 0:
        return None
    
    test_df['treatment'] = test_df[test_assignment_col].astype(str).str.extract(r'([A-Z])', expand=False)
    
    results = {'test': test_num, 'treatments': []}
    
    # MATRIX 1: DV1 (Purchase Intent)
    dv1_col = f'dv{test_num}_1_binary'
    
    control_data = test_df[test_df['treatment'] == 'A'][dv1_col].dropna()
    
    if len(control_data) == 0:
        return None
    
    control_n = len(control_data)
    control_intent = control_data.sum()
    control_pct = (control_intent / control_n) * 100
    
    results['matrix1_control'] = {
        'treatment': 'A',
        'n': control_n,
        'pct': control_pct,
        'p_value': None
    }
    
    treatments = sorted([t for t in test_df['treatment'].unique() 
                        if t not in ['A', None, np.nan, '']])
    
    matrix1_treatments = []
    for treatment_letter in treatments:
        treatment_data = test_df[test_df['treatment'] == treatment_letter][dv1_col].dropna()
        
        if len(treatment_data) == 0:
            continue
        
        treatment_n = len(treatment_data)
        treatment_intent = treatment_data.sum()
        treatment_pct = (treatment_intent / treatment_n) * 100
        
        contingency_table = np.array([
            [control_intent, control_n - control_intent],
            [treatment_intent, treatment_n - treatment_intent]
        ])
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        matrix1_treatments.append({
            'treatment': treatment_letter,
            'n': treatment_n,
            'pct': treatment_pct,
            'p_value': p_value
        })
    
    results['matrix1_treatments'] = sorted(matrix1_treatments, key=lambda x: x['treatment'])
    
    # MATRIX 2: DV2 (Behavioral)
    dv2_col = f'dv{test_num}_2_binary'
    
    control_data_dv2 = test_df[test_df['treatment'] == 'A'][dv2_col].dropna()
    
    if len(control_data_dv2) > 0:
        control_n_dv2 = len(control_data_dv2)
        control_intent_dv2 = control_data_dv2.sum()
        control_pct_dv2 = (control_intent_dv2 / control_n_dv2) * 100
        
        results['matrix2_control'] = {
            'treatment': 'A',
            'n': control_n_dv2,
            'pct': control_pct_dv2,
            'p_value': None
        }
        
        matrix2_treatments = []
        for treatment_letter in treatments:
            treatment_data_dv2 = test_df[test_df['treatment'] == treatment_letter][dv2_col].dropna()
            
            if len(treatment_data_dv2) == 0:
                continue
            
            treatment_n_dv2 = len(treatment_data_dv2)
            treatment_intent_dv2 = treatment_data_dv2.sum()
            treatment_pct_dv2 = (treatment_intent_dv2 / treatment_n_dv2) * 100
            
            contingency_table = np.array([
                [control_intent_dv2, control_n_dv2 - control_intent_dv2],
                [treatment_intent_dv2, treatment_n_dv2 - treatment_intent_dv2]
            ])
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            
            matrix2_treatments.append({
                'treatment': treatment_letter,
                'n': treatment_n_dv2,
                'pct': treatment_pct_dv2,
                'p_value': p_value
            })
        
        results['matrix2_treatments'] = sorted(matrix2_treatments, key=lambda x: x['treatment'])
    
    return results


def analyze_test_segment(test_num, segment_df, segment_name="All"):
    """Analyze a test for a specific demographic segment"""
    
    test_assignment_col = f'test{test_num}'
    
    test_df = segment_df[segment_df[test_assignment_col].notna() & 
                         (segment_df[test_assignment_col] != '') & 
                         (segment_df[test_assignment_col] != ' ')].copy()
    
    if len(test_df) == 0:
        return None
    
    test_df['treatment'] = test_df[test_assignment_col].astype(str).str.extract(r'([A-Z])', expand=False)
    
    result = {'test': test_num, 'segment': segment_name}
    
    # DV1 Analysis
    dv1_col = f'dv{test_num}_1_binary'
    control_data_dv1 = test_df[test_df['treatment'] == 'A'][dv1_col].dropna()
    
    if len(control_data_dv1) > 0:
        control_n = len(control_data_dv1)
        control_intent = control_data_dv1.sum()
        control_pct = (control_intent / control_n) * 100
        
        result['dv1_control_pct'] = control_pct
        
        treatments = sorted([t for t in test_df['treatment'].unique() 
                            if t not in ['A', None, np.nan, '']])
        
        best_treatment = None
        best_pct = 0
        best_lift = -999999
        best_p = 1.0
        
        for treatment_letter in treatments:
            treatment_data = test_df[test_df['treatment'] == treatment_letter][dv1_col].dropna()
            
            if len(treatment_data) == 0:
                continue
            
            treatment_n = len(treatment_data)
            treatment_intent = treatment_data.sum()
            treatment_pct = (treatment_intent / treatment_n) * 100
            lift = treatment_pct - control_pct
            
            contingency_table = np.array([
                [control_intent, control_n - control_intent],
                [treatment_intent, treatment_n - treatment_intent]
            ])
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            
            if lift > best_lift:
                best_lift = lift
                best_treatment = treatment_letter
                best_pct = treatment_pct
                best_p = p_value
        
        result['dv1_best_treatment'] = best_treatment
        result['dv1_best_pct'] = best_pct
        result['dv1_net_improvement'] = best_lift
        result['dv1_p_value'] = best_p
        result['dv1_significant'] = best_p < 0.05
    
    # DV2 Analysis
    dv2_col = f'dv{test_num}_2_binary'
    control_data_dv2 = test_df[test_df['treatment'] == 'A'][dv2_col].dropna()
    
    if len(control_data_dv2) > 0:
        control_n_dv2 = len(control_data_dv2)
        control_intent_dv2 = control_data_dv2.sum()
        control_pct_dv2 = (control_intent_dv2 / control_n_dv2) * 100
        
        result['dv2_control_pct'] = control_pct_dv2
        
        best_treatment_dv2 = None
        best_pct_dv2 = 0
        best_lift_dv2 = -999999
        best_p_dv2 = 1.0
        
        for treatment_letter in treatments:
            treatment_data_dv2 = test_df[test_df['treatment'] == treatment_letter][dv2_col].dropna()
            
            if len(treatment_data_dv2) == 0:
                continue
            
            treatment_n_dv2 = len(treatment_data_dv2)
            treatment_intent_dv2 = treatment_data_dv2.sum()
            treatment_pct_dv2 = (treatment_intent_dv2 / treatment_n_dv2) * 100
            lift_dv2 = treatment_pct_dv2 - control_pct_dv2
            
            contingency_table = np.array([
                [control_intent_dv2, control_n_dv2 - control_intent_dv2],
                [treatment_intent_dv2, treatment_n_dv2 - treatment_intent_dv2]
            ])
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            
            if lift_dv2 > best_lift_dv2:
                best_lift_dv2 = lift_dv2
                best_treatment_dv2 = treatment_letter
                best_pct_dv2 = treatment_pct_dv2
                best_p_dv2 = p_value
        
        result['dv2_best_treatment'] = best_treatment_dv2
        result['dv2_best_pct'] = best_pct_dv2
        result['dv2_net_improvement'] = best_lift_dv2
        result['dv2_p_value'] = best_p_dv2
    
    return result


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    print("="*80)
    print("TGIF A/B TEST ANALYSIS")
    print("="*80)
    
    # Load data
    print("\n1. Loading data...")
    df = pd.read_csv(DATA_FILE)
    print(f"   Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Add binary columns
    print("\n2. Converting to binary (1-3=0, 4-5=1)...")
    for test_num in TEST_NUMBERS:
        df[f'dv{test_num}_1_binary'] = df[f'dv{test_num}_1'].apply(convert_to_binary)
        df[f'dv{test_num}_2_binary'] = df[f'dv{test_num}_2'].apply(convert_to_binary)
    
    # Create demographics
    print("\n3. Creating demographic segments...")
    df['age'] = 2025 - pd.to_numeric(df['q18'], errors='coerce')
    df['older_age'] = df['age'] >= 45
    df['has_children'] = pd.to_numeric(df['q20'], errors='coerce') == 1
    
    # Run primary analysis
    print("\n4. Running primary analysis (DV1)...")
    all_results = []
    
    for test_num in TEST_NUMBERS:
        result = analyze_ab_test(test_num, df)
        if result and result['treatments']:
            has_significant = any(t['is_significant'] for t in result['treatments'])
            
            if not has_significant:
                dv2_result = analyze_ab_test_dv2(test_num, df)
                if dv2_result and dv2_result['treatments']:
                    result['dv2_analysis'] = dv2_result
            
            all_results.append(result)
    
    all_results = sorted(all_results, key=lambda x: x['test_number'])
    print(f"   Analyzed {len(all_results)} tests")
    
    # Generate detailed results markdown
    print("\n5. Generating detailed results...")
    output_lines = []
    output_lines.append("# TGIF Frozen Snacks A/B Test Analysis\n")
    output_lines.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    output_lines.append("---\n")
    output_lines.append("\n## Methodology\n")
    output_lines.append("- Purchase intent measured on 1-5 scale")
    output_lines.append("- Scores 1-3 coded as 0 (no intent)")
    output_lines.append("- Scores 4-5 coded as 1 (purchase intent)")
    output_lines.append("- Chi-square tests for statistical significance")
    output_lines.append("- Lift calculated as percentage change from control\n")
    output_lines.append("**Significance Levels:**")
    output_lines.append("- ✓ p<0.05: Statistically significant")
    output_lines.append("- ~ p<0.10: Marginally significant")
    output_lines.append("- ° p<0.20: Directionally interesting")
    output_lines.append("- ✗ p≥0.20: Not significant\n")
    output_lines.append("**Dual Analysis:**")
    output_lines.append("- For tests with no significant results at p<0.05, both primary (dv1) and dv2 analyses shown")
    output_lines.append("- Primary analysis: Purchase intent question (DV1)")
    output_lines.append("- DV2 analysis: Behavioral consideration question (DV2)\n")
    output_lines.append("---\n")
    
    for result in all_results:
        test_num = result['test_number']
        has_dv2 = 'dv2_analysis' in result
        
        output_lines.append(f"\n## Test {test_num}\n")
        
        if has_dv2:
            output_lines.append("**Note:** This test had no statistically significant results at p<0.05 in the primary analysis. Both primary (dv1) and dv2 analyses are shown below.\n")
        
        if has_dv2:
            output_lines.append("### PRIMARY ANALYSIS (DV1 - Purchase Intent)\n")
        
        output_lines.append(f"**Control (A)**")
        output_lines.append(f"- Sample Size: {result['control_n']}")
        output_lines.append(f"- Purchase Intent: {result['control_intent']} respondents")
        output_lines.append(f"- Purchase Intent Rate: {result['control_pct']:.2f}%")
        output_lines.append("")
        
        if result['treatments']:
            output_lines.append("**Treatment Results (Ranked by Lift):**\n")
            
            for idx, treatment in enumerate(result['treatments'], 1):
                p_val = treatment['p_value']
                if p_val < 0.05:
                    sig_marker = "✓ (p<0.05)"
                    sig_text = "Significant"
                elif p_val < 0.10:
                    sig_marker = "~ (p<0.10)"
                    sig_text = "Marginally Significant"
                elif p_val < 0.20:
                    sig_marker = "° (p<0.20)"
                    sig_text = "Directionally Interesting"
                else:
                    sig_marker = "✗"
                    sig_text = "NOT Significant"
                
                output_lines.append(f"### {idx}. Treatment {treatment['treatment_letter']} - {sig_text} {sig_marker}")
                output_lines.append(f"- Sample Size: {treatment['treatment_n']}")
                output_lines.append(f"- Purchase Intent: {treatment['treatment_intent']} respondents")
                output_lines.append(f"- Purchase Intent Rate: {treatment['treatment_pct']:.2f}%")
                output_lines.append(f"- Lift vs Control: {treatment['lift_pct']:.2f}%")
                output_lines.append(f"- Absolute Lift: {treatment['absolute_lift']:.2f} percentage points")
                output_lines.append(f"- Chi-square: {treatment['chi2']:.4f}")
                output_lines.append(f"- P-value: {treatment['p_value']:.4f}")
                output_lines.append("")
            
            best_treatment = result['treatments'][0]
            p_val = best_treatment['p_value']
            if p_val < 0.05:
                output_lines.append(f"**Most Effective Treatment:** {best_treatment['treatment_letter']} with {best_treatment['lift_pct']:.2f}% lift (p={p_val:.4f})")
            elif p_val < 0.10:
                output_lines.append(f"**Most Effective Treatment:** {best_treatment['treatment_letter']} with {best_treatment['lift_pct']:.2f}% lift (marginally significant, p={p_val:.4f})")
            elif p_val < 0.20:
                output_lines.append(f"**Most Effective Treatment:** {best_treatment['treatment_letter']} with {best_treatment['lift_pct']:.2f}% lift (directionally interesting, p={p_val:.4f})")
            else:
                output_lines.append(f"**Most Effective Treatment:** {best_treatment['treatment_letter']} with {best_treatment['lift_pct']:.2f}% lift (NOT statistically significant, p={p_val:.4f})")
        
        # DV2 Analysis
        if has_dv2:
            dv2_result = result['dv2_analysis']
            output_lines.append("\n### DV2 ANALYSIS (Behavioral Question)\n")
            
            output_lines.append(f"**Control (A)**")
            output_lines.append(f"- Sample Size: {dv2_result['control_n']}")
            output_lines.append(f"- Response Rate (4-5): {dv2_result['control_intent']} respondents")
            output_lines.append(f"- Response Rate: {dv2_result['control_pct']:.2f}%")
            output_lines.append("")
            
            if dv2_result['treatments']:
                output_lines.append("**Treatment Results (Ranked by Lift):**\n")
                
                for idx, treatment in enumerate(dv2_result['treatments'], 1):
                    p_val = treatment['p_value']
                    if p_val < 0.05:
                        sig_marker = "✓ (p<0.05)"
                        sig_text = "Significant"
                    elif p_val < 0.10:
                        sig_marker = "~ (p<0.10)"
                        sig_text = "Marginally Significant"
                    elif p_val < 0.20:
                        sig_marker = "° (p<0.20)"
                        sig_text = "Directionally Interesting"
                    else:
                        sig_marker = "✗"
                        sig_text = "NOT Significant"
                    
                    output_lines.append(f"### {idx}. Treatment {treatment['treatment_letter']} - {sig_text} {sig_marker}")
                    output_lines.append(f"- Sample Size: {treatment['treatment_n']}")
                    output_lines.append(f"- Response Rate (4-5): {treatment['treatment_intent']} respondents")
                    output_lines.append(f"- Response Rate: {treatment['treatment_pct']:.2f}%")
                    output_lines.append(f"- Lift vs Control: {treatment['lift_pct']:.2f}%")
                    output_lines.append(f"- Absolute Lift: {treatment['absolute_lift']:.2f} percentage points")
                    output_lines.append(f"- Chi-square: {treatment['chi2']:.4f}")
                    output_lines.append(f"- P-value: {treatment['p_value']:.4f}")
                    output_lines.append("")
                
                best_treatment = dv2_result['treatments'][0]
                p_val = best_treatment['p_value']
                if p_val < 0.05:
                    output_lines.append(f"**DV2 Most Effective Treatment:** {best_treatment['treatment_letter']} with {best_treatment['lift_pct']:.2f}% lift (p={p_val:.4f})")
                elif p_val < 0.10:
                    output_lines.append(f"**DV2 Most Effective Treatment:** {best_treatment['treatment_letter']} with {best_treatment['lift_pct']:.2f}% lift (marginally significant, p={p_val:.4f})")
                elif p_val < 0.20:
                    output_lines.append(f"**DV2 Most Effective Treatment:** {best_treatment['treatment_letter']} with {best_treatment['lift_pct']:.2f}% lift (directionally interesting, p={p_val:.4f})")
                else:
                    output_lines.append(f"**DV2 Most Effective Treatment:** {best_treatment['treatment_letter']} with {best_treatment['lift_pct']:.2f}% lift (NOT statistically significant, p={p_val:.4f})")
        
        output_lines.append("\n---\n")
    
    output_lines.append(f"\n## Summary\n\nTotal tests analyzed: {len(all_results)}\n")
    
    with open('tgif_ab_test_results.md', 'w') as f:
        f.write("\n".join(output_lines))
    
    print("   Saved: tgif_ab_test_results.md")
    
    # Generate complete matrices
    print("\n6. Generating complete matrices...")
    matrices_results = []
    for test_num in TEST_NUMBERS:
        result = get_full_test_matrices(test_num, df)
        if result:
            matrices_results.append(result)
    
    matrix_lines = []
    matrix_lines.append("# Complete Test Matrices: All Tests\n")
    matrix_lines.append("This document contains two matrices for each test:")
    matrix_lines.append("- **Matrix 1 (DV1):** Purchase intent question across control and treatments")
    matrix_lines.append("- **Matrix 2 (DV2):** Behavioral consideration question across control and treatments\n")
    matrix_lines.append("Note: Each participant only sees ONE condition and answers BOTH questions about that condition.\n")
    matrix_lines.append("**Significance Markers:**")
    matrix_lines.append("- ✓ p<0.05: Statistically significant")
    matrix_lines.append("- ~ p<0.10: Marginally significant")
    matrix_lines.append("- ° p<0.20: Directionally interesting")
    matrix_lines.append("- ✗ p≥0.20: Not significant\n")
    matrix_lines.append("---\n")
    
    for result in matrices_results:
        test_num = result['test']
        matrix_lines.append(f"\n## Test {test_num}\n")
        
        # Matrix 1
        matrix_lines.append("### Matrix 1: DV1 - Purchase Intent Question\n")
        matrix_lines.append("| Treatment | N | % Rated 4-5 | P-Value | Significance |")
        matrix_lines.append("|-----------|---|-------------|---------|--------------")
        
        ctrl = result['matrix1_control']
        matrix_lines.append(f"| {ctrl['treatment']} (Control) | {ctrl['n']} | {ctrl['pct']:.2f}% | - | - |")
        
        for treatment in result['matrix1_treatments']:
            sig = get_significance_marker(treatment['p_value'])
            matrix_lines.append(f"| {treatment['treatment']} | {treatment['n']} | {treatment['pct']:.2f}% | {treatment['p_value']:.4f} | {sig} |")
        
        # Matrix 2
        if 'matrix2_control' in result:
            matrix_lines.append("\n### Matrix 2: DV2 - Behavioral Consideration Question\n")
            matrix_lines.append("| Treatment | N | % Rated 4-5 | P-Value | Significance |")
            matrix_lines.append("|-----------|---|-------------|---------|--------------")
            
            ctrl2 = result['matrix2_control']
            matrix_lines.append(f"| {ctrl2['treatment']} (Control) | {ctrl2['n']} | {ctrl2['pct']:.2f}% | - | - |")
            
            for treatment in result['matrix2_treatments']:
                sig = get_significance_marker(treatment['p_value'])
                matrix_lines.append(f"| {treatment['treatment']} | {treatment['n']} | {treatment['pct']:.2f}% | {treatment['p_value']:.4f} | {sig} |")
        
        matrix_lines.append("\n---\n")
    
    with open('complete_test_matrices.md', 'w') as f:
        f.write("\n".join(matrix_lines))
    
    print("   Saved: complete_test_matrices.md")
    
    # Generate summary table with segments
    print("\n7. Generating summary table with segments...")
    summary_results = []
    
    for test_num in TEST_NUMBERS:
        overall = analyze_test_segment(test_num, df, "All")
        if overall:
            summary_results.append(overall)
            
            # Demographic slices
            if test_num == 9:
                older_df = df[df['older_age'] == True]
                older_result = analyze_test_segment(test_num, older_df, "  └─ Older Ages (45+)")
                if older_result:
                    summary_results.append(older_result)
            
            if test_num == 12:
                children_df = df[df['has_children'] == True]
                children_result = analyze_test_segment(test_num, children_df, "  └─ Has Children")
                if children_result:
                    summary_results.append(children_result)
            
            if test_num == 34:
                older_df = df[df['older_age'] == True]
                older_result = analyze_test_segment(test_num, older_df, "  └─ Older Ages (45+)")
                if older_result:
                    summary_results.append(older_result)
    
    # CSV
    csv_lines = []
    csv_lines.append("Test,Segment,DV1_Control_%,DV1_Best_Treatment,DV1_Best_%,DV1_Net_Improvement,DV1_P_Value,DV2_Control_%,DV2_Best_Treatment,DV2_Best_%,DV2_Net_Improvement,DV2_P_Value")
    
    for result in summary_results:
        show_dv2 = not result.get('dv1_significant', False)
        
        dv2_control = f"{result['dv2_control_pct']:.2f}" if show_dv2 and 'dv2_control_pct' in result else ""
        dv2_best_treatment = result.get('dv2_best_treatment', '') if show_dv2 else ""
        dv2_best_pct = f"{result['dv2_best_pct']:.2f}" if show_dv2 and 'dv2_best_pct' in result else ""
        dv2_net_improvement = f"{result['dv2_net_improvement']:.2f}" if show_dv2 and 'dv2_net_improvement' in result else ""
        dv2_p_value = f"{result['dv2_p_value']:.4f}" if show_dv2 and 'dv2_p_value' in result else ""
        
        csv_lines.append(f"{result['test']},{result['segment']},{result['dv1_control_pct']:.2f},"
                        f"{result['dv1_best_treatment']},{result['dv1_best_pct']:.2f},"
                        f"{result['dv1_net_improvement']:.2f},{result['dv1_p_value']:.4f},"
                        f"{dv2_control},{dv2_best_treatment},{dv2_best_pct},"
                        f"{dv2_net_improvement},{dv2_p_value}")
    
    with open('summary_table_with_segments.csv', 'w') as f:
        f.write("\n".join(csv_lines))
    
    print("   Saved: summary_table_with_segments.csv")
    
    # Markdown
    md_lines = []
    md_lines.append("# Summary Table: All Tests with Demographic Segments\n")
    md_lines.append("**Note:** DV2 columns only populated when no DV1 treatment shows statistical significance (p<0.05)\n")
    md_lines.append("**Demographic segments shown for Tests 9, 12, and 34**\n")
    md_lines.append("| Test | Segment | DV1 Control % | DV1 Best Treatment | DV1 Best % | DV1 Net Improvement | DV1 P-Value | DV2 Control % | DV2 Best Treatment | DV2 Best % | DV2 Net Improvement | DV2 P-Value |")
    md_lines.append("|------|---------|---------------|--------------------|-----------|--------------------|-------------|---------------|--------------------|-----------|--------------------|-------------|")
    
    for result in summary_results:
        show_dv2 = not result.get('dv1_significant', False)
        
        p_val = result['dv1_p_value']
        sig = get_significance_marker(p_val)
        dv1_p_display = f"{p_val:.4f} {sig}"
        
        dv2_control = f"{result['dv2_control_pct']:.2f}%" if show_dv2 and 'dv2_control_pct' in result else "-"
        dv2_best_treatment = result.get('dv2_best_treatment', '-') if show_dv2 else "-"
        dv2_best_pct = f"{result['dv2_best_pct']:.2f}%" if show_dv2 and 'dv2_best_pct' in result else "-"
        dv2_net_improvement = f"{result['dv2_net_improvement']:.2f} pp" if show_dv2 and 'dv2_net_improvement' in result else "-"
        
        if show_dv2 and 'dv2_p_value' in result:
            dv2_p = result['dv2_p_value']
            dv2_sig = get_significance_marker(dv2_p)
            dv2_p_display = f"{dv2_p:.4f} {dv2_sig}"
        else:
            dv2_p_display = "-"
        
        md_lines.append(f"| {result['test']} | {result['segment']} | {result['dv1_control_pct']:.2f}% | "
                       f"{result['dv1_best_treatment']} | {result['dv1_best_pct']:.2f}% | "
                       f"{result['dv1_net_improvement']:.2f} pp | {dv1_p_display} | "
                       f"{dv2_control} | {dv2_best_treatment} | {dv2_best_pct} | "
                       f"{dv2_net_improvement} | {dv2_p_display} |")
    
    with open('summary_table_with_segments.md', 'w') as f:
        f.write("\n".join(md_lines))
    
    print("   Saved: summary_table_with_segments.md")
    
    # Save processed data
    print("\n8. Saving processed data...")
    df.to_csv('tgif_processed_data.csv', index=False)
    print("   Saved: tgif_processed_data.csv")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  1. tgif_ab_test_results.md - Detailed results")
    print("  2. complete_test_matrices.md - Two matrices per test")
    print("  3. summary_table_with_segments.csv - Summary CSV")
    print("  4. summary_table_with_segments.md - Summary markdown")
    print("  5. tgif_processed_data.csv - Data with binary columns")
    print("\n")


if __name__ == "__main__":
    main()