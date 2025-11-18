"""
TGIF A/B Test Analysis - Professional Report Generator

This script generates professional, easily modifiable reports for TGIF frozen snack A/B tests.

Copyright 2025

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an AS IS BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Requirements:
- pandas
- numpy
- scipy
- openpyxl (for Excel reading if needed)

Usage:
    python tgif_analysis_professional.py

Outputs:
    All files are saved to separate directories based on statistical test method:
    - output_chi2/ (for chi-square test results)
    - output_ttest/ (for one-sided t-test results)
    
    Files in each directory (filenames include test method suffix):
    - tgif_detailed_results_chi2.csv or tgif_detailed_results_ttest.csv (complete detailed results)
    - tgif_summary_table_chi2.csv or tgif_summary_table_ttest.csv (landscape summary table)
    - tgif_summary_table_chi2.md or tgif_summary_table_ttest.md (markdown version of summary)
    - tgif_filtered_results_summary_chi2.csv or tgif_filtered_results_summary_ttest.csv (filtered results for Tests 9, 12, 34)
    - tgif_detailed_results_with_filters_chi2.csv or tgif_detailed_results_with_filters_ttest.csv (detailed results with filters)
    - tgif_summary_table_with_filters_chi2.csv or tgif_summary_table_with_filters_ttest.csv (summary table with filters)
    - tgif_summary_table_with_filters_chi2.md or tgif_summary_table_with_filters_ttest.md (summary markdown with filters)
"""

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, ttest_ind
from datetime import datetime
import os
import argparse

# ============================================================================
# CONFIGURATION - EASILY MODIFIABLE
# ============================================================================

# File paths (script is in code/ folder, go up one level for data and reference)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'data')
REFERENCE_DIR = os.path.join(BASE_DIR, 'reference')

DATA_FILE = os.path.join(DATA_DIR, 'tgif_firstround_raw.csv')
CODEBOOK_FILE = os.path.join(DATA_DIR, 'codebook_tgif.xlsx')  # Optional, for reference

# Test numbers to analyze (in ascending order)
TEST_NUMBERS = sorted([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 
                       19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34])

# Significance thresholds
P_THRESHOLD_STRICT = 0.05
P_THRESHOLD_MODERATE = 0.10
P_THRESHOLD_DIRECTIONAL = 0.20

# Demographic filters configuration
DEMOGRAPHIC_FILTERS = {
    9: {'age_older': True},   # Test 9: filter by older ages
    12: {'has_children': True},  # Test 12: filter by those with kids
    34: {'age_older': True}    # Test 34: filter by older ages
}

# Age threshold for "older ages" (in years)
AGE_THRESHOLD = 45

# Paths to mapping files
TEST_MAPPING_FILE = os.path.join(REFERENCE_DIR, 'test_mapping.csv')

# Statistical test method: 'chi2' for chi-square test, 'ttest' for one-sided t-test
# Can be overridden via command-line argument
STATISTICAL_TEST = 'chi2'  # Default: chi-square test

# Output directory configuration (relative to base directory)
OUTPUT_BASE_NAME = 'output'  # Base directory name for outputs

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_output_directory(test_method=None):
    """
    Get output directory path based on test method.
    
    Parameters:
    -----------
    test_method : str, optional
        Statistical test method ('chi2' or 'ttest'). Defaults to STATISTICAL_TEST.
    
    Returns:
    --------
    str
        Full path to the output directory (e.g., 'output_chi2' or 'output_ttest').
    """
    if test_method is None:
        test_method = STATISTICAL_TEST
    dir_name = f"{OUTPUT_BASE_NAME}_{test_method}"
    output_dir = os.path.join(BASE_DIR, dir_name)
    return output_dir

def ensure_output_directory(test_method=None):
    """
    Create output directory if it does not exist.
    
    Parameters:
    -----------
    test_method : str, optional
        Statistical test method (chi2 or ttest). Defaults to STATISTICAL_TEST.
    
    Returns:
    --------
    str
        Full path to the output directory.
    """
    output_dir = get_output_directory(test_method)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"   Created output directory: {output_dir}/")
    return output_dir

def add_test_suffix_to_filename(filename, test_method=None):
    """
    Add test method suffix to filename before extension.
    
    Parameters:
    -----------
    filename : str
        Original filename (e.g., 'results.csv').
    test_method : str, optional
        Statistical test method ('chi2' or 'ttest'). Defaults to STATISTICAL_TEST.
    
    Returns:
    --------
    str
        Filename with test method suffix (e.g., 'results_chi2.csv').
    """
    if test_method is None:
        test_method = STATISTICAL_TEST
    
    # Split filename into name and extension
    name, ext = os.path.splitext(filename)
    # Add test method suffix
    return f"{name}_{test_method}{ext}"

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_test_insights():
    """
    Load test insight abbreviations from test_mapping.csv.
    
    Returns:
    --------
    dict
        Dictionary mapping test numbers to insight abbreviations.
    """
    test_insights = {}
    
    try:
        mapping_df = pd.read_csv(TEST_MAPPING_FILE)
        for _, row in mapping_df.iterrows():
            test_key = row['Test']
            if pd.notna(test_key):
                test_key_str = str(test_key).strip()
                if test_key_str.startswith('Test '):
                    test_num_str = test_key_str.replace('Test ', '').strip()
                    try:
                        test_num = int(test_num_str)
                        insight = row['Insight']
                        if pd.notna(insight):
                            test_insights[test_num] = str(insight).strip()
                    except ValueError:
                        continue
    except Exception as e:
        print(f"Warning: Could not load {TEST_MAPPING_FILE}: {e}")
    
    return test_insights

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def convert_to_binary(value):
    """
    Convert 1-5 scale to binary (0 for 1-3, 1 for 4-5).
    
    Parameters:
    -----------
    value : numeric or str
        Value to convert (1-5 scale).
    
    Returns:
    --------
    int or float
        Binary value: 0 for scores 1-3, 1 for scores 4-5, np.nan for invalid values.
    """
    if pd.isna(value) or value == '' or value == ' ':
        return np.nan
    try:
        score = int(float(str(value).strip()))
        if score >= 4:
            return 1
        elif score >= 1:
            return 0
        else:
            return np.nan
    except (ValueError, TypeError):
        return np.nan

def get_significance_marker(p_value):
    """
    Get significance marker for p-value.
    
    Parameters:
    -----------
    p_value : float
        P-value to evaluate.
    
    Returns:
    --------
    str
        Significance marker: "✓ (p<0.05)", "~ (p<0.10)", "→ (p<0.20)", or "".
    """
    if p_value < P_THRESHOLD_STRICT:
        return "✓ (p<0.05)"
    elif p_value < P_THRESHOLD_MODERATE:
        return "~ (p<0.10)"
    elif p_value < P_THRESHOLD_DIRECTIONAL:
        return "° (p<0.20)"
    else:
        return "✗ (p≥0.20)"

def calculate_chi_square(control_intent, control_n, treatment_intent, treatment_n):
    """
    Calculate chi-square test for independence.
    
    Parameters:
    -----------
    control_intent : int
        Number of positive responses in control group.
    control_n : int
        Total number of responses in control group.
    treatment_intent : int
        Number of positive responses in treatment group.
    treatment_n : int
        Total number of responses in treatment group.
    
    Returns:
    --------
    chi2 : float or None
        Chi-square statistic.
    p_value : float or None
        P-value from chi-square test.
    expected : array or None
        Expected frequencies.
    """
    contingency_table = np.array([
        [control_intent, control_n - control_intent],
        [treatment_intent, treatment_n - treatment_intent]
    ])
    
    # Check if we have valid data
    if control_n == 0 or treatment_n == 0:
        return None, None, None
    
    # Check if expected frequencies are sufficient
    try:
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        return chi2, p_value, expected
    except:
        return None, None, None

def calculate_one_sided_ttest(control_data, treatment_data):
    """
    Calculate one-sided t-test for binary data (testing if treatment > control)
    
    Parameters:
    -----------
    control_data : array-like
        Binary data (0/1) for control group
    treatment_data : array-like
        Binary data (0/1) for treatment group
    
    Returns:
    --------
    t_stat : float or None
        t-statistic
    p_value : float or None
        One-sided p-value (testing treatment > control)
    """
    control_data = np.array(control_data)
    treatment_data = np.array(treatment_data)
    
    # Check if we have valid data
    if len(control_data) == 0 or len(treatment_data) == 0:
        return None, None
    
    # Remove any NaN values
    control_data = control_data[~np.isnan(control_data)]
    treatment_data = treatment_data[~np.isnan(treatment_data)]
    
    if len(control_data) == 0 or len(treatment_data) == 0:
        return None, None
    
    try:
        # Perform one-sided t-test (alternative='greater' tests if treatment mean > control mean)
        t_stat, p_value_two_sided = ttest_ind(treatment_data, control_data, equal_var=False)
        
        # Convert to one-sided p-value
        # If t_stat > 0, treatment mean > control mean, so p_one_sided = p_two_sided / 2
        # If t_stat <= 0, treatment mean <= control mean, so p_one_sided = 1 - p_two_sided / 2
        if t_stat > 0:
            p_value_one_sided = p_value_two_sided / 2
        else:
            p_value_one_sided = 1 - (p_value_two_sided / 2)
        
        return t_stat, p_value_one_sided
    except:
        return None, None

def calculate_statistical_test(control_data, treatment_data, test_method='chi2'):
    """
    Calculate statistical test based on method specified
    
    Parameters:
    -----------
    control_data : array-like
        Binary data (0/1) for control group
    treatment_data : array-like
        Binary data (0/1) for treatment group
    test_method : str
        'chi2' for chi-square test, 'ttest' for one-sided t-test
    
    Returns:
    --------
    test_stat : float or None
        Test statistic (chi2 or t-stat)
    p_value : float or None
        P-value
    """
    if test_method == 'ttest':
        t_stat, p_value = calculate_one_sided_ttest(control_data, treatment_data)
        return t_stat, p_value, None  # Return None for expected (not applicable for t-test)
    else:  # Default to chi-square
        control_intent = np.sum(control_data)
        control_n = len(control_data)
        treatment_intent = np.sum(treatment_data)
        treatment_n = len(treatment_data)
        chi2, p_value, expected = calculate_chi_square(
            control_intent, control_n, treatment_intent, treatment_n
        )
        return chi2, p_value, expected

# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_data(df):
    """
    Prepare data with binary conversions and demographic variables.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Raw data frame with test responses.
    
    Returns:
    --------
    pandas.DataFrame
        Data frame with binary conversions and demographic variables added.
    """
    df = df.copy()
    
    # Convert purchase intent to binary for all tests
    print("Converting purchase intent scores to binary...")
    for test_num in TEST_NUMBERS:
        dv1_col = f'dv{test_num}_1'
        dv2_col = f'dv{test_num}_2'
        
        if dv1_col in df.columns:
            df[f'dv{test_num}_1_binary'] = df[dv1_col].apply(convert_to_binary)
        if dv2_col in df.columns:
            df[f'dv{test_num}_2_binary'] = df[dv2_col].apply(convert_to_binary)
    
    # Create demographic variables
    print("Creating demographic variables...")
    if 'q18' in df.columns:
        df['age'] = 2025 - pd.to_numeric(df['q18'], errors='coerce')
        df['age_older'] = df['age'] >= AGE_THRESHOLD
    
    if 'q20' in df.columns:
        # Assuming 1 = has children, 2 = no children (adjust if different)
        df['has_children'] = pd.to_numeric(df['q20'], errors='coerce') == 1
    
    return df

def apply_demographic_filter(df, test_num):
    """
    Apply demographic filter for a specific test.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Data frame to filter.
    test_num : int
        Test number to apply filter for.
    
    Returns:
    --------
    pandas.DataFrame
        Filtered data frame based on test-specific demographic criteria.
    """
    if test_num not in DEMOGRAPHIC_FILTERS:
        return df
    
    filters = DEMOGRAPHIC_FILTERS[test_num]
    filtered_df = df.copy()
    
    if 'age_older' in filters and filters['age_older']:
        filtered_df = filtered_df[filtered_df['age_older'] == True]
    
    if 'has_children' in filters and filters['has_children']:
        filtered_df = filtered_df[filtered_df['has_children'] == True]
    
    return filtered_df

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_test_dv1(test_num, df):
    """
    Analyze DV1 (purchase intent) for a test.
    
    Parameters:
    -----------
    test_num : int
        Test number to analyze.
    df : pandas.DataFrame
        Data frame with prepared data.
    
    Returns:
    --------
    dict or None
        Dictionary with control and treatment results, or None if test cannot be analyzed.
    """
    test_assignment_col = f'test{test_num}'
    dv1_binary_col = f'dv{test_num}_1_binary'
    
    # Filter to respondents who took this test
    test_df = df[df[test_assignment_col].notna() & 
                (df[test_assignment_col] != '') & 
                (df[test_assignment_col] != ' ')].copy()
    
    if len(test_df) == 0 or dv1_binary_col not in test_df.columns:
        return None
    
    # Extract treatment letter
    test_df['treatment'] = test_df[test_assignment_col].astype(str).str.extract(r'([A-Z])', expand=False)
    
    # Control (A) data
    control_data = test_df[test_df['treatment'] == 'A'][dv1_binary_col].dropna()
    
    if len(control_data) == 0:
        return None
    
    control_n = len(control_data)
    control_intent = control_data.sum()
    control_pct = (control_intent / control_n) * 100 if control_n > 0 else 0
    
    # Treatment data
    treatments = sorted([t for t in test_df['treatment'].unique() 
                        if t and t != 'A' and pd.notna(t)])
    
    treatment_results = []
    
    for treatment_letter in treatments:
        treatment_data = test_df[test_df['treatment'] == treatment_letter][dv1_binary_col].dropna()
        
        if len(treatment_data) == 0:
            continue
        
        treatment_n = len(treatment_data)
        treatment_intent = treatment_data.sum()
        treatment_pct = (treatment_intent / treatment_n) * 100 if treatment_n > 0 else 0
        
        # Calculate lift
        net_increase = treatment_pct - control_pct
        lift_pct = ((treatment_pct - control_pct) / control_pct * 100) if control_pct > 0 else 0
        
        # Statistical test (chi-square or one-sided t-test)
        test_stat, p_value, expected = calculate_statistical_test(
            control_data.values, treatment_data.values, test_method=STATISTICAL_TEST
        )
        
        if p_value is None:
            continue
        
        treatment_results.append({
            'treatment': treatment_letter,
            'n': treatment_n,
            'intent': treatment_intent,
            'pct': treatment_pct,
            'net_increase': net_increase,
            'lift_pct': lift_pct,
            'p_value': p_value,
            'test_stat': test_stat,
            'test_method': STATISTICAL_TEST
        })
    
    # Sort by lift (descending)
    treatment_results = sorted(treatment_results, key=lambda x: x['net_increase'], reverse=True)
    
    return {
        'test_number': test_num,
        'control_n': control_n,
        'control_intent': control_intent,
        'control_pct': control_pct,
        'treatments': treatment_results
    }

def analyze_test_dv2(test_num, df):
    """
    Analyze DV2 (behavioral question) for a test.
    
    Parameters:
    -----------
    test_num : int
        Test number to analyze.
    df : pandas.DataFrame
        Data frame with prepared data.
    
    Returns:
    --------
    dict or None
        Dictionary with control and treatment results, or None if test cannot be analyzed.
    """
    test_assignment_col = f'test{test_num}'
    dv2_binary_col = f'dv{test_num}_2_binary'
    
    # Filter to respondents who took this test
    test_df = df[df[test_assignment_col].notna() & 
                (df[test_assignment_col] != '') & 
                (df[test_assignment_col] != ' ')].copy()
    
    if len(test_df) == 0 or dv2_binary_col not in test_df.columns:
        return None
    
    # Extract treatment letter
    test_df['treatment'] = test_df[test_assignment_col].astype(str).str.extract(r'([A-Z])', expand=False)
    
    # Control (A) data
    control_data = test_df[test_df['treatment'] == 'A'][dv2_binary_col].dropna()
    
    if len(control_data) == 0:
        return None
    
    control_n = len(control_data)
    control_intent = control_data.sum()
    control_pct = (control_intent / control_n) * 100 if control_n > 0 else 0
    
    # Treatment data
    treatments = sorted([t for t in test_df['treatment'].unique() 
                        if t and t != 'A' and pd.notna(t)])
    
    treatment_results = []
    
    for treatment_letter in treatments:
        treatment_data = test_df[test_df['treatment'] == treatment_letter][dv2_binary_col].dropna()
        
        if len(treatment_data) == 0:
            continue
        
        treatment_n = len(treatment_data)
        treatment_intent = treatment_data.sum()
        treatment_pct = (treatment_intent / treatment_n) * 100 if treatment_n > 0 else 0
        
        # Calculate lift
        net_increase = treatment_pct - control_pct
        
        # Statistical test (chi-square or one-sided t-test)
        test_stat, p_value, expected = calculate_statistical_test(
            control_data.values, treatment_data.values, test_method=STATISTICAL_TEST
        )
        
        if p_value is None:
            continue
        
        treatment_results.append({
            'treatment': treatment_letter,
            'n': treatment_n,
            'intent': treatment_intent,
            'pct': treatment_pct,
            'net_increase': net_increase,
            'p_value': p_value,
            'test_stat': test_stat,
            'test_method': STATISTICAL_TEST
        })
    
    # Sort by lift (descending)
    treatment_results = sorted(treatment_results, key=lambda x: x['net_increase'], reverse=True)
    
    return {
        'test_number': test_num,
        'control_n': control_n,
        'control_intent': control_intent,
        'control_pct': control_pct,
        'treatments': treatment_results
    }

# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_detailed_csv(all_results, output_file=None, output_dir=None, test_method=None):
    """
    Generate detailed CSV report in the requested format.
    
    Format: Test, Control_DV1_%, Treatment_B_DV1_%, Treatment_B_DV1_Net_Increase, 
            Treatment_B_DV1_P_Value, Treatment_C_DV1_%, ..., Control_DV2_%, 
            Treatment_B_DV2_%, Treatment_B_DV2_P_Value, ...
    
    Parameters:
    -----------
    all_results : list
        List of analysis result dictionaries.
    output_file : str, optional
        Output filename. Defaults to 'tgif_detailed_results.csv'.
    output_dir : str, optional
        Output directory path.
    test_method : str, optional
        Statistical test method ('chi2' or 'ttest').
    
    Returns:
    --------
    pandas.DataFrame
        Generated detailed results data frame.
    """
    
    # Add note about statistical test method
    test_method_name = "One-sided t-test" if STATISTICAL_TEST == 'ttest' else "Chi-square test"
    
    # First, determine all possible treatment letters across all tests
    all_treatment_letters = set()
    for result in all_results:
        for treatment in result['dv1']['treatments']:
            all_treatment_letters.add(treatment['treatment'])
    all_treatment_letters = sorted(list(all_treatment_letters))
    
    rows = []
    
    for result in all_results:
        test_num = result['test_number']
        # Handle filtered test numbers
        if isinstance(test_num, str) and '_filtered' in test_num:
            test_display = test_num
            test_num_for_sort = int(test_num.split('_')[0])
        else:
            test_display = test_num
            test_num_for_sort = test_num
        
        dv1_result = result['dv1']
        dv2_result = result.get('dv2')
        
        # Build row
        row = {'Test': test_display, '_sort_key': test_num_for_sort}
        
        # DV1 Control
        row['Control_DV1_%'] = f"{dv1_result['control_pct']:.2f}"
        
        # DV1 Treatments (ordered by lift, but columns in alphabetical order)
        # Create a dictionary for easy lookup
        dv1_treatments_dict = {t['treatment']: t for t in dv1_result['treatments']}
        
        for letter in all_treatment_letters:
            if letter in dv1_treatments_dict:
                treatment = dv1_treatments_dict[letter]
                row[f'Treatment_{letter}_DV1_%'] = f"{treatment['pct']:.2f}"
                row[f'Treatment_{letter}_DV1_Net_Increase'] = f"{treatment['net_increase']:.2f}"
                row[f'Treatment_{letter}_DV1_P_Value'] = f"{treatment['p_value']:.4f}"
            else:
                # Test doesn't have this treatment
                row[f'Treatment_{letter}_DV1_%'] = ""
                row[f'Treatment_{letter}_DV1_Net_Increase'] = ""
                row[f'Treatment_{letter}_DV1_P_Value'] = ""
        
        # DV2 Control
        if dv2_result:
            row['Control_DV2_%'] = f"{dv2_result['control_pct']:.2f}"
            
            # DV2 Treatments (ordered by lift, but columns in alphabetical order)
            dv2_treatments_dict = {t['treatment']: t for t in dv2_result['treatments']}
            
            for letter in all_treatment_letters:
                if letter in dv2_treatments_dict:
                    treatment = dv2_treatments_dict[letter]
                    row[f'Treatment_{letter}_DV2_%'] = f"{treatment['pct']:.2f}"
                    row[f'Treatment_{letter}_DV2_P_Value'] = f"{treatment['p_value']:.4f}"
                else:
                    row[f'Treatment_{letter}_DV2_%'] = ""
                    row[f'Treatment_{letter}_DV2_P_Value'] = ""
        else:
            row['Control_DV2_%'] = ""
            for letter in all_treatment_letters:
                row[f'Treatment_{letter}_DV2_%'] = ""
                row[f'Treatment_{letter}_DV2_P_Value'] = ""
        
        rows.append(row)
    
    # Create DataFrame with consistent column order
    # Add statistical test method as first data row (after header)
    columns = ['Test', 'Control_DV1_%']
    for letter in all_treatment_letters:
        columns.extend([f'Treatment_{letter}_DV1_%', 
                       f'Treatment_{letter}_DV1_Net_Increase', 
                       f'Treatment_{letter}_DV1_P_Value'])
    columns.append('Control_DV2_%')
    for letter in all_treatment_letters:
        columns.extend([f'Treatment_{letter}_DV2_%', 
                       f'Treatment_{letter}_DV2_P_Value'])
    
    df_output = pd.DataFrame(rows)
    # Sort by test number
    df_output = df_output.sort_values('_sort_key')
    # Remove sort key before saving
    df_output = df_output.drop('_sort_key', axis=1)
    # Reorder columns
    existing_columns = [c for c in columns if c in df_output.columns]
    df_output = df_output[existing_columns]
    
    # Set default filename if not provided
    if output_file is None:
        output_file = "tgif_detailed_results.csv"
    
    # Add test method suffix to filename
    if test_method:
        output_file = add_test_suffix_to_filename(output_file, test_method)
    
    # Save CSV file (standard format, no comments)
    if output_dir:
        output_file = os.path.join(output_dir, output_file)
    df_output.to_csv(output_file, index=False)
    
    print(f"   Saved: {output_file} (using {test_method_name})")
    return df_output

def generate_summary_table(all_results, output_csv=None, 
                          output_md=None, test_insights=None, output_dir=None, test_method=None):
    """
    Generate landscape summary table.
    
    Parameters:
    -----------
    all_results : list
        List of analysis result dictionaries.
    output_csv : str, optional
        Output CSV filename. Defaults to 'tgif_summary_table.csv'.
    output_md : str, optional
        Output markdown filename. Defaults to 'tgif_summary_table.md'.
    test_insights : dict, optional
        Dictionary mapping test numbers to insight abbreviations.
    output_dir : str, optional
        Output directory path.
    test_method : str, optional
        Statistical test method ('chi2' or 'ttest').
    
    Returns:
    --------
    pandas.DataFrame
        Generated summary table data frame.
    """
    if test_method is None:
        test_method = STATISTICAL_TEST
    if output_csv is None:
        output_csv = add_test_suffix_to_filename('tgif_summary_table.csv', test_method)
    if output_md is None:
        output_md = add_test_suffix_to_filename('tgif_summary_table.md', test_method)
    
    if test_insights is None:
        test_insights = load_test_insights()
    
    rows = []
    
    for result in all_results:
        test_num = result['test_number']
        # Handle filtered test numbers
        if isinstance(test_num, str) and '_filtered' in test_num:
            test_display = test_num
            test_num_for_sort = int(test_num.split('_')[0])
            base_test_num = test_num_for_sort
        else:
            test_display = test_num
            test_num_for_sort = test_num
            base_test_num = test_num
        
        # Get insight abbreviation
        insight_abbr = test_insights.get(base_test_num, "")
        
        dv1_result = result['dv1']
        dv2_result = result.get('dv2')
        
        # Find best treatment for DV1
        if dv1_result['treatments']:
            best_dv1 = dv1_result['treatments'][0]
            best_dv1_letter = best_dv1['treatment']
            best_dv1_pct = best_dv1['pct']
            best_dv1_p = best_dv1['p_value']
        else:
            best_dv1_letter = "N/A"
            best_dv1_pct = 0
            best_dv1_p = 1.0
        
        # Find best treatment for DV2
        if dv2_result and dv2_result['treatments']:
            best_dv2 = dv2_result['treatments'][0]
            best_dv2_letter = best_dv2['treatment']
            best_dv2_pct = best_dv2['pct']
            best_dv2_p = best_dv2['p_value']
            dv2_control_pct = dv2_result['control_pct']
        else:
            best_dv2_letter = "N/A"
            best_dv2_pct = 0
            best_dv2_p = 1.0
            dv2_control_pct = 0
        
        row = {
            'Test': test_display,
            'Insight': insight_abbr,
            '_sort_key': test_num_for_sort,
            'Control_DV1_%': f"{dv1_result['control_pct']:.2f}%",
            'Best_Treatment_DV1': best_dv1_letter,
            'Best_Treatment_DV1_%': f"{best_dv1_pct:.2f}%",
            'Best_Treatment_DV1_P_Value': f"{best_dv1_p:.4f}",
            'Control_DV2_%': f"{dv2_control_pct:.2f}%" if dv2_result else "N/A",
            'Best_Treatment_DV2': best_dv2_letter,
            'Best_Treatment_DV2_%': f"{best_dv2_pct:.2f}%" if dv2_result else "N/A",
            'Best_Treatment_DV2_P_Value': f"{best_dv2_p:.4f}" if dv2_result else "N/A"
        }
        
        rows.append(row)
    
    # Create DataFrame
    df_summary = pd.DataFrame(rows)
    # Sort by test number
    df_summary = df_summary.sort_values('_sort_key')
    # Remove sort key before saving
    df_summary = df_summary.drop('_sort_key', axis=1)
    
    # Save CSV
    if output_dir:
        output_csv = os.path.join(output_dir, output_csv)
    df_summary.to_csv(output_csv, index=False)
    print(f"   Saved: {output_csv}")
    
    # Generate markdown version (using sorted DataFrame)
    md_lines = []
    md_lines.append("# TGIF A/B Test Summary Table\n")
    md_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    test_method_name = "One-sided t-test" if test_method == 'ttest' else "Chi-square test"
    md_lines.append(f"**Statistical Test Method:** {test_method_name}\n")
    if test_method == 'ttest':
        md_lines.append("(One-sided t-test: testing if treatment proportion > control proportion)\n")
    else:
        md_lines.append("(Chi-square test: testing for independence between treatment and outcome)\n")
    md_lines.append("| Test | Insight | Control DV1 % | Best Treatment DV1 | Best Treatment DV1 % | Best Treatment DV1 P-Value | Control DV2 % | Best Treatment DV2 | Best Treatment DV2 % | Best Treatment DV2 P-Value |")
    md_lines.append("|------|---------|---------------|-------------------|---------------------|---------------------------|---------------|-------------------|---------------------|---------------------------|")
    
    for _, row in df_summary.iterrows():
        md_lines.append(f"| {row['Test']} | {row['Insight']} | {row['Control_DV1_%']} | {row['Best_Treatment_DV1']} | {row['Best_Treatment_DV1_%']} | {row['Best_Treatment_DV1_P_Value']} | {row['Control_DV2_%']} | {row['Best_Treatment_DV2']} | {row['Best_Treatment_DV2_%']} | {row['Best_Treatment_DV2_P_Value']} |")
    
    # Save markdown
    if output_dir:
        output_md = os.path.join(output_dir, output_md)
    with open(output_md, 'w') as f:
        f.write("\n".join(md_lines))
    print(f"   Saved: {output_md}")
    
    return df_summary

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
    --------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description='TGIF A/B Test Analysis - Professional Report Generator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tgif_analysis_professional.py                    # Use chi-square (default)
  python tgif_analysis_professional.py --test-method chi2  # Use chi-square
  python tgif_analysis_professional.py --test-method ttest # Use one-sided t-test
  python tgif_analysis_professional.py -t ttest           # Short form
        """
    )
    parser.add_argument(
        '--test-method', '-t',
        type=str,
        choices=['chi2', 'ttest'],
        default=STATISTICAL_TEST,
        help='Statistical test method: "chi2" for chi-square test (default), "ttest" for one-sided t-test'
    )
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_arguments()
    
    # Use command-line argument if provided, otherwise use default from config
    global STATISTICAL_TEST
    STATISTICAL_TEST = args.test_method
    
    print("="*80)
    print("TGIF A/B TEST ANALYSIS - PROFESSIONAL REPORT GENERATOR")
    print("="*80)
    
    # Display statistical test method
    test_method_name = "One-sided t-test" if STATISTICAL_TEST == 'ttest' else "Chi-square test"
    print(f"\nStatistical Test Method: {test_method_name}")
    if STATISTICAL_TEST == 'ttest':
        print("  (Testing if treatment proportion > control proportion)")
    else:
        print("  (Testing for independence between treatment and outcome)")
    
    # Load data
    print("\n1. Loading data...")
    if not os.path.exists(DATA_FILE):
        print(f"   ERROR: Data file '{DATA_FILE}' not found!")
        return
    
    df = pd.read_csv(DATA_FILE)
    print(f"   Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Prepare data
    print("\n2. Preparing data...")
    df = prepare_data(df)
    
    # Run analysis for all tests
    print("\n3. Running analysis for all tests...")
    all_results = []
    
    for test_num in TEST_NUMBERS:
        print(f"   Analyzing Test {test_num}...")
        
        # Check if test exists in data
        test_col = f'test{test_num}'
        if test_col not in df.columns:
            print(f"      Warning: {test_col} not found in data, skipping...")
            continue
        
        # Use full dataset for main analysis (no filters)
        test_df = df.copy()
        filter_applied = False
        
        # Analyze DV1
        dv1_result = analyze_test_dv1(test_num, test_df)
        if dv1_result is None:
            print(f"      Warning: No valid data for Test {test_num}, skipping...")
            continue
        
        # Check if any treatment is significant
        has_significant = any(t['p_value'] < P_THRESHOLD_STRICT 
                             for t in dv1_result['treatments'])
        
        # Analyze DV2 if no significant results
        dv2_result = None
        if not has_significant:
            dv2_result = analyze_test_dv2(test_num, test_df)
        
        result = {
            'test_number': test_num,
            'filter_applied': filter_applied,
            'dv1': dv1_result,
            'dv2': dv2_result
        }
        
        all_results.append(result)
    
    print(f"\n   Analyzed {len(all_results)} tests")
    
    # Create output directory for this test method
    print("\n3a. Setting up output directory...")
    output_dir = ensure_output_directory(STATISTICAL_TEST)
    
    # Load test insights once
    print("\n3b. Loading test insights...")
    test_insights = load_test_insights()
    print(f"   Loaded {len(test_insights)} test insights")
    
    # Generate detailed CSV report
    print("\n4. Generating detailed CSV report...")
    generate_detailed_csv(all_results, output_dir=output_dir, test_method=STATISTICAL_TEST)
    
    # Generate summary table
    print("\n5. Generating summary table...")
    generate_summary_table(all_results, test_insights=test_insights, output_dir=output_dir, test_method=STATISTICAL_TEST)
    
    # Generate filtered results for specific tests and append to main results
    print("\n6. Generating filtered results for Tests 9, 12, and 34...")
    filtered_results = []
    
    for test_num in [9, 12, 34]:
        if test_num not in DEMOGRAPHIC_FILTERS:
            continue
        
        print(f"   Generating filtered results for Test {test_num}...")
        test_df = apply_demographic_filter(df, test_num)
        
        # Analyze DV1
        dv1_result = analyze_test_dv1(test_num, test_df)
        if dv1_result is None:
            continue
        
        # Analyze DV2
        dv2_result = analyze_test_dv2(test_num, test_df)
        
        # Determine filter description
        filter_desc = ""
        if test_num == 9 or test_num == 34:
            filter_desc = "Older Ages (45+)"
        elif test_num == 12:
            filter_desc = "Has Children"
        
        result = {
            'test_number': test_num,
            'filter_applied': True,
            'filter_description': filter_desc,
            'dv1': dv1_result,
            'dv2': dv2_result
        }
        
        filtered_results.append(result)
        # Also add to all_results with a note
        result_with_filter_note = result.copy()
        result_with_filter_note['test_number'] = f"{test_num}_filtered"
        all_results.append(result_with_filter_note)
    
    if filtered_results:
        # Generate filtered summary
        filter_summary_rows = []
        for result in filtered_results:
            test_num = result['test_number']
            # Get insight abbreviation
            insight_abbr = test_insights.get(test_num, "")
            
            dv1_result = result['dv1']
            dv2_result = result.get('dv2')
            
            if dv1_result['treatments']:
                best_dv1 = dv1_result['treatments'][0]
            else:
                continue
            
            if dv2_result and dv2_result['treatments']:
                best_dv2 = dv2_result['treatments'][0]
            else:
                best_dv2 = None
            
            row = {
                'Test': f"{test_num} (Filtered: {result.get('filter_description', '')})",
                'Insight': insight_abbr,
                'Control_DV1_%': f"{dv1_result['control_pct']:.2f}%",
                'Best_Treatment_DV1': best_dv1['treatment'],
                'Best_Treatment_DV1_%': f"{best_dv1['pct']:.2f}%",
                'Best_Treatment_DV1_P_Value': f"{best_dv1['p_value']:.4f}",
                'Control_DV2_%': f"{dv2_result['control_pct']:.2f}%" if dv2_result else "N/A",
                'Best_Treatment_DV2': best_dv2['treatment'] if best_dv2 else "N/A",
                'Best_Treatment_DV2_%': f"{best_dv2['pct']:.2f}%" if best_dv2 else "N/A",
                'Best_Treatment_DV2_P_Value': f"{best_dv2['p_value']:.4f}" if best_dv2 else "N/A"
            }
            filter_summary_rows.append(row)
        
        df_filtered = pd.DataFrame(filter_summary_rows)
        filtered_summary_filename = add_test_suffix_to_filename('tgif_filtered_results_summary.csv', STATISTICAL_TEST)
        filtered_summary_file = os.path.join(output_dir, filtered_summary_filename)
        df_filtered.to_csv(filtered_summary_file, index=False)
        print(f"   Saved: {filtered_summary_file}")
        
        # Regenerate detailed CSV and summary with filtered results included
        print("\n7. Regenerating reports with filtered results included...")
        filtered_detailed_filename = add_test_suffix_to_filename('tgif_detailed_results_with_filters.csv', STATISTICAL_TEST)
        filtered_summary_csv_filename = add_test_suffix_to_filename('tgif_summary_table_with_filters.csv', STATISTICAL_TEST)
        filtered_summary_md_filename = add_test_suffix_to_filename('tgif_summary_table_with_filters.md', STATISTICAL_TEST)
        generate_detailed_csv(all_results, filtered_detailed_filename, output_dir=output_dir, test_method=STATISTICAL_TEST)
        generate_summary_table(all_results, filtered_summary_csv_filename, 
                              filtered_summary_md_filename, test_insights=test_insights, output_dir=output_dir, test_method=STATISTICAL_TEST)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nAll output files saved to: {output_dir}/")
    print("\nGenerated files:")
    detailed_filename = add_test_suffix_to_filename('tgif_detailed_results.csv', STATISTICAL_TEST)
    summary_csv_filename = add_test_suffix_to_filename('tgif_summary_table.csv', STATISTICAL_TEST)
    summary_md_filename = add_test_suffix_to_filename('tgif_summary_table.md', STATISTICAL_TEST)
    print(f"  1. {output_dir}/{detailed_filename} - Complete detailed results")
    print(f"  2. {output_dir}/{summary_csv_filename} - Landscape summary table")
    print(f"  3. {output_dir}/{summary_md_filename} - Markdown version of summary")
    if filtered_results:
        filtered_summary_filename = add_test_suffix_to_filename('tgif_filtered_results_summary.csv', STATISTICAL_TEST)
        filtered_detailed_filename = add_test_suffix_to_filename('tgif_detailed_results_with_filters.csv', STATISTICAL_TEST)
        filtered_summary_csv_filename = add_test_suffix_to_filename('tgif_summary_table_with_filters.csv', STATISTICAL_TEST)
        filtered_summary_md_filename = add_test_suffix_to_filename('tgif_summary_table_with_filters.md', STATISTICAL_TEST)
        print(f"  4. {output_dir}/{filtered_summary_filename} - Filtered results for Tests 9, 12, 34")
        print(f"  5. {output_dir}/{filtered_detailed_filename} - Detailed results with filters")
        print(f"  6. {output_dir}/{filtered_summary_csv_filename} - Summary table with filters")
        print(f"  7. {output_dir}/{filtered_summary_md_filename} - Summary markdown with filters")
    print("\n")

if __name__ == "__main__":
    main()

