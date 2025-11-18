"""
TGIF A/B Test Analysis - Professional Report Generator

This script generates professional, easily modifiable reports for TGIF frozen snack A/B tests.

Requirements:
- pandas
- numpy
- scipy
- openpyxl (for Excel reading if needed)

Usage:
    python tgif_analysis_professional.py

Outputs:
    - tgif_detailed_results.csv (complete detailed results)
    - tgif_summary_table.csv (landscape summary table)
    - tgif_summary_table.md (markdown version of summary)
"""

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from datetime import datetime
import os

# ============================================================================
# CONFIGURATION - EASILY MODIFIABLE
# ============================================================================

DATA_FILE = 'tgif_firstround_raw.csv'
CODEBOOK_FILE = 'tgif_2025_codebook.xlsx'  # Optional, for reference

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

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def convert_to_binary(value):
    """Convert 1-5 scale to binary (0 for 1-3, 1 for 4-5)"""
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
    """Get significance marker for p-value"""
    if p_value < P_THRESHOLD_STRICT:
        return "✓ (p<0.05)"
    elif p_value < P_THRESHOLD_MODERATE:
        return "~ (p<0.10)"
    elif p_value < P_THRESHOLD_DIRECTIONAL:
        return "° (p<0.20)"
    else:
        return "✗ (p≥0.20)"

def calculate_chi_square(control_intent, control_n, treatment_intent, treatment_n):
    """Calculate chi-square test for independence"""
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

# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_data(df):
    """Prepare data with binary conversions and demographic variables"""
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
    """Apply demographic filter for a specific test"""
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
    """Analyze DV1 (purchase intent) for a test"""
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
        
        # Chi-square test
        chi2, p_value, expected = calculate_chi_square(
            control_intent, control_n, treatment_intent, treatment_n
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
            'chi2': chi2
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
    """Analyze DV2 (behavioral question) for a test"""
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
        
        # Chi-square test
        chi2, p_value, expected = calculate_chi_square(
            control_intent, control_n, treatment_intent, treatment_n
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
            'chi2': chi2
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

def generate_detailed_csv(all_results, output_file='tgif_detailed_results.csv'):
    """Generate detailed CSV report in the requested format
    
    Format: Test, Control_DV1_%, Treatment_B_DV1_%, Treatment_B_DV1_Net_Increase, 
            Treatment_B_DV1_P_Value, Treatment_C_DV1_%, ..., Control_DV2_%, 
            Treatment_B_DV2_%, Treatment_B_DV2_P_Value, ...
    """
    
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
    df_output.to_csv(output_file, index=False)
    print(f"   Saved: {output_file}")
    return df_output

def generate_summary_table(all_results, output_csv='tgif_summary_table.csv', 
                          output_md='tgif_summary_table.md'):
    """Generate landscape summary table"""
    
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
    df_summary.to_csv(output_csv, index=False)
    print(f"   Saved: {output_csv}")
    
    # Generate markdown version (using sorted DataFrame)
    md_lines = []
    md_lines.append("# TGIF A/B Test Summary Table\n")
    md_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    md_lines.append("| Test | Control DV1 % | Best Treatment DV1 | Best Treatment DV1 % | Best Treatment DV1 P-Value | Control DV2 % | Best Treatment DV2 | Best Treatment DV2 % | Best Treatment DV2 P-Value |")
    md_lines.append("|------|---------------|-------------------|---------------------|---------------------------|---------------|-------------------|---------------------|---------------------------|")
    
    for _, row in df_summary.iterrows():
        md_lines.append(f"| {row['Test']} | {row['Control_DV1_%']} | {row['Best_Treatment_DV1']} | {row['Best_Treatment_DV1_%']} | {row['Best_Treatment_DV1_P_Value']} | {row['Control_DV2_%']} | {row['Best_Treatment_DV2']} | {row['Best_Treatment_DV2_%']} | {row['Best_Treatment_DV2_P_Value']} |")
    
    with open(output_md, 'w') as f:
        f.write("\n".join(md_lines))
    print(f"   Saved: {output_md}")
    
    return df_summary

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    print("="*80)
    print("TGIF A/B TEST ANALYSIS - PROFESSIONAL REPORT GENERATOR")
    print("="*80)
    
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
    
    # Generate detailed CSV report
    print("\n4. Generating detailed CSV report...")
    generate_detailed_csv(all_results)
    
    # Generate summary table
    print("\n5. Generating summary table...")
    generate_summary_table(all_results)
    
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
        df_filtered.to_csv('tgif_filtered_results_summary.csv', index=False)
        print("   Saved: tgif_filtered_results_summary.csv")
        
        # Regenerate detailed CSV and summary with filtered results included
        print("\n7. Regenerating reports with filtered results included...")
        generate_detailed_csv(all_results, 'tgif_detailed_results_with_filters.csv')
        generate_summary_table(all_results, 'tgif_summary_table_with_filters.csv', 
                              'tgif_summary_table_with_filters.md')
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  1. tgif_detailed_results.csv - Complete detailed results")
    print("  2. tgif_summary_table.csv - Landscape summary table")
    print("  3. tgif_summary_table.md - Markdown version of summary")
    if filtered_results:
        print("  4. tgif_filtered_results_summary.csv - Filtered results for Tests 9, 12, 34")
    print("\n")

if __name__ == "__main__":
    main()

