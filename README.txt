================================================================================
TGIF A/B TEST ANALYSIS - CODEBASE DOCUMENTATION
================================================================================

This document provides a comprehensive guide to understanding and using the
TGIF A/B test analysis codebase. It is designed for both technical and
non-technical users who need to understand, run, or modify the analysis.

================================================================================
PROJECT OVERVIEW
================================================================================

This project analyzes A/B test data for TGIF frozen snack products. It performs
statistical analysis on purchase intent and behavioral questions across 34
different A/B tests, comparing control groups against various treatment
conditions.

KEY FEATURES:
- Statistical significance testing (chi-square or one-sided t-test)
- Automatic binary conversion of 1-5 scale responses
- Lift calculation and treatment ranking
- Demographic filtering for specific tests
- Professional CSV and Markdown report generation

================================================================================
FOLDER STRUCTURE
================================================================================

data/
  Contains the raw data files:
  - tgif_firstround_raw.csv          Main survey data file
  - codebook_tgif.xlsx                Excel codebook (reference)

code/
  Contains the Python analysis script:
  - tgif_analysis_professional.py    Main analysis script

reference/
  Contains all coding/reference files that explain numerical mappings:
  - test_mapping.csv                  Maps test numbers to short insight names
  - insight_pairing.csv               Maps test numbers to full insight descriptions
  - children_coding.csv               Explains children status codes (1=Yes, 2=No, etc.)
  - frequency_coding.csv              Explains frequency codes (1=Once/week, etc.)
  - occasions_coding.csv              Explains occasion codes (1=Date night, etc.)
  - restaurant_coding.csv             Explains restaurant visit frequency codes
  - statement_occasion_coding.csv     Explains statement occasion codes
  - age_coding.txt                    Notes about age data format

docs/
  Contains documentation files:
  - README_ANALYSIS.md                Technical documentation for developers
  - tgif_ab_test_results.md           Previous analysis results (if any)

output_chi2/
  Generated output folder for chi-square test results:
  - tgif_detailed_results_chi2.csv
  - tgif_summary_table_chi2.csv
  - tgif_summary_table_chi2.md
  - tgif_filtered_results_summary_chi2.csv
  - tgif_detailed_results_with_filters_chi2.csv
  - tgif_summary_table_with_filters_chi2.csv
  - tgif_summary_table_with_filters_chi2.md

output_ttest/
  Generated output folder for one-sided t-test results:
  - tgif_detailed_results_ttest.csv
  - tgif_summary_table_ttest.csv
  - tgif_summary_table_ttest.md
  - tgif_filtered_results_summary_ttest.csv
  - tgif_detailed_results_with_filters_ttest.csv
  - tgif_summary_table_with_filters_ttest.csv
  - tgif_summary_table_with_filters_ttest.md

================================================================================
QUICK START GUIDE
================================================================================

1. PREREQUISITES
   - Python 3.7 or higher
   - Required packages: pandas, numpy, scipy
   - Install with: pip install pandas numpy scipy

2. RUNNING THE ANALYSIS
   
   Default (chi-square test):
   > cd code
   > python tgif_analysis_professional.py
   
   With one-sided t-test:
   > cd code
   > python tgif_analysis_professional.py --test-method ttest
   > python tgif_analysis_professional.py -t ttest
   
   View help:
   > cd code
   > python tgif_analysis_professional.py --help
   
   NOTE: Always run the script from the 'code' folder. The script will
   automatically find data and reference files in their respective folders.

3. FINDING RESULTS
   - Results are saved in output_chi2/ or output_ttest/ folders
   - Filenames include the test method suffix (_chi2 or _ttest)
   - Check the console output for exact file paths

================================================================================
UNDERSTANDING THE DATA
================================================================================

RAW DATA FILE (data/tgif_firstround_raw.csv):
  - Each row represents one survey respondent
  - Columns include:
    * test1, test2, ..., test34: Treatment assignments (A=control, B-F=treatments)
    * dv1_1, dv1_2, ..., dv34_1, dv34_2: Response scores (1-5 scale)
    * q18: Birth year (used to calculate age)
    * q20: Children status (1=has children, 2=no children)

CODING FILES (reference/ folder):
  These files explain what the numbers mean in the data:
  
  - test_mapping.csv: What each test number represents
    Example: Test 1 = "happy hour at home"
  
  - insight_pairing.csv: Full descriptions of what each test is testing
    Example: Test 1 = "Happy Hour Date Nights at Home"
  
  - children_coding.csv: Children status codes
    1 = Yes, 2 = No, 99 = Prefer not to say
  
  - frequency_coding.csv: Frequency codes
    1 = Once per week or more
    2 = About 2-3 times per month
    ... (see file for full list)
  
  - occasions_coding.csv: Occasion codes
    1 = Date night in with spouse or significant other
    2 = Family movie night
    ... (see file for full list)
  
  - restaurant_coding.csv: Restaurant visit frequency
    1 = Within the past week
    2 = 1-2 weeks ago
    ... (see file for full list)
  
  - statement_occasion_coding.csv: Statement occasion codes
    Similar to occasions_coding.csv but for different question context
  
  - age_coding.txt: Notes about age data (open-ended text format)

================================================================================
UNDERSTANDING THE OUTPUT FILES
================================================================================

DETAILED RESULTS (tgif_detailed_results_*.csv):
  - One row per test
  - Shows control percentage and all treatment percentages
  - Includes net increase and p-values for each treatment
  - Treatments ordered by performance (best to worst)

SUMMARY TABLE (tgif_summary_table_*.csv and *.md):
  - One row per test
  - Shows only the BEST performing treatment for each test
  - Includes test description/insight for easy interpretation
  - Markdown version is formatted for easy reading

FILTERED RESULTS:
  - Separate analysis for Tests 9, 12, and 34 with demographic filters
  - Test 9 & 34: Filtered to respondents age 45+
  - Test 12: Filtered to respondents with children

COLUMN EXPLANATIONS:
  - Test: Test number (1-34)
  - Insight: Short description of what the test is testing
  - Control_DV1_%: Control group purchase intent percentage
  - Best_Treatment_DV1: Letter of best performing treatment (B, C, D, E, or F)
  - Best_Treatment_DV1_%: Purchase intent percentage for best treatment
  - Best_Treatment_DV1_P_Value: Statistical significance (p-value)
  - DV2 columns: Same structure but for behavioral question (only shown if DV1
    had no significant results)

STATISTICAL TEST METHODS:
  - Chi-square test (default): Tests for independence between groups
  - One-sided t-test: Tests if treatment > control
  - Results are saved in separate folders based on test method
  - Filenames include suffix indicating which test was used

================================================================================
INTERPRETING RESULTS
================================================================================

P-VALUES:
  - p < 0.05: Statistically significant (marked with ✓)
  - p < 0.10: Marginally significant (marked with ~)
  - p < 0.20: Directionally interesting (marked with °)
  - p ≥ 0.20: Not significant (marked with ✗)

LIFT:
  - Net Increase: Absolute difference (Treatment % - Control %)
  - Lift %: Percentage change ((Treatment % - Control %) / Control % × 100)
  - Positive lift means treatment performed better than control

TREATMENT LETTERS:
  - A: Control group (baseline)
  - B, C, D, E, F: Different treatment conditions
  - Each letter represents a different message/insight being tested

TEST DESCRIPTIONS:
  - Check reference/test_mapping.csv for short insight names
  - Check reference/insight_pairing.csv for full descriptions
  - Example: Test 31 = "function as a meal due to protein"

================================================================================
MODIFYING THE ANALYSIS
================================================================================

To modify the analysis, edit code/tgif_analysis_professional.py:

CONFIGURATION SECTION (near top of file):
  - TEST_NUMBERS: Which tests to analyze
  - P_THRESHOLD_STRICT: Significance threshold (default 0.05)
  - DEMOGRAPHIC_FILTERS: Which tests to filter and how
  - AGE_THRESHOLD: Age cutoff for "older" segment (default 45)
  - STATISTICAL_TEST: Default test method ('chi2' or 'ttest')

DATA FILE PATHS:
  - Script automatically finds files in data/ and reference/ folders
  - No need to modify paths if folder structure is maintained

OUTPUT LOCATIONS:
  - Outputs go to: output_chi2/ or output_ttest/ folders
  - Filenames include test method suffix

================================================================================
TROUBLESHOOTING
================================================================================

COMMON ISSUES:

1. "File not found" errors:
   - Make sure you're running the script from the 'code' folder
   - Check that data/tgif_firstround_raw.csv exists
   - Verify folder structure matches this README

2. "Module not found" errors:
   - Install required packages: pip install pandas numpy scipy

3. Output files not appearing:
   - Check console output for exact file paths
   - Look in output_chi2/ or output_ttest/ folders
   - Files include test method suffix in filename

4. Understanding what a test number means:
   - Check reference/test_mapping.csv for short name
   - Check reference/insight_pairing.csv for full description

5. Understanding what codes mean:
   - Check appropriate file in reference/ folder
   - Each coding file explains the numerical mappings

================================================================================
CONTACT & SUPPORT
================================================================================

For questions about:
  - Data structure: Check reference/ folder files
  - Statistical methods: See docs/README_ANALYSIS.md
  - Code modifications: See code comments in tgif_analysis_professional.py
  - Results interpretation: See "INTERPRETING RESULTS" section above

================================================================================
VERSION INFORMATION
================================================================================

This codebase supports:
  - 34 A/B tests (Tests 1-34)
  - Two statistical test methods (chi-square and one-sided t-test)
  - Demographic filtering for Tests 9, 12, and 34
  - Binary conversion of 1-5 scale responses
  - Automatic DV2 analysis when DV1 shows no significance

Last updated: 2025

================================================================================

