# TGIF A/B Test Analysis - Professional Report Generator

## Overview

This script generates professional, easily modifiable reports for TGIF frozen snack A/B test data. It performs statistical analysis on purchase intent and behavioral questions across multiple A/B tests.

## Features

- **Binary Conversion**: Converts 1-5 purchase intent scores to binary (1-3 = 0, 4-5 = 1)
- **Chi-Square Testing**: Performs statistical significance testing for each control vs. treatment comparison
- **Lift Calculation**: Calculates both percentage lift and absolute net increase
- **Treatment Ranking**: Orders treatments by lift (highest to lowest) within each test
- **DV2 Analysis**: Automatically analyzes behavioral questions (DV2) for tests with no significant DV1 results
- **Demographic Filtering**: Supports filtering by age and children status for specific tests
- **Professional Reports**: Generates CSV and Markdown reports in landscape format

## Configuration

The script is easily modifiable through the configuration section at the top:

```python
# Test numbers to analyze (in ascending order)
TEST_NUMBERS = sorted([1, 2, 3, ...])

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
```

## Output Files

1. **tgif_detailed_results.csv**: Complete detailed results with:
   - Control DV1 percentage
   - For each treatment (B, C, D, E, F): DV1 %, Net Increase, P-Value
   - Control DV2 percentage
   - For each treatment: DV2 %, P-Value
   - Tests are in ascending order (1, 2, 3, ...)
   - Treatments are analyzed in order of lift (highest to lowest)

2. **tgif_summary_table.csv**: Landscape summary table with:
   - Test number
   - Control DV1 %
   - Best treatment by lift (DV1)
   - Best treatment DV1 % and P-Value
   - Control DV2 %
   - Best treatment by lift (DV2)
   - Best treatment DV2 % and P-Value

3. **tgif_summary_table.md**: Markdown version of the summary table

4. **tgif_filtered_results_summary.csv**: Filtered results for Tests 9, 12, and 34

5. **tgif_detailed_results_with_filters.csv**: Detailed results including filtered versions

6. **tgif_summary_table_with_filters.csv**: Summary table including filtered versions

## How to Modify

### Adding New Tests

Add test numbers to the `TEST_NUMBERS` list:

```python
TEST_NUMBERS = sorted([1, 2, 3, ..., 35, 36])
```

### Changing Significance Thresholds

Modify the threshold constants:

```python
P_THRESHOLD_STRICT = 0.05  # p < 0.05: Statistically significant
P_THRESHOLD_MODERATE = 0.10  # p < 0.10: Marginally significant
P_THRESHOLD_DIRECTIONAL = 0.20  # p < 0.20: Directionally interesting
```

### Adding Demographic Filters

Add filters to the `DEMOGRAPHIC_FILTERS` dictionary:

```python
DEMOGRAPHIC_FILTERS = {
    9: {'age_older': True},
    12: {'has_children': True},
    34: {'age_older': True},
    15: {'age_older': True, 'has_children': True}  # Multiple filters
}
```

### Changing Age Threshold

Modify the `AGE_THRESHOLD` constant:

```python
AGE_THRESHOLD = 50  # Change from 45 to 50
```

### Modifying Column Structure

The CSV column structure is defined in the `generate_detailed_csv()` function. To change the order or add columns, modify the `columns` list in that function.

## Statistical Methodology

- **Binary Conversion**: Scores 1-3 are coded as 0 (no purchase intent), scores 4-5 are coded as 1 (purchase intent)
- **Chi-Square Test**: Used to test independence between control and treatment groups
- **Lift Calculation**: 
  - Absolute Lift = Treatment % - Control %
  - Percentage Lift = ((Treatment % - Control %) / Control %) × 100
- **Treatment Ordering**: Treatments are sorted by absolute lift (net increase) in descending order

## Significance Markers

- ✓ (p<0.05): Statistically significant
- ~ (p<0.10): Marginally significant
- ° (p<0.20): Directionally interesting
- ✗ (p≥0.20): Not significant

## Data Requirements

- CSV file with columns: `test1`, `test2`, ..., `test34` (treatment assignments)
- DV columns: `dv1_1`, `dv1_2`, ..., `dv34_1`, `dv34_2` (purchase intent and behavioral questions)
- Demographic columns: `q18` (birth year), `q20` (children status)

## Usage

```bash
python tgif_analysis_professional.py
```

## Notes

- Tests are automatically sorted in ascending order
- Treatments within each test are ordered by lift (highest to lowest)
- DV2 analysis is only performed for tests with no significant DV1 results (p < 0.05)
- Filtered results are generated separately and also included in the "with_filters" versions of the main reports

