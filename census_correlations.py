import pandas as pd
import numpy as np
from scipy.stats import pearsonr, boxcox
from itertools import combinations
from multiprocessing import Pool, cpu_count
import time
import json

# Configuration Section
INPUT_CENSUS_FILE = 'census_aggregated_by_state_2020.csv'
INPUT_CMS_FILE = 'cms_aggregated_by_state_2020.csv'
OUTPUT_FILENAME_TEMPLATE = 'age_group_correlation_{population_type}.json'

# Age group for Group 1 (fixed)
GROUP1_AGE = ['85 years and over']

# Population types to analyze
POPULATION_TYPES = ['Total population', 'Male population', 'Female population']

# Top N significant results to display
TOP_N_RESULTS = 10

# Alpha level for statistical significance
ALPHA_LEVEL = 0.05

# Number of CPU cores to use (set to None to use all available cores minus one)
NUM_PROCESSES = None  # None means use cpu_count() - 1

# Load datasets
census_df = pd.read_csv(INPUT_CENSUS_FILE)
cms_df = pd.read_csv(INPUT_CMS_FILE)

# Adjust census data column names to a simpler naming convention
census_df.rename(columns={
    'Count!!SEX AND AGE!!Total population': 'Total_Population',
    'Count!!SEX AND AGE!!Male population': 'Male_Population',
    'Count!!SEX AND AGE!!Female population': 'Female_Population'
}, inplace=True)

# Function to extract the census age group columns based on the population type
def get_age_group_columns(df, population_type, exclude_ages=None):
    prefix = f'Count!!SEX AND AGE!!{population_type}!!'
    age_group_columns = [col for col in df.columns if col.startswith(prefix) and 'Selected Age Categories' not in col]
    if exclude_ages:
        exclude_columns = [f'{prefix}{age}' for age in exclude_ages]
        age_group_columns = [col for col in age_group_columns if col not in exclude_columns]
    return age_group_columns

# Merge the two dataframes on the 'State' column
df = pd.merge(census_df, cms_df, on='State', how='left')

# Function to compute the correlation between the sum of Group 1 and sum of Group 2
def compute_correlation(args):
    age_group2_combo, age_group_values1, age_group_values2 = args
    try:
        # Sum the age groups in Group 1
        Group1_Sum = np.sum([age_group_values1[age] for age in GROUP1_AGE], axis=0)

        # Sum the age groups in the current combination for Group 2
        Group2_Sum = np.sum([age_group_values2[age] for age in age_group2_combo], axis=0)

        # Compute the ratio of Group1_Sum to Group2_Sum
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(Group2_Sum != 0, Group1_Sum / Group2_Sum, np.nan)

        # Mask invalid values
        valid_mask = ~np.isnan(ratio)
        ratio = ratio[valid_mask]
        group1_values = Group1_Sum[valid_mask]

        # Ensure the data is strictly positive for Box-Cox transformation
        positive_mask = (ratio > 0) & (group1_values > 0)
        ratio = ratio[positive_mask]
        group1_values = group1_values[positive_mask]

        # Check if sufficient data points exist
        if len(ratio) > 1:
            # Apply Box-Cox transformation
            ratio_transformed, _ = boxcox(ratio)
            group1_transformed, _ = boxcox(group1_values)

            # Compute Pearson correlation
            corr_coef, p_value = pearsonr(ratio_transformed, group1_transformed)
            return (abs(corr_coef), corr_coef, p_value, age_group2_combo)
        else:
            return (0, 0, 1, age_group2_combo)
    except Exception as e:
        # Return a tuple with zero correlation on exception
        return (0, 0, 1, age_group2_combo)

# Function to perform the analysis for a given population type
def perform_analysis(population_type):
    # Get the age group columns for the specified population type
    age_group_columns = get_age_group_columns(df, population_type)
    age_group_columns = [col for col in age_group_columns if col not in GROUP1_AGE]

    # Ensure the age group columns are present
    if len(age_group_columns) == 0:
        print(f"No age group columns found for population type: {population_type}")
        return

    # Extract age group names
    age_group_names = [col.replace(f'Count!!SEX AND AGE!!{population_type}!!', '') for col in age_group_columns]

    # Extract values for each age group column and store them in dictionaries
    age_group_values1 = {age: df[f'Count!!SEX AND AGE!!{population_type}!!{age}'].values for age in GROUP1_AGE}
    age_group_values2 = {age: df[f'Count!!SEX AND AGE!!{population_type}!!{age}'].values for age in age_group_names}

    # Generate all possible combinations of age groups for Group 2
    combinations_list = []
    for r in range(1, len(age_group_names) + 1):
        combinations_list.extend(combinations(age_group_names, r))

    total_combinations = len(combinations_list)
    print(f"Total number of function calls that will be made: {total_combinations}\n")

    # Parallelize the computation of correlations
    def parallel_compute():
        num_processes = cpu_count() - 1 if NUM_PROCESSES is None else NUM_PROCESSES
        print(f"Using {num_processes} CPU cores for computation.\n")

        # Prepare arguments for each process
        args_list = [
            (combo, age_group_values1, age_group_values2) for combo in combinations_list
        ]

        with Pool(num_processes) as pool:
            results = pool.map(compute_correlation, args_list)

        return results

    start_time = time.time()
    results = parallel_compute()
    end_time = time.time()
    print(f"\nTotal computation time: {end_time - start_time:.2f} seconds")

    # Extract p-values from results
    p_values = [result[2] for result in results]

    # Filter significant results based on the original p-values
    significant_results = [
        result for result in results if result[2] < ALPHA_LEVEL
    ]

    # Sort significant results by absolute value of correlation coefficient
    significant_results.sort(key=lambda x: x[0], reverse=True)

    # Select top N results
    top_results = significant_results[:TOP_N_RESULTS]

    # Display the top significant results
    print(f"\nTop {TOP_N_RESULTS} significant combinations for {population_type}:")
    for i, result in enumerate(top_results, start=1):
        age_group_combo = list(result[3])
        print(f"{i}. Corr Coef: {result[1]:.4f}, P-value: {result[2]:.6f}, Age Group2: {age_group_combo}")

    # After computing the top_results
    if top_results:
        top_result = top_results[0]
        age_group_combo = list(top_result[3])

        # Recalculate the ratio using the top age group combination
        Group1_Sum = np.sum([age_group_values1[age] for age in GROUP1_AGE], axis=0)
        Group2_Sum = np.sum([age_group_values2[age] for age in age_group_combo], axis=0)

        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(Group2_Sum != 0, Group1_Sum / Group2_Sum, np.nan)

        # Prepare data for JSON
        output_data = {
            'population_type': population_type,
            'age_group1': GROUP1_AGE,
            'age_group2': age_group_combo,
            'correlation_coefficient': float(top_result[1]),
            'p_value': float(top_result[2]),
            'state_data': []
        }

        # Iterate over each state to collect ratio data
        for idx, state in enumerate(df['State']):
            if not np.isnan(ratio[idx]):
                output_data['state_data'].append({
                    'state': state,
                    'ratio': float(ratio[idx]),
                    'group1_sum': int(Group1_Sum[idx]),
                    'group2_sum': int(Group2_Sum[idx])
                })

        # Write to JSON file
        filename = OUTPUT_FILENAME_TEMPLATE.format(population_type=population_type.replace(' ', '_'))
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=4)

        print(f"\nTop result data saved to {filename}")

if __name__ == "__main__":
    # Perform analysis for each population type specified in the configuration
    for population_type in POPULATION_TYPES:
        print(f"\nPerforming analysis for {population_type}:\n")
        perform_analysis(population_type)