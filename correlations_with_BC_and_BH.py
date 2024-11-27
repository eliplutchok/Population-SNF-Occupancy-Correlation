import pandas as pd
import numpy as np
from scipy.stats import pearsonr, boxcox
from itertools import combinations
from multiprocessing import Pool, cpu_count
from statsmodels.stats.multitest import multipletests
import time

census_df = pd.read_csv('census_aggregated_by_state_2020.csv')
cms_df = pd.read_csv('cms_aggregated_by_state_2020.csv')

# Adjust census data column names to a simpler naming convention
census_df.rename(columns={
    'Count!!SEX AND AGE!!Total population': 'Total_Population',
    'Count!!SEX AND AGE!!Male population': 'Male_Population',
    'Count!!SEX AND AGE!!Female population': 'Female_Population'
}, inplace=True)

# Function to extract the census age group columns based on the population type
def get_age_group_columns(df, population_type):
    prefix = f'Count!!SEX AND AGE!!{population_type}!!'
    age_group_columns = [col for col in df.columns if col.startswith(prefix) and 'Selected Age Categories' not in col]
    return age_group_columns

# Merge the two dataframes on the 'State' column
df = pd.merge(census_df, cms_df, on='State', how='left')
# Convert 'total_residents' from string to float
df['total_residents'] = df['total_residents'].astype(float)

# Function to compute the correlation between the ratio of nursing home residents and the sum of age groups
def compute_correlation(args):
    age_group2_combo, age_group_values, df_group1_sum, nursing_home_ratio = args
    try:
        # Sum the age groups in age_group2_combo using precomputed values
        Group2_Sum = np.sum([age_group_values[age] for age in age_group2_combo], axis=0)

        # Compute the ratio
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(Group2_Sum != 0, df_group1_sum / Group2_Sum, np.nan)

        # Mask invalid values
        valid_mask = ~np.isnan(ratio) & ~np.isnan(nursing_home_ratio)
        ratio = ratio[valid_mask]
        nh_ratio = nursing_home_ratio[valid_mask]

        # Ensure the data is strictly positive for Box-Cox transformation
        positive_mask = (ratio > 0) & (nh_ratio > 0)
        ratio = ratio[positive_mask]
        nh_ratio = nh_ratio[positive_mask]

        # Check if sufficient data points exist
        if len(ratio) > 1:
            # Apply Box-Cox transformation
            ratio_transformed, _ = boxcox(ratio)
            nh_ratio_transformed, _ = boxcox(nh_ratio)

            # Compute Pearson correlation
            corr_coef, p_value = pearsonr(ratio_transformed, nh_ratio_transformed)
            return (abs(corr_coef), corr_coef, p_value, age_group2_combo)
        else:
            return (0, 0, 1, age_group2_combo)
    except Exception as e:
        # Return a tuple with zero correlation on exception
        return (0, 0, 1, age_group2_combo)

# Function to perform the analysis for a given population type
def perform_analysis(population_type):
    # Get the total population column based on the population type
    population_key = population_type.replace(" population", "").replace(" ", "_")
    total_population_column = f'{population_key}_Population'

    # Calculate the ratio of total nursing home residents to total state population
    df['NursingHomeResidents_to_TotalPopulation_Ratio'] = df['total_residents'] / df[total_population_column]

    # Define and construct the list of group1 column names, sum them across each row, and convert to NumPy array
    group1_age = ['80 to 84 years', '85 years and over']
    group1_columns = [f'Count!!SEX AND AGE!!{population_type}!!{age}' for age in group1_age]
    df['Group1_Sum'] = df[group1_columns].sum(axis=1)
    df_group1_sum = df['Group1_Sum'].values

    # Get the age group columns for the specified population type for Group2
    age_group2_columns = get_age_group_columns(df, population_type)

    # Ensure the age group columns are present
    if len(age_group2_columns) == 0:
        print(f"No age group columns found for population type: {population_type}")
        return

    # Extract values for each age group column and store them in a dictionary for Group2
    age_group_values = {col: df[col].values for col in age_group2_columns}

    # Store 'NursingHomeResidents_to_TotalPopulation_Ratio' as a NumPy array
    nursing_home_ratio = df['NursingHomeResidents_to_TotalPopulation_Ratio'].values

    # Generate all possible combinations of age groups for Group2
    combinations_list = []
    for r in range(1, len(age_group2_columns) + 1):
        combinations_list.extend(combinations(age_group2_columns, r))

    total_combinations = len(combinations_list)
    print(f"Total number of function calls that will be made: {total_combinations}\n")

    # Parallelize the computation of correlations
    def parallel_compute():
        num_processes = cpu_count() - 1
        print(f"Using {num_processes} CPU cores for computation.\n")

        # Prepare arguments for each process
        args_list = [
            (combo, age_group_values, df_group1_sum, nursing_home_ratio) for combo in combinations_list
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

    # Apply Benjamini-Hochberg correction
    corrected = multipletests(p_values, alpha=0.05, method='fdr_bh')
    adjusted_p_values = corrected[1]

    # Update results with adjusted p-values
    results_with_adjusted_p = [
        (abs_corr_coef, corr_coef, adj_p_value, age_group_combo)
        for (abs_corr_coef, corr_coef, _, age_group_combo), adj_p_value in zip(results, adjusted_p_values)
    ]

    # Filter significant results based on adjusted p-values
    significant_results = [
        result for result in results_with_adjusted_p if result[2] < 0.05
    ]

    # Sort significant results by absolute value of correlation coefficient
    significant_results.sort(key=lambda x: x[0], reverse=True)

    # Select top 10 results
    top_n = 10
    top_results = significant_results[:top_n]

    # Display the top significant results
    print(f"\nTop {top_n} significant combinations for {population_type}:")
    for i, result in enumerate(top_results, start=1):
        age_group_names = [age.replace(f'Count!!SEX AND AGE!!{population_type}!!', '') for age in result[3]]
        print(f"{i}. Corr Coef: {result[1]:.4f}, Adjusted P-value: {result[2]:.6f}, Age Group2: {age_group_names}")

# Function to compute the correlation between the ratio of female to male populations
# for every age group combination and the nursing home resident ratio
def compute_gender_ratio_correlation(args):
    age_group_combo, female_age_group_values, male_age_group_values, nursing_home_ratio = args
    try:
        # Sum the age groups in the combination for females and males
        female_sum = np.sum([female_age_group_values[age] for age in age_group_combo], axis=0)
        male_sum = np.sum([male_age_group_values[age] for age in age_group_combo], axis=0)

        # Compute the ratio of females to males
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(male_sum != 0, female_sum / male_sum, np.nan)

        # Mask invalid values
        valid_mask = ~np.isnan(ratio) & ~np.isnan(nursing_home_ratio)
        ratio = ratio[valid_mask]
        nh_ratio = nursing_home_ratio[valid_mask]

        # Ensure the data is strictly positive for Box-Cox transformation
        positive_mask = (ratio > 0) & (nh_ratio > 0)
        ratio = ratio[positive_mask]
        nh_ratio = nh_ratio[positive_mask]

        # Check if sufficient data points exist
        if len(ratio) > 1:
            # Apply Box-Cox transformation
            ratio_transformed, _ = boxcox(ratio)
            nh_ratio_transformed, _ = boxcox(nh_ratio)

            # Compute Pearson correlation
            corr_coef, p_value = pearsonr(ratio_transformed, nh_ratio_transformed)
            return (abs(corr_coef), corr_coef, p_value, age_group_combo)
        else:
            return (0, 0, 1, age_group_combo)
    except Exception as e:
        # Return a tuple with zero correlation on exception
        return (0, 0, 1, age_group_combo)

# Function to perform the analysis for the ratio of female to male populations
def perform_gender_ratio_analysis():
    # Define population types
    female_population_type = 'Female population'
    male_population_type = 'Male population'

    # Get age group columns for females and males
    female_age_group_columns = get_age_group_columns(df, female_population_type)
    male_age_group_columns = get_age_group_columns(df, male_population_type)

    # Ensure the age group columns are present
    if len(female_age_group_columns) == 0 or len(male_age_group_columns) == 0:
        print("No age group columns found for male or female populations.")
        return

    # Extract age group names from the columns
    female_age_groups = [col.replace(f'Count!!SEX AND AGE!!{female_population_type}!!', '') for col in female_age_group_columns]
    male_age_groups = [col.replace(f'Count!!SEX AND AGE!!{male_population_type}!!', '') for col in male_age_group_columns]

    # Find common age groups between male and female populations
    common_age_groups = set(female_age_groups).intersection(male_age_groups)

    # Create mapping from age group names to column names
    female_age_group_map = {age: f'Count!!SEX AND AGE!!{female_population_type}!!{age}' for age in common_age_groups}
    male_age_group_map = {age: f'Count!!SEX AND AGE!!{male_population_type}!!{age}' for age in common_age_groups}

    # Prepare combinations of age groups using common age groups
    age_group_list = list(common_age_groups)

    # Generate all possible combinations of age groups
    combinations_list = []
    for r in range(1, len(age_group_list) + 1):
        combinations_list.extend(combinations(age_group_list, r))

    total_combinations = len(combinations_list)
    print(f"Total number of function calls that will be made: {total_combinations}\n")

    # Precompute values for each individual age group for females and males
    female_age_group_values = {age: df[female_age_group_map[age]].values for age in age_group_list}
    male_age_group_values = {age: df[male_age_group_map[age]].values for age in age_group_list}

    # Store 'NursingHomeResidents_to_TotalPopulation_Ratio' as a NumPy array
    nursing_home_ratio = df['total_residents'].values / df['Total_Population'].values
    
    def parallel_compute():
        num_processes = cpu_count() - 1
        print(f"Using {num_processes} CPU cores for computation.\n")

        # Prepare arguments for each process
        args_list = [
            (combo, female_age_group_values, male_age_group_values, nursing_home_ratio) for combo in combinations_list
        ]

        with Pool(num_processes) as pool:
            results = pool.map(compute_gender_ratio_correlation, args_list)

        return results

    start_time = time.time()
    results = parallel_compute()
    end_time = time.time()
    print(f"\nTotal computation time: {end_time - start_time:.2f} seconds")

    # Extract p-values from results
    p_values = [result[2] for result in results]

    # Apply Benjamini-Hochberg correction
    corrected = multipletests(p_values, alpha=0.05, method='fdr_bh')
    adjusted_p_values = corrected[1]

    # Update results with adjusted p-values
    results_with_adjusted_p = [
        (abs_corr_coef, corr_coef, adj_p_value, age_group_combo)
        for (abs_corr_coef, corr_coef, _, age_group_combo), adj_p_value in zip(results, adjusted_p_values)
    ]

    # Filter significant results based on adjusted p-values
    significant_results = [
        result for result in results_with_adjusted_p if result[2] < 0.05
    ]

    # Sort significant results by absolute value of correlation coefficient
    significant_results.sort(key=lambda x: x[0], reverse=True)

    # Select top 10 results
    top_n = 10
    top_results = significant_results[:top_n]

    # Display the top significant results
    print(f"\nTop {top_n} significant combinations for Female to Male Ratios:")
    for i, result in enumerate(top_results, start=1):
        age_group_names = list(result[3])
        print(f"{i}. Corr Coef: {result[1]:.4f}, Adjusted P-value: {result[2]:.6f}, Age Groups: {age_group_names}")

if __name__ == "__main__":
    # Perform analysis for Total population, Male population, and Female population
    for population_type in ['Total population', 'Male population', 'Female population']:
        print(f"\nPerforming analysis for {population_type}:\n")
        perform_analysis(population_type)

    # Perform the gender ratio analysis
    print("\nPerforming analysis for Female to Male Ratios:\n")
    perform_gender_ratio_analysis()