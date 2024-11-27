import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, boxcox
import matplotlib.pyplot as plt
from scipy import stats

# Configuration Section
INPUT_CENSUS_FILE = 'census_aggregated_by_state_2020.csv'
INPUT_CMS_FILE = 'cms_aggregated_by_state_2020.csv'

# Define Group1 Ages
GROUP1_AGE = ['85 years and over']

# Define Group2 Ages (Manually Input)
GROUP2_AGE = ['45 to 49 years', '50 to 54 years', '55 to 59 years', '60 to 64 years']

# Population type to analyze
POPULATION_TYPE = 'Total population'

# Percent of residents that are in Group1
PERC_OF_RESIDENTS_THAT_ARE_IN_GROUP1 = 0.386

# Read input files
census_df = pd.read_csv(INPUT_CENSUS_FILE)
cms_df = pd.read_csv(INPUT_CMS_FILE)

# Adjust census data column names to a simpler naming convention
census_df.rename(columns={
    'Count!!SEX AND AGE!!Total population': 'Total_Population',
    'Count!!SEX AND AGE!!Male population': 'Male_Population',
    'Count!!SEX AND AGE!!Female population': 'Female_Population'
}, inplace=True)

# Merge the two dataframes on the 'State' column
df = pd.merge(census_df, cms_df, on='State', how='left')

# Convert 'total_residents' from string to float
df['total_residents'] = df['total_residents'].astype(float)

# Function to extract census age group columns for a given population type and age groups
def get_age_group_columns(df, population_type, age_groups):
    prefix = f'Count!!SEX AND AGE!!{population_type}!!'
    age_group_columns = [f'{prefix}{age}' for age in age_groups if f'{prefix}{age}' in df.columns]
    return age_group_columns

def perform_analysis():
    population_type = POPULATION_TYPE
    population_key = population_type.replace(" population", "").replace(" ", "_")
    total_population_column = f'{population_key}_Population'

    # Calculate the total number of people in GROUP1_AGE for the specified population type
    group1_columns = get_age_group_columns(df, population_type, GROUP1_AGE)
    df['Group1_Sum'] = df[group1_columns].sum(axis=1)

    # Calculate the estimated number of 85+ nursing home residents
    df['Estimated_85plus_NursingHome_Residents'] = PERC_OF_RESIDENTS_THAT_ARE_IN_GROUP1 * df['total_residents']

    # Calculate the Nursing Home Ratio (Estimated 85+ NH Residents / Total 85+ Population)
    df['NursingHome_Ratio'] = df['Estimated_85plus_NursingHome_Residents'] / df['Group1_Sum']

    # Calculate the total number of people in GROUP2_AGE for the specified population type
    group2_columns = get_age_group_columns(df, population_type, GROUP2_AGE)
    df['Group2_Sum'] = df[group2_columns].sum(axis=1)

    # Compute the ratio of Group1_Sum to Group2_Sum
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(df['Group2_Sum'] != 0, df['Group1_Sum'] / df['Group2_Sum'], np.nan)

    # Mask invalid values
    valid_mask = ~np.isnan(ratio) & ~np.isnan(df['NursingHome_Ratio'])
    ratio = ratio[valid_mask]
    nursing_home_ratio = df['NursingHome_Ratio'][valid_mask]

    if len(ratio) > 1:
        # Prepare data for Pearson correlation
        # Without Box-Cox transformation
        pearson_corr_coef, pearson_p_value = pearsonr(ratio, nursing_home_ratio)

        # With Box-Cox transformation
        # Ensure all data is positive for Box-Cox
        positive_mask = (ratio > 0) & (nursing_home_ratio > 0)
        ratio_positive = ratio[positive_mask]
        nursing_home_ratio_positive = nursing_home_ratio[positive_mask]
        ratio_boxcox, ratio_lambda = boxcox(ratio_positive)
        nursing_home_ratio_boxcox, nursing_home_lambda = boxcox(nursing_home_ratio_positive)
        pearson_boxcox_corr_coef, pearson_boxcox_p_value = pearsonr(ratio_boxcox, nursing_home_ratio_boxcox)

        # Spearman correlation (no Box-Cox needed)
        spearman_corr_coef, spearman_p_value = spearmanr(ratio, nursing_home_ratio)

        # Print the correlation results
        print(f"Pearson Correlation Coefficient (without Box-Cox): {pearson_corr_coef:.4f}")
        print(f"Pearson P-value (without Box-Cox): {pearson_p_value:.6f}\n")
        print(f"Pearson Correlation Coefficient (with Box-Cox): {pearson_boxcox_corr_coef:.4f}")
        print(f"Pearson P-value (with Box-Cox): {pearson_boxcox_p_value:.6f}\n")
        print(f"Spearman Correlation Coefficient: {spearman_corr_coef:.4f}")
        print(f"Spearman P-value: {spearman_p_value:.6f}")

        # Plot scatter plots
        plt.figure(figsize=(18, 5))

        # Plot 1: Pearson without Box-Cox
        plt.subplot(1, 3, 1)
        plt.scatter(ratio, nursing_home_ratio)
        plt.title('Pearson Correlation (without Box-Cox)')
        plt.xlabel('Group1/Group2 Ratio')
        plt.ylabel('Nursing Home Ratio')
        slope, intercept = np.polyfit(ratio, nursing_home_ratio, 1)
        x_vals = np.array([min(ratio), max(ratio)])
        y_vals = intercept + slope * x_vals
        plt.plot(x_vals, y_vals, '--', color='red')
        plt.text(0.05, 0.95, f'Corr Coef: {pearson_corr_coef:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

        # Plot 2: Pearson with Box-Cox
        plt.subplot(1, 3, 2)
        plt.scatter(ratio_boxcox, nursing_home_ratio_boxcox)
        plt.title('Pearson Correlation (with Box-Cox)')
        plt.xlabel('Box-Cox Transformed Group1/Group2 Ratio')
        plt.ylabel('Box-Cox Transformed Nursing Home Ratio')
        slope_bc, intercept_bc = np.polyfit(ratio_boxcox, nursing_home_ratio_boxcox, 1)
        x_vals_bc = np.array([min(ratio_boxcox), max(ratio_boxcox)])
        y_vals_bc = intercept_bc + slope_bc * x_vals_bc
        plt.plot(x_vals_bc, y_vals_bc, '--', color='red')
        plt.text(0.05, 0.95, f'Corr Coef: {pearson_boxcox_corr_coef:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

        # Plot 3: Spearman
        plt.subplot(1, 3, 3)
        plt.scatter(ratio, nursing_home_ratio)
        plt.title('Spearman Correlation')
        plt.xlabel('Group1/Group2 Ratio')
        plt.ylabel('Nursing Home Ratio')
        # For Spearman, we can plot the ranks
        rank_ratio = stats.rankdata(ratio)
        rank_nursing_home_ratio = stats.rankdata(nursing_home_ratio)
        slope_s, intercept_s = np.polyfit(rank_ratio, rank_nursing_home_ratio, 1)
        x_vals_s = np.array([min(rank_ratio), max(rank_ratio)])
        y_vals_s = intercept_s + slope_s * x_vals_s
        plt.plot(rank_ratio, rank_nursing_home_ratio, 'o', label='Ranked Data')
        plt.plot(x_vals_s, y_vals_s, '--', color='red', label='Best Fit Line')
        plt.legend()
        plt.text(0.05, 0.95, f'Corr Coef: {spearman_corr_coef:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

        plt.tight_layout()
        plt.show()

        # Create a dataframe to display results in table format
        results_df = pd.DataFrame({
            'Method': ['Pearson (without Box-Cox)', 'Pearson (with Box-Cox)', 'Spearman'],
            'Correlation Coefficient': [pearson_corr_coef, pearson_boxcox_corr_coef, spearman_corr_coef],
            'P-value': [pearson_p_value, pearson_boxcox_p_value, spearman_p_value]
        })

        print("\nCorrelation Results:")
        print(results_df.to_string(index=False))
    else:
        print("Not enough data points for correlation analysis.")

if __name__ == "__main__":
    perform_analysis()
