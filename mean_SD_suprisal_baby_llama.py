import pandas as pd

# Load the CSV
df = pd.read_csv("results_baby_llama.csv")

# Step 1: Filter Regular Plural and Irregular Plural
regular_plural = df[df['Category'] == 'Regular Plural'].reset_index(drop=True)
irregular_plural = df[df['Category'] == 'Irregular Plural'].reset_index(drop=True)

# Step 2–3: Alternating rows — even index = singular, odd index = plural
singular_rows_reg = regular_plural.iloc[::2]
plural_rows_reg = regular_plural.iloc[1::2]

singular_rows_irreg = irregular_plural.iloc[::2]
plural_rows_irreg = irregular_plural.iloc[1::2]

# Step 4: Compute stats for Regular Plural
stats_reg = {
    "Group": ["Singular", "Plural"],
    "Mean_Surprisal_Head": [
        singular_rows_reg["Surprisal head"].mean(),
        plural_rows_reg["Surprisal head"].mean()
    ],
    "Std_Surprisal_Head": [
        singular_rows_reg["Surprisal head"].std(),
        plural_rows_reg["Surprisal head"].std()
    ]
}

result_df_reg = pd.DataFrame(stats_reg)
print(f"\n\n Regular Plural \n\n {result_df_reg}")

# Step 4: Compute stats for Irregular Plural
stats_irreg = {
    "Group": ["Singular", "Plural"],
    "Mean_Surprisal_Head": [
        singular_rows_irreg["Surprisal head"].mean(),
        plural_rows_irreg["Surprisal head"].mean()
    ],
    "Std_Surprisal_Head": [
        singular_rows_irreg["Surprisal head"].std(),
        plural_rows_irreg["Surprisal head"].std()
    ]
}

result_df_irreg = pd.DataFrame(stats_irreg)
print(f"\n\n Irregular Plural \n\n {result_df_irreg}")

# Combine both results into one DataFrame
final_result_df = pd.concat([result_df_reg, result_df_irreg], ignore_index=True)

# Step 5: Save results to a CSV file
final_result_df.to_csv("mean_SD_babble.csv", index=False)

print("\nResults saved to 'mean_SD_baby_llama.csv'.")