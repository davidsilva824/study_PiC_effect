import pandas as pd

# Load the CSV
df = pd.read_csv("results_gpt_wee.csv")

# Filter Regular and Irregular Plural
regular_plural = df[df['Category'] == 'Regular Plural'].reset_index(drop=True)
irregular_plural = df[df['Category'] == 'Irregular Plural'].reset_index(drop=True)

# Alternating rows: even index = singular, odd index = plural
singular_reg = regular_plural.iloc[::2].reset_index(drop=True)
plural_reg = regular_plural.iloc[1::2].reset_index(drop=True)

singular_irreg = irregular_plural.iloc[::2].reset_index(drop=True)
plural_irreg = irregular_plural.iloc[1::2].reset_index(drop=True)

# Compute difference per pair
diff_reg = plural_reg["Surprisal head"] - singular_reg["Surprisal head"]
diff_irreg = plural_irreg["Surprisal head"] - singular_irreg["Surprisal head"]

# Compute stats
diff_stats = {
    "Category": ["Regular Plural", "Irregular Plural"],
    "Mean_Difference": [diff_reg.mean(), diff_irreg.mean()],
    "Std_Difference": [diff_reg.std(), diff_irreg.std()]
}

result_df_diff = pd.DataFrame(diff_stats)
print(result_df_diff)

# Save to file
result_df_diff.to_csv("mean_SD_gpt_wee_dif.csv", index=False)