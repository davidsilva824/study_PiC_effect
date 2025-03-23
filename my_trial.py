import pandas as pd

# Load the CSV
df = pd.read_csv("results_babble.csv")

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

# Step 4b: Mean difference (Plural - Singular)
mean_diff_reg = (plural_rows_reg["Surprisal head"] - singular_rows_reg["Surprisal head"]).mean()
mean_diff_irreg = (plural_rows_irreg["Surprisal head"] - singular_rows_irreg["Surprisal head"]).mean()

# Step 5: Save all results to txt
with open("plural_results_babble.txt", "w") as f:
    f.write("Regular Plural\n\n")
    f.write(result_df_reg.to_string(index=False))
    f.write(f"\n\nMean Difference (Plural - Singular): {mean_diff_reg:.4f}\n\n")

    f.write("Irregular Plural\n\n")
    f.write(result_df_irreg.to_string(index=False))
    f.write(f"\n\nMean Difference (Plural - Singular): {mean_diff_irreg:.4f}\n")

print("Results saved to 'plural_results_babble.txt'.")