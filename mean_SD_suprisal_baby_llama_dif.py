import pandas as pd
import scipy.stats as stats
import numpy as np

# Load the CSV
df = pd.read_csv("results_baby_llama.csv")

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

# Function to compute 95% CI
def compute_95ci(data):
    mean = np.mean(data)
    sem = stats.sem(data)
    ci = stats.t.interval(0.95, len(data)-1, loc=mean, scale=sem)
    return ci

ci_reg = compute_95ci(diff_reg)
ci_irreg = compute_95ci(diff_irreg)

# Compute stats
diff_stats = {
    "Category": ["Regular Plural", "Irregular Plural"],
    "Mean_Difference": [diff_reg.mean(), diff_irreg.mean()],
    "Std_Difference": [diff_reg.std(), diff_irreg.std()],
    "CI_95_Lower": [ci_reg[0], ci_irreg[0]],
    "CI_95_Upper": [ci_reg[1], ci_irreg[1]]
}

result_df_diff = pd.DataFrame(diff_stats)
print(result_df_diff)

# Save to file
result_df_diff.to_csv("mean_SD_CI95_baby_llama_dif.csv", index=False)