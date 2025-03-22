### Status: working

import pandas as pd
import statsmodels.formula.api as smf

# Load the surprisal data
df = pd.read_csv("results_babble.csv")

# Keep only Regular Plural category
df = df[df["Category"] == "Regular Plural"]

# Assign Plurality based on position in the list (First = Singular, Second = Plural)
df["Plurality"] = df.groupby("Head").cumcount().mod(2)

# Use only the last column (Surprisal head) for analysis
df = df.rename(columns={"Surprisal head": "Surprisal"})  # Rename for clarity

# Fit the LMM
model = smf.mixedlm("Surprisal ~ Plurality", df, groups=df["Head"])
result = model.fit(method='powell')

# Print and save results
print(result.summary())
with open("lmm_simple_results_babble.txt", "w") as f:
    f.write(result.summary().as_text())

# Fit the null model (without Plurality)
null_model = smf.mixedlm("Surprisal ~ 1", df, groups=df["Head"]).fit(method='powell')

# Print log-likelihoods for model comparison
print(null_model.llf)  # Log-likelihood of null model
print(result.llf)  # Log-likelihood of full model