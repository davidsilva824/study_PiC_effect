### Status: working 

import pandas as pd
import statsmodels.formula.api as smf

# Load the surprisal data
df = pd.read_csv("results_gpt_wee.csv")

# Keep only Regular Plural category
df = df[df["Category"] == "Regular Plural"]

# Assign Plurality based on position in the list (First = Singular, Second = Plural)
df["Plurality"] = df.groupby("Head").cumcount().mod(2)

# Fit the LMM
model = smf.mixedlm("Surprisal ~ Plurality", df, groups=df["Head"])
result = model.fit(method='powell')

# Print and save results
print(result.summary())
with open("lmm_simple_results_gpt_wee.txt", "w") as f:
    f.write(result.summary().as_text())

null_model = smf.mixedlm("Surprisal ~ 1", df, groups=df["Head"]).fit(method='powell')
print(null_model.llf)  # Log-likelihood of null model
print(result.llf)  