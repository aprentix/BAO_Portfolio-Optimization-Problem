# !! TODO: Remove this script after fixing the algorithm names in the CSV files. !!

import pandas as pd

filename = "fine_tuning_results.csv"

df = pd.read_csv(filename)

# PSO columns
pso_cols = ["swarm_size", "max_iterations", "w", "c1", "c2"]

# Find rows labeled as GA but with any PSO column filled (not empty and not NaN)
mask = (df["algorithm"] == "GA") & df[pso_cols].notnull().any(axis=1)

# Update those rows to PSO
df.loc[mask, "algorithm"] = "PSO"

df.to_csv(filename, index=False)
print("✅ Algorithm column updated for PSO rows where needed.")