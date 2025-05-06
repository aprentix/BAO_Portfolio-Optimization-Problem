import pandas as pd

def load_best_config(csv_path: str, algorithm: str, repair_method: str = "normalize") -> dict:
    df = pd.read_csv(csv_path)
    
    # Filter by algorithm and repair method
    df = df[(df["algorithm"] == algorithm) & (df["repair_method"] == repair_method)]

    if df.empty:
        raise ValueError(f"No configuration found for {algorithm} with repair method '{repair_method}'.")

    # Sort by best Sharpe ratio
    best = df.sort_values("mean_sharpe", ascending=False).iloc[0]

    # Dynamic loading of all relevant parameters
    exclude_cols = {"algorithm", "repair_method", "mean_sharpe", "std_sharpe", "mean_return", "mean_time"}
    config = {col: best[col] for col in df.columns if col not in exclude_cols and pd.notna(best[col])}

    # Convert automatically to int/float, if necessary
    for k, v in config.items():
        if isinstance(v, float) and v.is_integer():
            config[k] = int(v)

    return config