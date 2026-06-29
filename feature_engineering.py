def create_features(df):

    cols = list(df.columns)
    feature_cols = [col for col in cols if col != "LLM_SCORE"]

    for col in feature_cols:

        df[f"{col}_1D"] = df[col].pct_change(1)
        df[f"{col}_5D"] = df[col].pct_change(5)

    df["TARGET"] = (
        df["nikkei"].shift(-1)
        > df["nikkei"]
    ).astype(int)

    return df.dropna()