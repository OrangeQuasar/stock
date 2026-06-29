def create_features(df):

    cols = list(df.columns)

    for col in cols:

        df[f"{col}_1D"] = df[col].pct_change(1)
        df[f"{col}_5D"] = df[col].pct_change(5)

    df["TARGET"] = (
        df["nikkei"].shift(-1)
        > df["nikkei"]
    ).astype(int)

    return df.dropna()