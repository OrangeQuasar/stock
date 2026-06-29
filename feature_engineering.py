def create_features(df):

    # 日経平均と為替以外の米国指標などは、日本時間15時時点で当日の値が確定していないため、
    # 未来の情報をカンニングしてしまう「データリーク」を防ぐ目的で前日にシフトします。
    lag_cols = ["sp500", "nasdaq", "vix", "sox", "cme"]
    for col in lag_cols:
        if col in df.columns:
            df[col] = df[col].shift(1)

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