import ta

def add_indicators(df):

    df["RSI"] = ta.momentum.RSIIndicator(
        close=df["nikkei"],
        window=14
    ).rsi()

    macd = ta.trend.MACD(df["nikkei"])

    df["MACD"] = macd.macd()
    df["MACD_SIGNAL"] = macd.macd_signal()

    bb = ta.volatility.BollingerBands(
        close=df["nikkei"],
        window=20
    )

    df["BB_HIGH"] = bb.bollinger_hband()
    df["BB_LOW"] = bb.bollinger_lband()

    return df