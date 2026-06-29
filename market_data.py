import yfinance as yf
import pandas as pd
from config import START_DATE, SYM

def download_market():

    df = pd.DataFrame()

    for k, symbol in SYM.items():

        print("download", k)

        hist = yf.download(
            symbol,
            start=START_DATE,
            progress=False
        )

        df[k] = hist["Close"]

    return df.dropna()