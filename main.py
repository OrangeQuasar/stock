from stock.market_data import download_market
from stock.indicators import add_indicators
from stock.feature_engineering import create_features
from stock.news_fetcher import fetch_news
from stock.llm_analyzer import analyze_news
from stock.model import train
from stock.backtest import run_backtest

print("データ取得")

df = download_market()

print("テクニカル計算")

df = add_indicators(df)

print("ニュース取得")

news = fetch_news()

print(news)

score = analyze_news(news)

print("ニューススコア:",score)

df["LLM_SCORE"] = score

df = create_features(df)

print("学習")

model, X = train(df)

latest = X.iloc[-1:]

prob = model.predict_proba(
    latest
)[0][1]

print("\n=================")
print(
    f"翌日上昇確率:{prob*100:.1f}%"
)

if prob > 0.75:
    print("判定: 非常に強気")

elif prob > 0.60:
    print("判定: 強気")

elif prob > 0.45:
    print("判定: 中立")

else:
    print("判定: 弱気")

run_backtest(
    model,
    X,
    df["nikkei"]
)