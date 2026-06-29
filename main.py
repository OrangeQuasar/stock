from market_data import download_market
from indicators import add_indicators
from feature_engineering import create_features
from news_fetcher import fetch_news
from llm_analyzer import analyze_news
from model import train
from backtest import run_backtest

print("データ取得")

df = download_market()

print("\n========== 市場状況 ==========")

market_names = {
    "nikkei": "日経平均",
    "sp500": "S&P500",
    "nasdaq": "NASDAQ",
    "sox": "SOX",
    "vix": "VIX",
    "usdjpy": "ドル円",
    "cme": "CME日経先物"
}

for col in market_names:

    latest = df[col].iloc[-1]
    prev = df[col].iloc[-2]

    change = (latest-prev)/prev*100

    print(
        f"{market_names[col]:<12}: "
        f"{latest:,.2f} "
        f"({change:+.2f}%)"
    )

print("テクニカル計算")

df = add_indicators(df)

print("\n========== テクニカル ==========")

rsi = df["RSI"].iloc[-1]
macd = df["MACD"].iloc[-1]
signal = df["MACD_SIGNAL"].iloc[-1]

print(f"RSI            : {rsi:.2f}")
print(f"MACD           : {macd:.2f}")
print(f"MACD Signal    : {signal:.2f}")

if macd > signal:
    print("MACD判定       : 買い")
else:
    print("MACD判定       : 売り")

price = df["nikkei"].iloc[-1]
bb_high = df["BB_HIGH"].iloc[-1]
bb_low = df["BB_LOW"].iloc[-1]

dist = (price - bb_high) / price * 100

print(f"BB上限距離     : {dist:.2f}%")

if price > bb_high:
    print("ボリンジャー   : 過熱")
elif price < bb_low:
    print("ボリンジャー   : 売られ過ぎ")
else:
    print("ボリンジャー   : 中立")

print("ニュース取得")

news = fetch_news()

print("\n========== ニュース ==========")
print(news)

score = analyze_news(news)

print("\nLLMニューススコア:", score)

if score >= 3:
    print("ニュース判定: 強気")
elif score >= 1:
    print("ニュース判定: やや強気")
elif score == 0:
    print("ニュース判定: 中立")
else:
    print("ニュース判定: 弱気")

# 削除: df["LLM_SCORE"] = score
# (過去の全データに今日のスコアが混ざるのを防ぐため削除)

df = create_features(df)

print("学習")

# model.pyの変更に合わせて受け取る変数を増やす
model, X, X_test, df_test, importance = train(df)

print("\n========== 重要特徴量 TOP10 ==========")
for name, value in importance[:10]:
    print(f"{name:<25}{value:.4f}")

print("\n========== 最新特徴量 ==========")
latest = X.iloc[-1]
for col in ["cme_1D", "nasdaq_1D", "sox_1D", "sp500_1D", "usdjpy_1D", "RSI"]:
    if col in latest.index:
        print(f"{col:<15}: {latest[col]:.3f}")

latest = X.iloc[-1:]

prob = model.predict_proba(latest)[0][1]

print("\n=================")
print(f"翌日上昇確率:{prob*100:.1f}%")

print("\n========== 総合判定 ==========")
print(f"AI予測確率: {prob*100:.1f}%")
print(f"ニューススコア: {score}")

# 予測確率とニューススコアを組み合わせて最終判定を出すロジックに変更
if prob > 0.60 and score >= 2:
    print("判定: 非常に強気")
    print("推奨: 買い候補")
elif prob > 0.55:
    print("判定: 強気")
    print("推奨: 保有継続")
elif prob > 0.45:
    print("判定: 中立")
    print("推奨: 様子見")
elif prob > 0.35:
    print("判定: 弱気")
    print("推奨: ポジション縮小")
else:
    print("判定: 非常に弱気")
    print("推奨: 新規購入見送り")

# バックテストには、AIが学習していない「テストデータ(X_test)」のみを渡す
run_backtest(
    model,
    X_test,
    df_test["nikkei"]
)