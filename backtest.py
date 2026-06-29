import matplotlib.pyplot as plt

def run_backtest(model, X_test, price_test):

    cash = 1000000
    pos = 0
    assets = []
    wins = 0
    trades = 0

    for i in range(len(X_test)-1):

        p = model.predict_proba(
            X_test.iloc[i:i+1]
        )[0][1]

        today = price_test.iloc[i]
        tomorrow = price_test.iloc[i+1]

        # 買いシグナル
        if p > 0.7:
            if pos == 0:  # ポジションがない時だけ買う
                pos = cash / today
                cash = 0
                trades += 1
                # 買った翌日に上がっていれば勝ちとしてカウント
                if tomorrow > today:
                    wins += 1

        # 売りシグナル
        elif p < 0.3:
            if pos > 0:   # ポジションがある時だけ売る
                cash = pos * today
                pos = 0

        asset = (
            cash
            if pos == 0
            else pos * today
        )
        assets.append(asset)

    final_asset = assets[-1]

    win_rate = (
        wins/trades*100
        if trades > 0 else 0
    )

    print("\n========== バックテスト (テストデータのみ) ==========")
    print(f"取引回数: {trades}")
    print(f"勝率    : {win_rate:.1f}%")
    print(f"最終資産: {final_asset:,.0f}円")

    plt.figure(figsize=(10,5))
    plt.plot(assets)
    plt.title("Backtest Asset Curve (Out-of-Sample)")
    plt.grid()
    plt.show()
