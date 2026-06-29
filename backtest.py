import matplotlib.pyplot as plt

def run_backtest(model, X, price):

    cash = 1000000
    pos = 0

    assets = []

    wins = 0
    trades = 0

    for i in range(len(X)-1):

        p = model.predict_proba(
            X.iloc[i:i+1]
        )[0][1]

        today = price.iloc[i]
        tomorrow = price.iloc[i+1]

        if p > 0.7:

            trades += 1

            if tomorrow > today:
                wins += 1

            if pos == 0:
                pos = cash / today
                cash = 0

        elif p < 0.3:

            if pos > 0:
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

    print("\n========== バックテスト ==========")
    print(f"取引回数: {trades}")
    print(f"勝率    : {win_rate:.1f}%")
    print(f"最終資産: {final_asset:,.0f}円")

    plt.figure(figsize=(10,5))
    plt.plot(assets)
    plt.title("Backtest Asset Curve")
    plt.grid()
    plt.show()