import matplotlib.pyplot as plt

def run_backtest(model, X, price):

    cash = 1000000
    pos = 0

    assets = []

    for i in range(len(X)-1):

        p = model.predict_proba(
            X.iloc[i:i+1]
        )[0][1]

        now = price.iloc[i]

        if p > 0.7:

            if pos == 0:
                pos = cash / now
                cash = 0

        elif p < 0.3:

            if pos > 0:
                cash = pos * now
                pos = 0

        asset = cash if pos == 0 else pos * now

        assets.append(asset)

    plt.plot(assets)
    plt.title("Backtest")
    plt.show()