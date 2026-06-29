from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

def train(df):

    X = df.drop(
        ["TARGET","nikkei"],
        axis=1
    )

    y = df["TARGET"]

    # 80%を学習、20%をテスト（バックテスト用）に分割
    split = int(len(df)*0.8)

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    model = XGBClassifier(
        tree_method="hist",
        n_estimators=500,
        max_depth=5,
        learning_rate=0.03
    )

    # 学習データのみで訓練する
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    acc = accuracy_score(
        y_test,
        pred
    )

    print("Accuracy (Test Data):", acc)

    importance = sorted(
        zip(
            X.columns,
            model.feature_importances_
        ),
        key=lambda x: x[1],
        reverse=True
    )

    # バックテスト用に X_test と df のテスト期間も返す
    return model, X, X_test, df.iloc[split:], importance
