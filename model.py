from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

def train(df):

    X = df.drop(
        ["TARGET","nikkei"],
        axis=1
    )

    y = df["TARGET"]

    split = int(len(df)*0.8)

    X_train = X[:split]
    X_test = X[split:]

    y_train = y[:split]
    y_test = y[split:]

    model = XGBClassifier(
        tree_method="hist",
        n_estimators=500,
        max_depth=5,
        learning_rate=0.03
    )

    model.fit(X_train,y_train)

    pred = model.predict(X_test)

    acc = accuracy_score(
        y_test,
        pred
    )

    print("Accuracy:",acc)

    return model,X