import os
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = pd.Series(diabetes.target, name="target")
df = pd.concat([X, y], axis=1)


for i in range(1, 10):
    print(f"Processing split {i}...")
    if not os.path.exists(f"./data/diabetes_{i}"):
        os.makedirs(f"./data/diabetes_{i}")

    rand_seed = i

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=rand_seed)
    train_df, val_df = train_test_split(train_df, test_size=0.25, random_state=rand_seed)

    # normalize the data frame, except the target column
    features = [col for col in train_df.columns if col != "target"]
    scaler = StandardScaler()

    # Fit on train features only
    X_train = scaler.fit_transform(train_df[features])
    X_val = scaler.transform(val_df[features])
    X_test = scaler.transform(test_df[features])

    y_mean, y_std = train_df["target"].mean(), train_df["target"].std()
    y_train = (train_df["target"] - y_mean) / y_std
    y_val = (val_df["target"] - y_mean) / y_std
    y_test = (test_df["target"] - y_mean) / y_std

    # remerge back
    train_df = pd.DataFrame(X_train, columns=features, index=train_df.index)
    train_df["target"] = y_train
    val_df = pd.DataFrame(X_val, columns=features, index=val_df.index)
    val_df["target"] = y_val
    test_df = pd.DataFrame(X_test, columns=features, index=test_df.index)
    test_df["target"] = y_test

    # save the train_df, val_df, test_df to csv
    train_df.to_csv(f"./data/diabetes_{rand_seed}/train.csv", index=False)
    val_df.to_csv(f"./data/diabetes_{rand_seed}/val.csv", index=False)
    test_df.to_csv(f"./data/diabetes_{rand_seed}/test.csv", index=False)
