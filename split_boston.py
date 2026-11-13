import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml


dataset_name = "boston"
data = fetch_openml(name=dataset_name, as_frame=True)
X, y = data.data, data.target
df = pd.concat([X, y], axis=1)
df.rename(columns={"MEDV": "target"}, inplace=True)

for i in range(1, 10):
    print(f"Processing dataset with random seed {i}...")

    if not os.path.exists(f"./data/boston_{i}"):
        os.makedirs(f"./data/boston_{i}")

    rand_seed = i

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=rand_seed)
    train_df, val_df = train_test_split(train_df, test_size=0.25, random_state=rand_seed)

    # normalize the data frame
    scalar = StandardScaler()
    X_train = scalar.fit_transform(train_df.drop(columns=["target"]))
    X_val = scalar.transform(val_df.drop(columns=["target"]))
    X_test = scalar.transform(test_df.drop(columns=["target"]))

    y_mean, y_std = train_df["target"].mean(), train_df["target"].std()
    y_train = (train_df["target"] - y_mean) / y_std
    y_val = (val_df["target"] - y_mean) / y_std
    y_test = (test_df["target"] - y_mean) / y_std

    # remerge into a single dataframe
    columns = train_df.drop(columns=["target"]).columns
    train_df = pd.DataFrame(X_train, columns=columns, index=train_df.index)
    train_df["target"] = y_train
    val_df = pd.DataFrame(X_val, columns=columns, index=val_df.index)
    val_df["target"] = y_val
    test_df = pd.DataFrame(X_test, columns=columns, index=test_df.index)
    test_df["target"] = y_test

    # save the dataframes to csv
    train_df.to_csv(f"./data/boston_{rand_seed}/train.csv", index=False)
    val_df.to_csv(f"./data/boston_{rand_seed}/val.csv", index=False)
    test_df.to_csv(f"./data/boston_{rand_seed}/test.csv", index=False)
