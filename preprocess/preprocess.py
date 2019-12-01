import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import os

data_path = "./kaggle/data.csv"
housing = pd.read_csv(data_path, parse_dates=["date"])

# -- inspect
# housing.info()
# housing.head(10)
# housing.groupby(["date"]).mean()
# housing.describe()

# housing.boxplot(['price'], figsize=(10, 10))
# housing.hist(bins=50, figsize=(15, 15))

# -- preprocess
housing["date"] = pd.to_numeric(housing["date"]) / 1000000
housing = housing.sort_values(by=["date"])

groupby_statezip_mean_prices = housing.groupby("statezip").mean().price
city_mean_price = [0] * len(housing)
for index, housing_row in housing.iterrows():
    city_mean_price[index] = groupby_statezip_mean_prices[housing_row["statezip"]]
housing["city_mean_price"] = city_mean_price

housing = housing.drop(columns=["street", "city", "country", "statezip"])

# -- split features and prices and write them to csv
prices = housing.iloc[:, 1]

housing = housing.drop(columns=["price"])
preprocessed_features = housing

prices.to_csv(path_or_buf=os.path.dirname(os.path.realpath(__file__)) + "/prices.csv", header=False, index=False)
preprocessed_features.to_csv(path_or_buf=os.path.dirname(os.path.realpath(__file__)) + "/preprocessed_features.csv", header=False, index=False)