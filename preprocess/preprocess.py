import numpy as np
import pandas as pd
from scipy import stats

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
housing["timestamp"] = pd.to_numeric(housing["date"]) / 1000000
# sort to simulate realistic scenario (predicting future based on past)
housing = housing.sort_values(by=["date"])

groupby_statezip_mean_prices = housing.groupby("statezip").mean().price
sqft_basement_mean = housing.sqft_basement[housing.sqft_basement != 0].mean()
yr_renovated_mean = housing.yr_renovated[housing.yr_renovated != 0].mean()

city_mean_price = [0] * len(housing)
years_since_renovated_or_built = [0] * len(housing)
normalized_sqft_basement = housing.copy()["sqft_basement"]
normalized_yr_renovated = housing.copy()["yr_renovated"]

for index, housing_row in housing.iterrows():
    # use mean price per city to identify cheap and expensive cities
    city_mean_price[index] = groupby_statezip_mean_prices[housing_row["statezip"]]
    # calculate years since renovation or building date -> strong price indicator
    if housing_row["yr_renovated"] != 0:
        years_since_renovated_or_built[index] = housing_row.date.year - housing_row["yr_renovated"]
    else:
        years_since_renovated_or_built[index] = housing_row.date.year - housing_row["yr_built"]
    # set all sqft_basement null values to mean
    if housing_row["sqft_basement"] == 0:
        normalized_sqft_basement[index] = sqft_basement_mean
    # set all yr_renovated null values to mean
    if housing_row["yr_renovated"] == 0:
        normalized_yr_renovated[index] = yr_renovated_mean

housing["city_mean_price"] = city_mean_price
housing["years_since_renovated_or_built"] = years_since_renovated_or_built
housing["sqft_basement"] = normalized_sqft_basement
housing["yr_renovated"] = normalized_yr_renovated

housing = housing.drop(columns=["date", "street", "city", "country", "statezip"])

housing = housing[((housing.price - housing.price.mean()) / housing.price.std()).abs() < 3]

# -- split features and prices and write them to csv
prices = housing.iloc[:, 0]

housing = housing.drop(columns=["price"])
preprocessed_features = housing

prices.to_csv(path_or_buf=os.path.dirname(os.path.realpath(__file__)) + "/prices.csv", header=False, index=False)
preprocessed_features.to_csv(path_or_buf=os.path.dirname(os.path.realpath(__file__)) + "/preprocessed_features.csv", header=False, index=False)