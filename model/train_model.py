import time
import schedule
import joblib
import numpy as np
import warnings
from typing import Tuple
from dask_ml.impute import SimpleImputer
from dask.dataframe import read_csv, DataFrame
from dask.array import Array
from dask_ml.linear_model import LinearRegression
from dask_ml.preprocessing import StandardScaler
from dask_ml.model_selection import train_test_split
from dask_ml.metrics import mean_absolute_error, mean_squared_error


warnings.filterwarnings("ignore", category=FutureWarning)


# Funkcja do obsługi brakujących wartości
# If “mean”, then replace missing values using the mean along each column. Can only be used with numeric data.
def handle_missing_values(data: DataFrame) -> DataFrame:
    imputer = SimpleImputer(strategy='mean')
    return imputer.fit_transform(data)


# Funkcja do skalowania danych
def data_preparation(data: DataFrame) -> DataFrame:
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    return data


# Funkcja do podziału danych
def split_data(data: Array) -> Tuple[Array, Array, Array, Array]:
    x = data[:, 0:-1]
    y = data[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=33, shuffle=True)
    return x_train, x_test, y_train, y_test


# Funkcja z regresją liniową
def linear_regression(x_train, x_test, y_train, y_test) -> None:
    lr = LinearRegression()

    lr.fit(x_train, y_train)

    joblib.dump(lr, "./data/model.joblib")

    y_pred = lr.predict(x_test).compute()

    # Dask nie ma wbudowanych AOC i Loss
    with open("./data/output_log.txt", "a+") as f:
        f.write(f'Accuracy (R^2): {round(lr.score(x_test, y_test), 2) * 100}%\n')
        f.write(f'MAE : {round(mean_absolute_error(y_test, y_pred), 2)}\n')
        f.write(f'RMSE : {round(np.sqrt(mean_squared_error(y_test, y_pred)), 2)}\n')


# Funkcja tworząca model
def run():
    housing_data = read_csv("./data/housing.csv")

    housing_data = handle_missing_values(housing_data)

    # scaled_data = data_preparation(housing_data)

    data_array = housing_data.to_dask_array(lengths=True)

    x_train, x_test, y_train, y_test = split_data(data_array)

    linear_regression(x_train, x_test, y_train, y_test)


if __name__ == '__main__':
    schedule.every().hour.do(run)

    run()

    while True:
        schedule.run_pending()
        time.sleep(1)
