import joblib
import numpy as np
import warnings
from typing import Tuple
from dask.dataframe import read_csv, DataFrame
from dask.array import Array
from dask_ml.linear_model import LinearRegression
from dask_ml.preprocessing import StandardScaler
from dask_ml.model_selection import train_test_split
from dask_ml.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore", category=FutureWarning)


# Funkcja do skalowania danych
def data_preparation(data: DataFrame) -> DataFrame:
    data = data.dropna()
    scaler = StandardScaler()
    scaler.fit(data)

    data = scaler.transform(data)
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

    with open("./data/output_log.txt", "a+") as f:
        f.write(f'Accuracy (R^2): {round(lr.score(x_test, y_test), 2) * 100}%\n')
        f.write(f'MAE : {round(mean_absolute_error(y_test, y_pred), 2)}\n')
        f.write(f'RMSE : {round(np.sqrt(mean_squared_error(y_test, y_pred)), 2)}\n')


if __name__ == '__main__':
    housing_data = read_csv("./data/housing.csv")
    scaled_data = data_preparation(housing_data)

    data_array = scaled_data.to_dask_array(lengths=True)

    X_training, X_testing, y_training, y_testing = split_data(data_array)

    linear_regression(X_training, X_testing, y_training, y_testing)
