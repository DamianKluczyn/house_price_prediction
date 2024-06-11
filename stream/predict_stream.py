import numpy as np
import time
import warnings
import joblib
from typing import Generator, Tuple, Any
from streamz import Stream
from dask_ml.linear_model import LinearRegression
from dask.dataframe import read_csv, DataFrame


warnings.filterwarnings("ignore", category=FutureWarning)


def load_model() -> LinearRegression:
    return joblib.load("../model/model.joblib")


def generate_stream_data() -> Generator[Tuple[DataFrame], None, None]:
    file = read_csv("../data/housing.csv")
    data = file.to_dask_array(lengths=True)
    data = data[:, 0:-1]

    indices = np.random.choice(len(data), size=10, replace=False)
    selected_rows = data[indices]

    for row in selected_rows:
        yield row.reshape(1, -1)


def process_stream_data(data) -> Tuple[np.ndarray, Any]:
    model = load_model()
    prediction = model.predict(data).compute()
    return data, prediction


def save_predictions(predictions) -> None:
    with open("pred.csv", 'a+') as f:
        for data, prediction in predictions:
            data_values = data.compute().flatten().tolist()
            f.write(f"{data_values},{prediction}\n")


if __name__ == '__main__':
    stream = Stream()
    predictions = []

    stream.map(process_stream_data).sink(lambda x: predictions.append(x))

    data_generator = generate_stream_data()

    for new_data in data_generator:
        stream.emit(new_data)
        time.sleep(1)

    save_predictions(predictions)
