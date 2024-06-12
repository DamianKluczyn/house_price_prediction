import numpy as np
import time
import warnings
import joblib
from typing import Generator, Tuple, Any
from streamz import Stream
from dask.dataframe import read_csv, DataFrame

warnings.filterwarnings("ignore", category=FutureWarning)


# Funkcja do generowania strumienia danych
def generate_stream_data() -> Generator[Tuple[DataFrame], None, None]:
    data = read_csv("./data/housing.csv").to_dask_array(lengths=True)
    data = data[:, 0:-1]

    indices = np.random.choice(len(data), size=20, replace=False)
    selected_rows = data[indices]

    for row in selected_rows:
        yield row.reshape(1, -1)


# Funkcja do procesowania danych strumieniowych
def process_stream_data(data) -> Tuple[np.ndarray, Any]:
    model = joblib.load("./data/model.joblib")
    prediction = model.predict(data).compute()
    return data, prediction


# Funkcja do zapisu predykcji
def save_predictions(predictions):
    with open("./data/pred.csv", 'a+') as f:
        for data, prediction in predictions:
            data_values = data.compute().flatten().tolist()
            f.write(f"{data_values},{prediction}\n")


if __name__ == '__main__':
    while True:
        try:
            stream = Stream()
            predictions = []

            stream.map(process_stream_data).sink(lambda x: predictions.append(x))

            data_generator = generate_stream_data()

            for new_data in data_generator:
                stream.emit(new_data)
                time.sleep(1)

            print("Saving predictions")
            save_predictions(predictions)

            time.sleep(1)
        except:
            print("Model nie został załadowany, czekam 30 sekund")
            time.sleep(30)
