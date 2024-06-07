import dask.dataframe as dd
from dask.distributed import Client
from dask_ml.preprocessing import StandardScaler, OneHotEncoder
import joblib
import time

# Inicjalizacja klienta Dask
client = Client("dask-scheduler:8786")

# Wczytanie zapisanego modelu
model = joblib.load("/opt/dask/models/house_price_model.pkl")


# Funkcja do przetwarzania danych strumieniowych
def process_streaming_data(df):
    df = df.categorize(columns=['ocean_proximity'])
    encoder = OneHotEncoder(sparse=False)
    scaler = StandardScaler()
    X = encoder.fit_transform(df.drop(columns='median_house_value'))
    X = scaler.fit_transform(X)
    predictions = model.predict(X)
    print(predictions.compute())


# Wczytanie strumieniowych danych z pliku CSV
while True:
    try:
        stream_df = dd.read_csv("/opt/dask/data/streaming_data.csv", assume_missing=True)
        stream_df = stream_df.dropna()
        stream_df.map_partitions(process_streaming_data).compute()
    except FileNotFoundError:
        print("No new data found. Waiting...")
    time.sleep(5)
