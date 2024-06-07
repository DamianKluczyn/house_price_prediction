# scripts/data_generator.py

import pandas as pd
import numpy as np
import time
from faker import Faker

fake = Faker()

# Funkcja generująca pojedynczy rekord
def generate_record():
    return {
        "longitude": np.random.uniform(-124.35, -114.31),
        "latitude": np.random.uniform(32.54, 42.01),
        "housingMedianAge": np.random.randint(1, 52),
        "totalRooms": np.random.randint(2, 10),
        "totalBedrooms": np.random.randint(1, 5),
        "population": np.random.randint(1, 5000),
        "households": np.random.randint(1, 1000),
        "medianIncome": np.random.uniform(0.5, 15),
        "medianHouseValue": np.random.uniform(15000, 500000),
        "oceanProximity": fake.random_element(elements=("NEAR BAY", "<1H OCEAN", "INLAND", "NEAR OCEAN", "ISLAND"))
    }

# Generowanie danych strumieniowych
while True:
    records = [generate_record() for _ in range(100)]
    df = pd.DataFrame(records)
    df.to_csv('/opt/dask/data/streaming_data.csv', mode='a', header=False, index=False)
    print("Generated 100 records")
    time.sleep(5)
