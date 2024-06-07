# Użyj obrazu bazowego Dask
FROM daskdev/dask:latest

# Instalacja dodatkowych pakietów Pythona
RUN pip install pandas scikit-learn joblib dask-ml

# Kopiowanie skryptu do kontenera
COPY scripts/train_model.py /opt/dask/scripts/train_model.py

# Ustawienie punktu wejściowego
ENTRYPOINT ["python", "/opt/dask/scripts/train_model.py"]
