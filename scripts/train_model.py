import dask.dataframe as dd
from dask_ml.model_selection import train_test_split
from dask_ml.preprocessing import StandardScaler, OneHotEncoder
from dask_ml.linear_model import LinearRegression
from dask_ml.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
import joblib

# Wczytanie danych
df = dd.read_csv('/opt/dask/data/housing.csv')

# Przetwarzanie danych
df = df.categorize(columns=['ocean_proximity'])
df = df.dropna()

# Podział na cechy i etykiety
X = df.drop(columns='median_house_value')
y = df['median_house_value']

# Kodowanie cech kategorycznych
encoder = OneHotEncoder()
X = encoder.fit_transform(X)

# Skalowanie cech
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Podział na zbiory treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Trening modelu
model = LinearRegression()
model.fit(X_train, y_train)

# Ewaluacja modelu
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test.compute(), y_pred.compute())

print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"R2: {r2}")

# Zapisanie modelu
joblib.dump(model, '/opt/dask/models/house_price_model.pkl')
