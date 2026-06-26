!pip install xgboost tensorflow scikit-learn pyarrow


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM


df = pd.read_csv('/content/demand_forecasting.csv')
print(df.shape)
df.head()


print(df.columns)
df.info()
df.isnull().sum()


df = df.dropna()


categorical_cols = [
    'Store ID',
    'Product ID',
    'Category',
    'Region',
    'Weather Condition',
    'Seasonality'
]

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))


X = df.drop('Demand', axis=1)
y = df['Demand']


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)



rf_mae = mean_absolute_error(y_test, rf_pred)

rf_rmse = np.sqrt(
    mean_squared_error(y_test, rf_pred)
)

rf_mape = np.mean(
    np.abs((y_test-rf_pred)/y_test)
)*100
rf_r2 = r2_score(y_test, rf_pred)
print("RF MAE:", rf_mae)
print("RF RMSE:", rf_rmse)
print("RF MAPE:", rf_mape)
print("RF R2:", rf_r2)





