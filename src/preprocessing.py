import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os

def load_and_preprocess_data(csv_path='data/raw/weather.csv'):
    df = pd.read_csv(csv_path)

    df.columns = ['max_temp', 'min_temp', 'wind_speed', 'wind_dir',
                  'rainfall', 'humidity', 'cloud', 'pressure', 'date']

    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    df.dropna(inplace=True)

    df['rain_label'] = df['rainfall'].apply(lambda x: 1 if x > 0 else 0)

    df = pd.get_dummies(df, columns=['wind_dir'], drop_first=True)

    feature_cols = [col for col in df.columns if col not in ['rainfall', 'rain_label', 'date']]
    X = df[feature_cols]
    y = df['rain_label']

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test
