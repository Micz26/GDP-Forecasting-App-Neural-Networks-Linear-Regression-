import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt


class GdpNnModel:
    def __init__(self, df):
        self.df = df
        self.years = [x for x in range(1960, 2022)]
        self.model = None


    def create_model(self):
        X = pd.get_dummies(self.df.drop(['country', '2021'], axis=1))
        y = self.df['2021']
        X.reset_index(drop=True, inplace=True)
        y.reset_index(drop=True, inplace=True)

        X = X / 204000
        y = y / 204000
        X = X.astype(np.float32)
        y = y.astype(np.float32)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(units=32, activation='relu', input_shape=(X_train.shape[1],)))
        model.add(tf.keras.layers.Dense(units=64, activation='relu'))
        model.add(tf.keras.layers.Dense(units=64, activation='relu'))
        model.add(tf.keras.layers.Dense(units=1, activation='linear'))

        model.compile(loss='mean_squared_error', optimizer='adam')

        model.fit(X_train, y_train, epochs=120, batch_size=32, validation_split=0.1)
        self.model = model

    def make_predictions(self, country, k=1):
        countries = self.df['country'].tolist()
        try:
            x = countries.index(country)
        except ValueError:
            return
        model = self.model
        p_reshaped = np.empty((1, 0), dtype=np.float32)
        countries = self.df['country'].tolist()
        X = pd.get_dummies(self.df.drop(['country'], axis=1))
        GDPs = X.iloc[x, 1:].tolist()
        X = pd.get_dummies(X.drop(['1960'], axis=1))
        for n in range(k):
            if p_reshaped.size == 0:
                X.reset_index(drop=True, inplace=True)
                X = X.astype(np.float32)
                p = X.iloc[x, :]
                p = p.astype(np.float32)
                p_reshaped = p.values.reshape(1, -1)
            else:
                p_reshaped = p_reshaped[:, 1:]
                p_reshaped = np.append(p_reshaped, [[value]], axis=1)
            p_pred = model.predict(p_reshaped)
            value = p_pred[0, 0]
            GDPs.append(value)
        future_years = [z for z in range(2022, 2022 + k)]
        all_years = self.years + future_years
        plt.plot(all_years, GDPs)
        plt.title(countries[x])
        plt.xlabel('Year')
        plt.ylabel('GDP (per person)')
        plt.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
        plt.axvspan(2022, 2040, facecolor='red', alpha=0.2)
        return plt





