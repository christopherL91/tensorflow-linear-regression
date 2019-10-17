import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

tf.debugging.set_log_device_placement(True)

class Model(object):
    def __init__(self):
        self.__model__ = None
        self.__train_metadata__ = None

    def __get_model__(self):
        if self.__model__ == None:
            raise ValueError('You must train a model before using it')
        return self.__model__

    def train(self, df):
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=None)
        target_label = 'MPG'
        train_stats = train_df\
            .describe()\
            .pop(target_label)\
            .transpose()
        self.__train_metadata__ = {
            'mean': train_stats['mean'],
            'std': train_stats['std']
        }

        train_labels = train_df.pop(target_label)
        test_labels = test_df.pop(target_label)

        def norm(x):
            return (x - train_stats["mean"]) / train_stats["std"]
        normed_train_df = norm(train_df.pop(target_label))
        normed_test_df = norm(test_df.pop(target_label))

        "Define the network as a stack of layers"
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=[len(normed_train_df.keys())]),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ])

        optimizer = tf.keras.optimizers.RMSprop(0.001)
        model.compile(loss='mse',
            optimizer=optimizer,
            metrics=['mae', 'mse'])

        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        model.fit(
            normed_train_df, train_labels,
            epochs=1000, validation_split = 0.2, verbose=0,
            callbacks=[early_stop])

        loss, mae, mse = model.evaluate(normed_test_df, test_labels, verbose=2)
        print(f'loss: {loss} mae: {mae} mse: {mse}')
        self.__model__ = model

    def predict(self, df):
        model = self.__get_model__()
        normalized_df = (df - self.__train_metadata__['mean']) / self.__train_metadata__['std']
        return model.predict(normalized_df)

    def save(self, model_path):
        print('saving model to', model_path)
        self.__get_model__().save(f'{model_path}.h5')
        with open(f'{model_path}.json', 'w') as f:
            json.dump(self.__train_metadata__, f)

    def load(self, model_path):
        print('loading model from', model_path)
        self.__model__ = tf.keras.models.load_model(f'{model_path}.h5')
        with open(f'{model_path}.json', 'r') as f:
            self.__train_metadata__ = json.load(f)
