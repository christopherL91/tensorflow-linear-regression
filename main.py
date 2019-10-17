#!/usr/bin/env python3

from model import Model
import fire

class Cli(object):
    def train(self, model_path):
        """Train a model"""
        m = Model()
        m.save(model_path)

    def predict(self, model_path):
        """Use a model to predict"""
        m = Model()
        m.load(model_path)

    def version(self):
        import tensorflow as tf
        print('Running tensorflow version:', tf.__version__)
        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
        print("Num CPUs Available: ", len(tf.config.experimental.list_physical_devices('CPU')))
def main():
    fire.Fire(Cli())

if __name__ == '__main__':
    main()
