import numpy as np
from tensorflow import keras


class Model:
    def __init__(self):
        self.model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

    def train(self):
        self.model.compile(optimizer='sgd', loss='mean_squared_error')
        xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
        ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)
        self.model.fit(xs, ys, epochs=500)
        self.model.save("number-rule_model")
        return self

    def summary(self):
        print(self.model.summary())
        return self


class TrainedModel:
    def __init__(self):
        self.model = None

    def get_model(self):
        if self.model:
            return self.model
        try:
            self.model = keras.models.load_model("number-rule_model")
        except OSError:
            print("Trained model not found")
            return
        return self.model

    def predict(self):
        if not self.get_model():
            return
        print(
            "There is a array of numbers with rule like this.\nX: [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]\nY: [-3.0, -1.0, 1.0, 3.0, 5.0, 7.0]")
        x = input("Let's the model to guess the value of Y is the given X value is: ")
        print(self.model.predict([int(x)]))

    def summary(self):
        if not self.get_model():
            return
        print(self.model.summary())


def conduct_training():
    Model().summary().train()
