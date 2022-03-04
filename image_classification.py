from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()


class Model:
    def __init__(self):
        self.model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])
        self.training_images = train_images
        self.test_images = test_images

    def prepare_images(self):
        self.training_images, self.test_images = self.training_images / 255.0, self.test_images / 255.0
        return self

    def train(self):
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(train_images, train_labels, epochs=3)
        self.model.save("fashion_classify_model")
        # test_loss, test_acc = model.evaluate(test_images, test_labels)
        return self

    def summary(self):
        print(self.model.summary())
        return self


def conduct_training():
    Model().prepare_images().summary().train()


def predict():
    model = keras.models.load_model("fashion_classify_model")
    prediction = model.predict(test_images)
    for i in range(2):
        print("Actual: "+class_names[test_labels[i]])
        print("Prediction: "+class_names[np.argmax(prediction[i])])
        print(prediction)
    # plt.grid(True)
    # plt.imshow(test_images[i], cmap=plt.cm.binary)
    # plt.xlabel('Actual: ' + class_names[test_labels[i]])
    # plt.title("Prediction: " + class_names[np.argmax(prediction[i])])
    # plt.show()
