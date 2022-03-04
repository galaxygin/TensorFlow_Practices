import keras.models
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
(training_images, training_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()


class Model:
    def __init__(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        self.training_images = training_images
        self.test_images = test_images

    def prepare_images(self):
        self.training_images = self.training_images.reshape(60000, 28, 28, 1)
        self.test_images = self.test_images.reshape(10000, 28, 28, 1)
        self.training_images, self.test_images = self.training_images / 255.0, self.test_images / 255.0
        return self

    def train(self):
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(training_images, training_labels, epochs=5)
        self.model.save("fashion_mnist_model")
        return self

    def summary(self):
        print(self.model.summary())
        return self


def conduct_training():
    Model().prepare_images().summary().train()


def predict():
    model = keras.models.load_model("fashion_mnist_model")
    prediction = model.predict(test_images)
    for i in range(len(class_names)-1):
        print('Actual: ' + class_names[test_labels[i]])
        print("Prediction: " + class_names[np.argmax(prediction[i])])
        print(prediction)
        # plt.grid(False)
        # plt.imshow(test_images[i], cmap=plt.cm.binary)
        # plt.xlabel('Actual: ' + class_names[test_labels[i]])
        # plt.title("Prediction: " + class_names[np.argmax(prediction[i])])
        # plt.show()

# f, axarr = plt.subplots(3, 4)
# FIRST_IMAGE = 0
# SECOND_IMAGE = 23
# THIRD_IMAGE = 28
# CONVOLUTION_NUMBER = 6
# layer_outputs = [layer.output for layer in model.layers]
# activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
# for x in range(1):
#     f1 = activation_model.predict(test_images[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]
#     print(class_names[np.argmax(f1[x])])
#     print(x)
#     print(f1)
#     axarr[0, x].imshow(f1[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
#     axarr[0, x].grid(False)
#     f2 = activation_model.predict(test_images[SECOND_IMAGE].reshape(1, 28, 28, 1))[x]
#     axarr[1, x].imshow(f2[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
#     axarr[1, x].grid(False)
#     f3 = activation_model.predict(test_images[THIRD_IMAGE].reshape(1, 28, 28, 1))[x]
#     axarr[2, x].imshow(f3[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
#     axarr[2, x].grid(False)
