import keras.models
import numpy as np
from tensorflow import keras

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
(training_images, training_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()


class Model:
    def __init__(self):
        self.model = keras.models.Sequential([
            keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
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
        self.model.fit(training_images, training_labels, epochs=10)
        self.model.save("fashion-mnist_model")
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
            self.model = keras.models.load_model("fashion-mnist_model")
        except OSError:
            print("Trained model not found")
            return
        return self.model

    def predict(self):
        self.validate_with_all_test_images()

    def summary(self):
        if not self.get_model():
            return
        print(self.model.summary())

    def evaluate(self):
        if not self.get_model():
            return
        test_loss, test_acc = self.model.evaluate(test_images, test_labels)
        print("Loss: "+str(test_loss))
        print("Accuracy: "+str(test_acc))

    def random_validate10(self):
        if not self.get_model():
            return
        prediction = self.model.predict(test_images)
        for i in range(len(class_names) - 1):
            print(test_images[i])
            print('Actual: ' + class_names[test_labels[i]])
            print("Prediction: " + class_names[np.argmax(prediction[i])])
            print(prediction)
            # plt.grid(False)
            # plt.imshow(test_images[i], cmap=plt.cm.binary)
            # plt.xlabel('Actual: ' + class_names[test_labels[i]])
            # plt.title("Prediction: " + class_names[np.argmax(prediction[i])])
            # plt.show()

    def validate_with_all_test_images(self):
        if not self.get_model():
            return
        print("Validating fashion images (" + str(len(test_images)) + ")")
        wrong_predictions = []
        prediction = self.model.predict(test_images)
        for i in range(len(test_images)):
            if class_names[test_labels[i]] != class_names[np.argmax(prediction[i])]:
                wrong_predictions.append(class_names[test_labels[i]])
        print("Number of wrong predictions: " + str(len(wrong_predictions)) + ", " + str(wrong_predictions))


def conduct_training():
    Model().prepare_images().summary().train()

# f, axarr = plt.subplots(3, 4)
# FIRST_IMAGE = 0
# SECOND_IMAGE = 23
# THIRD_IMAGE = 28
# CONVOLUTION_NUMBER = 6
# layer_outputs = [layer.output for layer in model.layers]
# activation_model = keras.models.Model(inputs=model.input, outputs=layer_outputs)
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
