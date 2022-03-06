import numpy as np
from tensorflow import keras

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
        self.model.fit(train_images, train_labels, epochs=10)
        self.model.save("fashion-classify_model")
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
            self.model = keras.models.load_model("fashion-classify_model")
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
        print("Loss: "+test_loss)
        print("Accuracy: "+test_acc)

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
