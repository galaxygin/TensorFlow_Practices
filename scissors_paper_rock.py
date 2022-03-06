import os
import random

import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image as keras_image
from keras_preprocessing.image import ImageDataGenerator
from tensorflow import keras

class_names = ['Paper', 'Rock', 'Scissors']


class Model:
    def __init__(self):
        self.model = keras.models.Sequential([
            # Note the input shape is the desired size of the image 300x300 with 3 bytes color
            # This is the first convolution
            keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(300, 300, 3)),
            keras.layers.MaxPooling2D(2, 2),
            # The second convolution
            keras.layers.Conv2D(128, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D(2, 2),
            # The third convolution
            keras.layers.Conv2D(256, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D(2, 2),
            # The fourth convolution
            keras.layers.Conv2D(512, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D(2, 2),
            # Flatten the results to feed into a DNN
            keras.layers.Flatten(),
            keras.layers.Dropout(0.5),
            # 64 neuron hidden layer
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(3, activation='softmax')
        ])
        self.training_generator = None
        self.validation_generator = None

    def prepare_images(self):
        self.training_generator = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        ).flow_from_directory(
            "datasets/rps/",
            target_size=(300, 300),
            class_mode='categorical',
            batch_size=128)
        self.validation_generator = ImageDataGenerator(rescale=1. / 255) \
            .flow_from_directory(
            "datasets/validation-rps",
            target_size=(300, 300),
            class_mode='categorical',
            batch_size=32)
        return self

    def train(self):
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        history = self.model.fit(self.training_generator, epochs=100, verbose=1,
                                 validation_data=self.validation_generator)
        self.model.save("rps_model")
        epochs = range(len(history.history['accuracy']))
        plt.plot(epochs, history.history['accuracy'], 'r', label='Training accuracy')
        plt.plot(epochs, history.history['val_accuracy'], 'b', label='Validation accuracy')
        plt.plot(epochs, history.history['loss'], 'g', label='Training loss')
        plt.plot(epochs, history.history['val_loss'], 'y', label='validation_loss')
        plt.title('Training and validation accuracy')
        plt.legend(loc=0)
        plt.figure()
        plt.show()
        return self

    def summary(self):
        print(self.model.summary())
        return self


class TrainedModel:
    def __init__(self):
        self.model = None
        self.base_path = None
        
    def get_model(self):
        if self.model:
            return self.model
        try:
            self.model = keras.models.load_model("Trained_models/rps_model")
        except OSError:
            print("Trained model not found")
            return
        return self.model
    
    def get_validation_path(self):
        if self.base_path:
            return self.base_path
        if not os.path.exists("datasets/validation-rps"):
            print("Validation dataset not found at /datasets/")
            return
        self.base_path = os.path.join("datasets/validation-rps")
        return self.base_path
    
    def predict(self):
        self.validate_with_all_test_images()

    def summary(self):
        if not self.get_model():
            return
        print(self.model.summary())

    def random_validate10(self):
        if not self.get_model():
            return
        if not self.get_validation_path():
            return 
        for i in range(10):
            rps = random.randint(0, 2)
            index = random.randint(0, 100)
            path = None
            if rps == 0:
                path = self.base_path + '/paper/' + os.listdir(self.base_path + '/paper')[index]
                print("Actual: " + os.listdir(self.base_path + '/paper')[index])
            if rps == 1:
                path = self.base_path + '/rock/' + os.listdir(self.base_path + '/rock')[index]
                print("Actual: " + os.listdir(self.base_path + '/rock')[index])
            if rps == 2:
                path = self.base_path + '/scissors/' + os.listdir(self.base_path + '/scissors')[index]
                print("Actual: " + os.listdir(self.base_path + '/scissors')[index])
            x = keras_image.img_to_array(keras_image.load_img(path, target_size=(300, 300)))
            x = np.expand_dims(x, axis=0)
            prediction = self.model.predict(np.vstack([x]))
            print("Prediction: " + class_names[np.argmax(prediction)])
            print(self.model.predict(np.vstack([x])))

    def validate_with_all_test_images(self):
        if not self.get_model():
            return
        if not self.get_validation_path():
            return 
        print("Validating papers (" + str(len(os.listdir(self.base_path + "/paper"))) + ")")
        paper_wrong_predictions = []
        for index in range(len(os.listdir(self.base_path + "/paper"))):
            x = keras_image.img_to_array(
                keras_image.load_img(self.base_path + '/paper/' + os.listdir(self.base_path + '/paper')[index],
                                     target_size=(300, 300)))
            x = np.expand_dims(x, axis=0)
            if class_names[np.argmax(self.model.predict(np.vstack([x])))] != "Paper":
                paper_wrong_predictions.append(os.listdir(self.base_path + '/paper')[index])
        print("Number of wrong predictions: " + str(len(paper_wrong_predictions)) + ", " + str(paper_wrong_predictions))
        print("Validating rocks (" + str(len(os.listdir(self.base_path + "/rock"))) + ")")
        rock_wrong_predictions = []
        for index in range(len(os.listdir(self.base_path + "/rock"))):
            x = keras_image.img_to_array(
                keras_image.load_img(self.base_path + '/rock/' + os.listdir(self.base_path + '/rock')[index],
                                     target_size=(300, 300)))
            x = np.expand_dims(x, axis=0)
            if class_names[np.argmax(self.model.predict(np.vstack([x])))] != "Rock":
                rock_wrong_predictions.append(os.listdir(self.base_path + '/rock')[index])
        print("Number of wrong predictions: " + str(len(rock_wrong_predictions)) + ", " + str(rock_wrong_predictions))
        print("Validating scissors (" + str(len(os.listdir(self.base_path + "/scissors"))) + ")")
        scissors_wrong_predictions = []
        for index in range(len(os.listdir(self.base_path + "/scissors"))):
            x = keras_image.img_to_array(
                keras_image.load_img(self.base_path + '/scissors/' + os.listdir(self.base_path + '/scissors')[index],
                                     target_size=(300, 300)))
            x = np.expand_dims(x, axis=0)
            if class_names[np.argmax(self.model.predict(np.vstack([x])))] != "Scissors":
                scissors_wrong_predictions.append(os.listdir(self.base_path + '/scissors')[index])
        print("Number of wrong predictions: " + str(len(scissors_wrong_predictions)) + ", " + str(
            scissors_wrong_predictions))

    def evaluate(self):
        if not self.get_model():
            return
        if not self.get_validation_path():
            return 
        print("Validating papers (" + str(len(os.listdir(self.base_path + "/paper"))) + ")")
        for index in range(len(os.listdir(self.base_path + "/paper"))):
            x = keras_image.img_to_array(
                keras_image.load_img(self.base_path + '/paper/' + os.listdir(self.base_path + '/paper')[index],
                                     target_size=(300, 300)))
            x = np.expand_dims(x, axis=0)
            print(self.model.evaluate(np.vstack([x])))


def conduct_training():
    Model().prepare_images().summary().train()
