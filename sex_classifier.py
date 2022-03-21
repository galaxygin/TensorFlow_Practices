import os

import matplotlib.pyplot as plt
from tensorflow import keras
from keras_preprocessing import image as keras_image
from keras_preprocessing.image import ImageDataGenerator
import numpy as np

class_names = ["Boobs", "Creampie", "Pussy"]
target_size = (300, 300)


class Model:
    def __init__(self):
        self.model = keras.models.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(300, 300, 3)),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Conv2D(128, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Conv2D(256, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Conv2D(512, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Flatten(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(3, activation='softmax')
        ])
        self.training_generator = None
        self.validation_generator = None

    def prepare_images(self):
        self.training_generator = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode="nearest"
        ).flow_from_directory(
            "datasets/sex",
            target_size=target_size,
            class_mode="categorical",
            batch_size=128)
        self.validation_generator = ImageDataGenerator(rescale=1./255)\
            .flow_from_directory(
            "datasets/validation-sex",
            target_size=target_size,
            class_mode="categorical",
            batch_size=32)
        return self

    def train(self):
        self.model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
        history = self.model.fit(self.training_generator, epochs=50, verbose=1,
                                 validation_data=self.validation_generator)
        self.model.save("sex_model")
        epochs = range(len(history.history["accuracy"]))
        plt.plot(epochs, history.history["accuracy"], 'r', label='Training accuracy')
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
            self.model = keras.models.load_model("sex_model")
        except OSError:
            print("Trained model not found")
            return
        return self.model

    def get_validation_path(self):
        if self.base_path:
            return self.base_path
        if not os.path.exists("datasets/validation-sex"):
            print("Validation dataset not found at /datasets/")
            return
        self.base_path = os.path.join("datasets/validation-sex")
        return self.base_path

    def predict(self):
        self.validate_with_all_test_images()

    def summary(self):
        if not self.get_model():
            return
        print(self.model.summary())

    def validate_with_all_test_images(self):
        if not self.get_model():
            return
        if not self.get_validation_path():
            return
        print("Validating boobs ("+str(len(os.listdir(self.base_path+"/boobs")))+")")
        boobs_wrong_predictions = []
        for index in range(len(os.listdir(self.base_path+"/boobs"))):
            if os.listdir(self.base_path + "/boobs")[index] != ".DS_Store":
                x = keras_image.img_to_array(
                    keras_image.load_img(self.base_path+"/boobs/"+os.listdir(self.base_path+"/boobs")[index],
                                         target_size=target_size))
                x = np.expand_dims(x, axis=0)
                prediction = self.model.predict(np.vstack([x]))
                if class_names[np.argmax(prediction)] != "Boobs":
                    boobs_wrong_predictions.append(os.listdir(self.base_path+"/boobs")[index])
        print("Number of wrong predictions: "+str(len(boobs_wrong_predictions))+", "+str(boobs_wrong_predictions))
        print("Validating creampie (" + str(len(os.listdir(self.base_path + "/creampie"))) + ")")
        creampie_wrong_predictions = []
        predicted_as_pussy = []
        for index in range(len(os.listdir(self.base_path + "/creampie"))):
            # print(os.listdir(self.base_path + "/creampie")[index])
            if os.listdir(self.base_path + "/creampie")[index] != ".DS_Store":
                x = keras_image.img_to_array(
                    keras_image.load_img(self.base_path + "/creampie/" + os.listdir(self.base_path + "/creampie")[index],
                                         target_size=target_size))
                x = np.expand_dims(x, axis=0)
                prediction = self.model.predict(np.vstack([x]))
                if class_names[np.argmax(prediction)] == "Pussy":
                    predicted_as_pussy.append(os.listdir(self.base_path + "/pussy")[index])
                if class_names[np.argmax(prediction)] != "Creampie":
                    creampie_wrong_predictions.append(os.listdir(self.base_path + "/creampie")[index])
        print("Number of wrong predictions: " + str(len(creampie_wrong_predictions)) + ", " + str(creampie_wrong_predictions))
        print("Predicted as pussy: " + str(len(predicted_as_pussy)) + ", " + str(predicted_as_pussy))
        print("Validating pussy (" + str(len(os.listdir(self.base_path + "/pussy"))) + ")")
        pussy_wrong_predictions = []
        for index in range(len(os.listdir(self.base_path + "/pussy"))):
            if os.listdir(self.base_path + "/pussy")[index] != ".DS_Store":
                x = keras_image.img_to_array(
                    keras_image.load_img(self.base_path + "/pussy/" + os.listdir(self.base_path + "/pussy")[index],
                                         target_size=target_size))
                x = np.expand_dims(x, axis=0)
                prediction = self.model.predict(np.vstack([x]))
                if class_names[np.argmax(prediction)] != "Pussy":
                    pussy_wrong_predictions.append(os.listdir(self.base_path + "/pussy")[index])
        print("Number of wrong predictions: " + str(len(pussy_wrong_predictions)) + ", " + str(pussy_wrong_predictions))


def conduct_training():
    Model().prepare_images().summary().train()
