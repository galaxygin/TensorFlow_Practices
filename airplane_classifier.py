import os

import matplotlib.pyplot as plt
from tensorflow import keras
from keras_preprocessing import image as keras_image
from keras_preprocessing.image import ImageDataGenerator
import numpy as np

class_names = ["Airbus A380", "Boeing 747"]
target_size = (300, 300)
valA380_files = os.listdir("datasets/validation-Airplanes/Airbus A380")
valB747_files = os.listdir("datasets/validation-Airplanes/Boeing 747")
total_train = len(os.listdir("datasets/Airplanes/Airbus A380")) + len(os.listdir("datasets/Airplanes/Boeing 747"))
total_val = len(valA380_files) + len(valB747_files)
epochs = 50
batch_size = 16


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
            keras.layers.Flatten(),
            keras.layers.Dense(units=128, activation='relu'),  # input layers
            keras.layers.Dropout(0.5),
            keras.layers.Dense(units=256, activation='relu'),
            keras.layers.Dropout(0.5),
            # keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(len(class_names), activation='softmax')
        ])
        self.training_generator = None
        self.validation_generator = None

    def prepare_images(self):
        self.training_generator = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        ).flow_from_directory(
            "datasets/Airplanes",
            target_size=target_size,
            class_mode="categorical",
            shuffle=True,
            batch_size=batch_size)
        self.validation_generator = ImageDataGenerator(rescale=1./255)\
            .flow_from_directory(
            "datasets/validation-Airplanes",
            target_size=target_size,
            class_mode="categorical",
            batch_size=batch_size)
        return self

    def train(self):
        self.model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        history = self.model.fit(self.training_generator, epochs=epochs, steps_per_epoch=total_train // batch_size,
                                 validation_data=self.validation_generator, validation_steps=total_val // batch_size)
        self.model.save("airplane_model")
        epochs_range = range(epochs)
        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, history.history["accuracy"], label='Training Accuracy')
        plt.plot(epochs_range, history.history['val_accuracy'], label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, history.history['loss'], label='Training Loss')
        plt.plot(epochs_range, history.history['val_loss'], label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
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
            self.model = keras.models.load_model("airplane_model")
        except OSError:
            print("Trained model not found")
            return
        return self.model

    def get_validation_path(self):
        if self.base_path:
            return self.base_path
        if not os.path.exists("datasets/validation-Airplanes"):
            print("Validation dataset not found at /datasets/")
            return
        self.base_path = os.path.join("datasets/validation-Airplanes")
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
        print("Validating A380 ("+str(len(valA380_files))+")")
        A380_wrong_predictions = []
        for index in range(len(valA380_files)):
            if valA380_files[index] != ".DS_Store":
                x = keras_image.img_to_array(keras_image.load_img(self.base_path+"/Airbus A380/"+valA380_files[index], target_size=target_size))
                x = np.expand_dims(x, axis=0)
                prediction = self.model.predict(np.vstack([x]))
                if class_names[np.argmax(prediction)] != "Airbus A380":
                    A380_wrong_predictions.append(valA380_files[index])
        print("Wrong predictions: "+str(len(A380_wrong_predictions))+", "+str(A380_wrong_predictions))
        print("Validating 747 (" + str(len(valB747_files)) + ")")
        B747_wrong_predictions = []
        for index in range(len(valB747_files)):
            if valB747_files[index] != ".DS_Store":
                x = keras_image.img_to_array(keras_image.load_img(self.base_path + "/Boeing 747/" + valB747_files[index], target_size=target_size))
                x = np.expand_dims(x, axis=0)
                prediction = self.model.predict(np.vstack([x]))
                if class_names[np.argmax(prediction)] != "Boeing 747":
                    B747_wrong_predictions.append(valB747_files[index])
        print("Wrong predictions: " + str(len(B747_wrong_predictions)) + ", " + str(B747_wrong_predictions))


def conduct_training():
    Model().prepare_images().summary().train()
