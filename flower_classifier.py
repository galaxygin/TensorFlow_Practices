import os

import matplotlib.pyplot as plt
from tensorflow import keras
from keras_preprocessing import image as keras_image
from keras_preprocessing.image import ImageDataGenerator
import numpy as np

class_names = ["Daisy", "Dandelion", "Rose", "Sunflower", "Tulip"]
target_size = (300, 300)
daisy_files = os.listdir("datasets/validation-flowers/daisies")
dandelion_files = os.listdir("datasets/validation-flowers/dandelions")
rose_files = os.listdir('datasets/validation-flowers/roses')
sunflower_files = os.listdir("datasets/validation-flowers/sunflowers")
tulip_files = os.listdir("datasets/validation-flowers/tulips")
total_train = len(os.listdir("datasets/flowers/daisies")) + len(os.listdir("datasets/flowers/dandelions")) + len(os.listdir("datasets/flowers/roses")) + len(os.listdir("datasets/flowers/sunflowers")) + len(os.listdir("datasets/flowers/tulips"))
total_val = len(daisy_files) + len(dandelion_files) + len(rose_files) + len(sunflower_files) + len(tulip_files)
epochs = 50
batch_size = 16


class Model:
    def __init__(self):
        self.model = keras.models.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(300, 300, 3)),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Conv2D(32, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Flatten(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(len(class_names), activation='softmax')
        ])
        self.training_generator = None
        self.validation_generator = None

    def prepare_images(self):
        self.training_generator = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
        ).flow_from_directory(
            "datasets/flowers",
            target_size=target_size,
            class_mode="categorical",
            shuffle=True,
            batch_size=batch_size)
        self.validation_generator = ImageDataGenerator(rescale=1. / 255) \
            .flow_from_directory(
            "datasets/validation-flowers",
            target_size=target_size,
            class_mode="categorical",
            batch_size=batch_size)
        return self

    def train(self):
        self.model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
        history = self.model.fit(self.training_generator, epochs=epochs, steps_per_epoch=total_train // batch_size,
                                 validation_data=self.validation_generator, validation_steps=total_val // batch_size)
        self.model.save("flower_model")
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
            self.model = keras.models.load_model("flower_model")
        except OSError:
            print("Trained model not found")
            return
        return self.model

    def get_validation_path(self):
        if self.base_path:
            return self.base_path
        if not os.path.exists("datasets/validation-flowers"):
            print("Validation dataset not found at /datasets/")
            return
        self.base_path = os.path.join("datasets/validation-flowers")
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
        print("Validating Daisy ("+str(len(daisy_files))+")")
        daisy_wrong_predictions = []
        for index in range(len(daisy_files)):
            if daisy_files[index] != ".DS_Store":
                x = keras_image.img_to_array(keras_image.load_img(self.base_path + "/daisies/"+daisy_files[index], target_size=target_size))
                x = np.expand_dims(x, axis=0)
                prediction = self.model.predict(np.vstack([x]))
                if class_names[np.argmax(prediction)] != "Daisy":
                    daisy_wrong_predictions.append(daisy_files[index])
        print("Number of wrong predictions: "+str(len(daisy_wrong_predictions))+", "+str(daisy_wrong_predictions))
        print("Validating Dandelion (" + str(len(dandelion_files)) + ")")
        dandelion_wrong_predictions = []
        for index in range(len(dandelion_files)):
            if dandelion_files[index] != ".DS_Store":
                x = keras_image.img_to_array(keras_image.load_img(self.base_path + "/dandelions/" + dandelion_files[index], target_size=target_size))
                x = np.expand_dims(x, axis=0)
                prediction = self.model.predict(np.vstack([x]))
                if class_names[np.argmax(prediction)] != "Dandelion":
                    dandelion_wrong_predictions.append(dandelion_files[index])
        print("Number of wrong predictions: " + str(len(dandelion_wrong_predictions)) + ", " + str(dandelion_wrong_predictions))
        print("Validating Rose (" + str(len(rose_files)) + ")")
        rose_wrong_predictions = []
        for index in range(len(rose_files)):
            if rose_files[index] != ".DS_Store":
                x = keras_image.img_to_array(keras_image.load_img(self.base_path + "/roses/" + rose_files[index], target_size=target_size))
                x = np.expand_dims(x, axis=0)
                prediction = self.model.predict(np.vstack([x]))
                if class_names[np.argmax(prediction)] != "Rose":
                    rose_wrong_predictions.append(rose_files[index])
        print("Number of wrong predictions: " + str(len(rose_wrong_predictions)) + ", " + str(rose_wrong_predictions))
        print("Validating Sunflower (" + str(len(sunflower_files)) + ")")
        sunflower_wrong_predictions = []
        for index in range(len(sunflower_files)):
            if sunflower_files[index] != ".DS_Store":
                x = keras_image.img_to_array(keras_image.load_img(self.base_path + "/sunflowers/" + sunflower_files[index], target_size=target_size))
                x = np.expand_dims(x, axis=0)
                prediction = self.model.predict(np.vstack([x]))
                if class_names[np.argmax(prediction)] != "Sunflower":
                    sunflower_wrong_predictions.append(sunflower_files[index])
        print("Number of wrong predictions: " + str(len(sunflower_wrong_predictions)) + ", " + str(sunflower_wrong_predictions))
        print("Validating Tulip (" + str(len(tulip_files)) + ")")
        tulip_wrong_predictions = []
        for index in range(len(tulip_files)):
            if tulip_files[index] != ".DS_Store":
                x = keras_image.img_to_array(keras_image.load_img(self.base_path + "/tulips/" + tulip_files[index], target_size=target_size))
                x = np.expand_dims(x, axis=0)
                prediction = self.model.predict(np.vstack([x]))
                if class_names[np.argmax(prediction)] != "Tulip":
                    tulip_wrong_predictions.append(tulip_files[index])
        print("Number of wrong predictions: " + str(len(tulip_wrong_predictions)) + ", " + str(tulip_wrong_predictions))


def conduct_training():
    Model().prepare_images().summary().train()
