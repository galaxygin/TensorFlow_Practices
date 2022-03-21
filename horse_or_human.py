import os
import random

import numpy as np
from keras_preprocessing import image as keras_image
from keras_preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from tensorflow import keras

class_names = ['Horse', 'Human']
target_size = (300, 300)
horse_files = os.listdir("datasets/validation-horse-or-human/horses")
human_files = os.listdir("datasets/validation-horse-or-human/humans")
total_train = len(os.listdir("datasets/horse-or-human/horses")) + len(os.listdir("datasets/horse-or-human/humans"))
total_val = len(horse_files) + len(human_files)
epochs = 50
batch_size = 16
BatchNormalization = keras.layers.BatchNormalization


class Model:
    def __init__(self):
        # Sample from Colab
        # self.model = keras.models.Sequential([
        #     # Note the input shape is the desired size of the image 300x300 with 3 bytes color
        #     # This is the first convolution
        #     keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)),
        #     keras.layers.MaxPooling2D(2, 2),
        #     # The second convolution
        #     keras.layers.Conv2D(32, (3, 3), activation='relu'),
        #     keras.layers.MaxPooling2D(2, 2),
        #     # The third convolution
        #     keras.layers.Conv2D(64, (3, 3), activation='relu'),
        #     keras.layers.MaxPooling2D(2, 2),
        #     # The fourth convolution
        #     keras.layers.Conv2D(64, (3, 3), activation='relu'),
        #     keras.layers.MaxPooling2D(2, 2),
        #     # The fifth convolution
        #     keras.layers.Conv2D(64, (3, 3), activation='relu'),
        #     keras.layers.MaxPooling2D(2, 2),
        #     # Flatten the results to feed into a DNN
        #     keras.layers.Flatten(),
        #     keras.layers.Dropout(0.5),
        #     # 512 neuron hidden layer
        #     keras.layers.Dense(512, activation='relu'),
        #     # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')
        #     keras.layers.Dense(1, activation='sigmoid')
        # ])
        self.model = keras.models.Sequential([
            # Note the input shape is the desired size of the image 150x150 with 3 bytes color
            # This is the first convolution
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(300, 300, 3)),
            keras.layers.MaxPooling2D(2, 2),
            # The second convolution
            keras.layers.Conv2D(32, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D(2, 2),
            # The third convolution
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D(2, 2),
            # The fourth convolution
            # keras.layers.Conv2D(64, (3, 3), activation='relu'),
            # keras.layers.MaxPooling2D(2, 2),
            # Flatten the results to feed into a DNN
            keras.layers.Flatten(),
            keras.layers.Dropout(0.5),
            # 64 neuron hidden layer
            keras.layers.Dense(64, activation='relu'),
            # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')
            keras.layers.Dense(1, activation='sigmoid')
        ])
        self.training_generator = None
        self.validation_generator = None

    def prepare_images(self):
        self.training_generator = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
        ).flow_from_directory(
            "datasets/horse-or-human/",
            target_size=target_size,
            class_mode="binary",
            shuffle=True,
            batch_size=batch_size)
        self.validation_generator = ImageDataGenerator(rescale=1. / 255) \
            .flow_from_directory(
            "datasets/validation-horse-or-human/",
            target_size=target_size,
            class_mode="binary",
            batch_size=batch_size)
        return self

    def train(self):
        self.model.compile(loss="binary_crossentropy", optimizer='rmsprop', metrics=['accuracy'])
        history = self.model.fit(self.training_generator, epochs=epochs, steps_per_epoch=total_train // batch_size,
                                 validation_data=self.validation_generator, validation_steps=total_val // batch_size)
        self.model.save("horse-human_model")
        epochs_range = range(epochs)
        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, history.history['accuracy'], 'r', label='Training accuracy')
        plt.plot(epochs_range, history.history['val_accuracy'], 'b', label='Validation accuracy')
        plt.legend(loc="lower right")
        plt.title("Training and Validation accuracy")
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, history.history['loss'], 'g', label='Training loss')
        plt.plot(epochs_range, history.history['val_loss'], 'y', label='validation_loss')
        plt.legend("upper right")
        plt.title('Training and validation loss')
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
            self.model = keras.models.load_model("horse-human_model")
        except OSError:
            print("Trained model not found")
            return
        return self.model

    def get_validation_path(self):
        if self.base_path:
            return self.base_path
        if not os.path.exists("datasets/validation-horse-or-human"):
            print("Validation dataset not found at /datasets/")
            return
        self.base_path = os.path.join("datasets/validation-horse-or-human")
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
            hh = random.randint(0, 1)
            index = random.randint(0, 100)
            path = None
            if hh == 0:
                path = self.base_path + '/horses/' + os.listdir(self.base_path + '/horses')[index]
                print("Actual: " + os.listdir(self.base_path + '/horses')[index])
            if hh == 1:
                path = self.base_path + '/humans/' + os.listdir(self.base_path + '/humans')[index]
                print("Actual: " + os.listdir(self.base_path + '/humans')[index])
            x = keras_image.img_to_array(keras_image.load_img(path, target_size=(300, 300)))
            x = np.expand_dims(x, axis=0)
            prediction = self.model.predict(np.vstack([x]))
            if prediction[0][0] == 0:
                print("Prediction: Horse")
            if prediction[0][0] == 1:
                print("Prediction: Human")

    def validate_with_all_test_images(self):
        if not self.get_model():
            return
        if not self.get_validation_path():
            return 
        print("Validating horses (" + str(len(horse_files)) + ")")
        horse_wrong_predictions = []
        for index in range(len(horse_files)):
            x = keras_image.img_to_array(keras_image.load_img(self.base_path + '/horses/' + horse_files[index], target_size=target_size))
            x = np.expand_dims(x, axis=0)
            prediction = self.model.predict(np.vstack([x]))
            if prediction[0][0] == 1:
                horse_wrong_predictions.append(human_files[index])
        print("Number of wrong predictions: " + str(len(horse_wrong_predictions)) + ", " + str(horse_wrong_predictions))
        print("Validating humans (" + str(len(human_files)) + ")")
        human_wrong_predictions = []
        for index in range(len(human_files)):
            x = keras_image.img_to_array(
                keras_image.load_img(self.base_path + '/humans/' + human_files[index], target_size=target_size))
            x = np.expand_dims(x, axis=0)
            prediction = self.model.predict(np.vstack([x]))
            if prediction[0][0] == 0:
                human_wrong_predictions.append(human_files[index])
        print("Number of wrong predictions: " + str(len(human_wrong_predictions)) + ", " + str(human_wrong_predictions))
        
        
def conduct_training():
    Model().prepare_images().summary().train()
