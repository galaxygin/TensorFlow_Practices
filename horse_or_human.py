import os
import random

import keras.models
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras_preprocessing import image as keras_image
from keras_preprocessing.image import ImageDataGenerator

horse_dir = os.path.join('datasets/horse-or-human/horses')
human_dir = os.path.join('datasets/horse-or-human/humans')
horse_files = os.listdir(horse_dir)
human_files = os.listdir(human_dir)


class Model:
    def __init__(self):
        self.model = self.model = tf.keras.models.Sequential([
            # Note the input shape is the desired size of the image 300x300 with 3 bytes color
            # This is the first convolution
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)),
            tf.keras.layers.MaxPooling2D(2, 2),
            # The second convolution
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            # The third convolution
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            # The fourth convolution
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            # The fifth convolution
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            # Flatten the results to feed into a DNN
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),
            # 512 neuron hidden layer
            tf.keras.layers.Dense(512, activation='relu'),
            # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        # self.model = tf.keras.models.Sequential([
        #     # Note the input shape is the desired size of the image 150x150 with 3 bytes color
        #     # This is the first convolution
        #     tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        #     tf.keras.layers.MaxPooling2D(2, 2),
        #     # The second convolution
        #     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        #     tf.keras.layers.MaxPooling2D(2, 2),
        #     # The third convolution
        #     tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        #     tf.keras.layers.MaxPooling2D(2, 2),
        #     # The fourth convolution
        #     tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        #     tf.keras.layers.MaxPooling2D(2, 2),
        #     # Flatten the results to feed into a DNN
        #     tf.keras.layers.Flatten(),
        #     tf.keras.layers.Dropout(0.5),
        #     # 512 neuron hidden layer
        #     tf.keras.layers.Dense(512, activation='relu'),
        #     tf.keras.layers.Dense(1, activation='softmax')
        # ])
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
            fill_mode="nearest"
        ).flow_from_directory(
            "datasets/horse-or-human/",
            target_size=(300, 300),
            class_mode="binary",
            batch_size=128)
        self.validation_generator = ImageDataGenerator(rescale=1. / 255) \
            .flow_from_directory("datasets/validation-horse-or-human/",
                                 target_size=(300, 300),
                                 class_mode="binary",
                                 batch_size=32)
        return self

    def train(self):
        self.model.compile(loss="categorical_crossentropy", optimizer='rmsprop', metrics=['accuracy'])
        history = self.model.fit(self.training_generator, epochs=2, verbose=1,
                                 validation_data=self.validation_generator)
        self.model.save("horse-human_model")
        # acc = history.history['accuracy']
        # val_acc = history.history['val_accuracy']
        # loss = history.history['loss']
        # val_loss = history.history['val_loss']
        # epochs = range(len(acc))
        #
        # plt.plot(epochs, acc, 'r', label='Training accuracy')
        # plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
        # plt.plot(epochs, loss, 'g', label='Training loss')
        # plt.plot(epochs, val_loss, 'y', label='validation_loss')
        # plt.title('Training and validation accuracy')
        # plt.legend(loc=0)
        # plt.figure()
        # plt.show()
        return self

    def summary(self):
        print(self.model.summary())
        return self


def conduct_training():
    Model().prepare_images().summary().train()


def predict():
    model = keras.models.load_model("horse-human_model")
    for i in range(10):
        hh = random.randint(0, 1)
        index = random.randint(0, 100)
        path = None
        if hh == 0:
            path = horse_dir + '/' + horse_files[index]
            print(horse_files[index])
        if hh == 1:
            path = human_dir + '/' + human_files[index]
            print(human_files[index])
        img = keras_image.load_img(path, target_size=(300, 300))
        x = keras_image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        print(model.predict(images, batch_size=10))
