import os
import random

import keras.models
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.preprocessing import image as keras_image
from keras_preprocessing.image import ImageDataGenerator

rock_dir = os.path.join('datasets/rps/rock')
paper_dir = os.path.join('datasets/rps/paper')
scissors_dir = os.path.join('datasets/rps/scissors')
rock_files = os.listdir(rock_dir)
paper_files = os.listdir(paper_dir)
scissors_files = os.listdir(scissors_dir)


class Model:
    def __init__(self):
        self.model = self.model = tf.keras.models.Sequential([
            # Note the input shape is the desired size of the image 150x150 with 3 bytes color
            # This is the first convolution
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)),
            tf.keras.layers.MaxPooling2D(2, 2),
            # The second convolution
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            # The third convolution
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            # The fourth convolution
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            # Flatten the results to feed into a DNN
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),
            # 512 neuron hidden layer
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax')
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
            target_size=(150, 150),
            class_mode='categorical',
            batch_size=126)
        self.validation_generator = ImageDataGenerator(rescale=1. / 255) \
            .flow_from_directory(
            "datasets/validation-rps",
            target_size=(150, 150),
            class_mode='categorical',
            batch_size=126)
        return self

    def train(self):
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        history = self.model.fit(self.training_generator, epochs=25, steps_per_epoch=20,
                                 validation_data=self.validation_generator,
                                 verbose=1,
                                 validation_steps=3)
        self.model.save("rps_model")
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


def show_images():
    print('total training rock images:', len(os.listdir(rock_dir)))
    print('total training paper images:', len(os.listdir(paper_dir)))
    print('total training scissors images:', len(os.listdir(scissors_dir)))

    print(rock_files[:10])
    print(paper_files[:10])
    print(scissors_files[:10])
    pic_index = 2

    next_rock = [os.path.join(rock_dir, fname) for fname in rock_files[pic_index - 2:pic_index]]
    next_paper = [os.path.join(paper_dir, fname) for fname in paper_files[pic_index - 2:pic_index]]
    next_scissors = [os.path.join(scissors_dir, fname) for fname in scissors_files[pic_index - 2:pic_index]]
    for i, img_path in enumerate(next_rock + next_paper + next_scissors):
        # print(img_path)
        img = mpimg.imread(img_path)
        plt.imshow(img)
        plt.axis('Off')
        plt.show()


def conduct_training():
    Model().prepare_images().summary().train()


def predict():
    model = keras.models.load_model("rps_model")
    for i in range(10):
        rps = random.randint(0, 2)
        index = random.randint(0, 100)
        path = None
        if rps == 0:
            path = rock_dir + '/' + rock_files[index]
            print(rock_files[index])
        if rps == 1:
            path = paper_dir + '/' + paper_files[index]
            print(paper_files[index])
        if rps == 2:
            path = scissors_dir + '/' + scissors_files[index]
            print(scissors_files[index])
        img = keras_image.load_img(path, target_size=(150, 150))
        x = keras_image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        print(model.predict(images, batch_size=10))
