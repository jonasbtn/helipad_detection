import os
import numpy as np

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator


class BBTrainingManager:

    def __init__(self, image_folder):

        self.image_folder = image_folder

        self.datagen = ImageDataGenerator(rescale=1.0/255.0)

        self.train_it = self.datagen.flow_from_directory(os.path.join(self.image_folder, "train"),
                                                    class_mode='binary', batch_size=64, target_size=(32,32))
        self.test_it = self.datagen.flow_from_directory(os.path.join(self.image_folder, "test"),
                                                         class_mode='binary', batch_size=64, target_size=(32, 32))

        self.model = self.define_model()

    def define_model(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                         input_shape=(32, 32, 3)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(1, activation='sigmoid'))
        # compile model
        opt = SGD(lr=0.001, momentum=0.9)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def run(self):
        # fit model
        self.history = self.model.fit_generator(self.train_it, steps_per_epoch=len(self.train_it),
                                      validation_data=self.test_it, validation_steps=len(self.test_it),
                                      epochs=20, verbose=1)

    def evaluate(self):
        # evaluate model
        _, acc = self.model.evaluate_generator(self.test_it, steps=len(self.test_it), verbose=0)
        print('> %.3f' % (acc * 100.0))


if __name__ == "__main__":

    image_folder = "C:\\Users\\jonas\\Desktop\\Helipad\\Detected_Boxes_2\\model_7_0.0"

    bbtraining_manager = BBTrainingManager(image_folder)

    bbtraining_manager.run()

    bbtraining_manager.evaluate()
