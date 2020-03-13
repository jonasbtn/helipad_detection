import os
import numpy as np
from matplotlib import pyplot
import sys

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator


class BBTrainingManager:

    def __init__(self, image_folder):

        self.image_folder = image_folder

        self.datagen = ImageDataGenerator(rescale=1.0/255.0)

        self.train_it = self.datagen.flow_from_directory(os.path.join(self.image_folder, "train"),
                                                    class_mode='binary', batch_size=128, target_size=(32,32))
        self.test_it = self.datagen.flow_from_directory(os.path.join(self.image_folder, "test"),
                                                         class_mode='binary', batch_size=128, target_size=(32, 32))

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
        # opt = Adam(lr=0.001)
        opt = SGD(lr=0.001, momentum=0.9)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def run(self):
        # fit model
        self.history = self.model.fit_generator(self.train_it, steps_per_epoch=len(self.train_it),
                                      validation_data=self.test_it, validation_steps=len(self.test_it),
                                      epochs=200, verbose=1)

    def evaluate(self):
        # evaluate model
        _, acc = self.model.evaluate_generator(self.test_it, steps=len(self.test_it), verbose=0)
        print('> %.3f' % (acc * 100.0))

    def plot(self):
        # plot diagnostic learning curves
        def summarize_diagnostics(history):
            # plot loss
            pyplot.subplot(211)
            pyplot.title('Cross Entropy Loss')
            pyplot.plot(history.history['loss'], color='blue', label='train')
            pyplot.plot(history.history['val_loss'], color='orange', label='test')
            # plot accuracy
            pyplot.subplot(212)
            pyplot.title('Classification Accuracy')
            if "acc" in history.history:
                pyplot.plot(history.history['acc'], color='blue', label='train')
            elif "accuracy" in history.history:
                pyplot.plot(history.history['accuracy'], color='blue', label='train')
            if "val_acc" in history.history:
                pyplot.plot(history.history['val_acc'], color='orange', label='test')
            elif "val_accuracy" in history.history:
                pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
            # save plot to file
            # filename = sys.argv[0].split('/')[-1]
            pyplot.savefig('plot_sgd.png')
            pyplot.close()

        summarize_diagnostics(self.history)

    def save(self):
        # save model
        self.model.save('final_model_sgd.h5')


if __name__ == "__main__":

    # image_folder = "C:\\Users\\jonas\\Desktop\\Helipad\\Detected_Boxes_2\\model_7_0.0"

    image_folder = "C:\\Users\\AISG\\Documents\\Jonas\\Detected_Boxes_3\\model_7_0.0"

    bbtraining_manager = BBTrainingManager(image_folder)

    bbtraining_manager.run()

    bbtraining_manager.evaluate()

    bbtraining_manager.save()

    bbtraining_manager.plot()
