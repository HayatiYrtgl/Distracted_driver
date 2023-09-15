from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping, ProgbarLogger
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.metrics import metrics


# class for Ai
class Distracted:
    def __init__(self):

        # const
        self.train_path = "archive (1)/distracted-driver-detection/train"
        self.val_path = "archive (1)/distracted-driver-detection/val"
        self.epochs = 60
        self.optimizer = "sgd"
        self.fin_opt = "softmax"

        # datasets
        self.train, self.val = self.image_generator()

        # callbacks
        self.callback_1, self.callback_2 = self.callbacks()

        # model
        self.model = self.model_create()

    # image dataset generator func
    def image_generator(self):

        # generators
        train_gen = ImageDataGenerator(rescale=1./255,
                                       vertical_flip=True,
                                       horizontal_flip=True,
                                       rotation_range=45,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       fill_mode="nearest",
                                       )

        val_gen = ImageDataGenerator(rescale=1./255,
                                       vertical_flip=True,
                                       horizontal_flip=True,
                                       rotation_range=45,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       fill_mode="nearest",
                                       )
        # create data
        train = train_gen.flow_from_directory(self.train_path, (224, 224),
                                              class_mode="categorical", shuffle=True, batch_size=64, color_mode="grayscale")

        val = val_gen.flow_from_directory(self.val_path, (224, 224),
                                              class_mode="categorical", shuffle=True, batch_size=32, color_mode="grayscale")

        return train, val

    # model callbacks
    @staticmethod
    def callbacks():

        # callbacks
        c1 = ProgbarLogger("samples", None)
        c2 = EarlyStopping(monitor="accuracy", patience=10, min_delta=0.003, verbose=1)

        return c1, c2

    # model creator
    def model_create(self):

        # model is sequential
        model = Sequential()

        # add conv1 layer
        model.add(Conv2D(128, (3, 3), (4, 4), padding="same", input_shape=(224, 224, 1)))
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(MaxPool2D((2, 2), (2, 2)))

        # add conv 2 layer
        model.add(Conv2D(256, (3, 3), padding="same"))
        model.add(Conv2D(256, (3, 3), padding="same"))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(MaxPool2D((2, 2), (2, 2)))

        # add conv 3 layer
        model.add(Conv2D(512, (3, 3), padding="same"))
        model.add(Conv2D(512, (3, 3), padding="same"))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(MaxPool2D((2, 2), (2, 2)))

        # add conv 4 layer
        model.add(Conv2D(256, (3, 3), padding="same"))
        model.add(Conv2D(256, (3, 3), padding="same"))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(MaxPool2D((2, 2), (2, 2)))

        # flatten
        model.add(Flatten())
        model.add(Dropout(0.3))
        # dense and dropout
        model.add(Dense(2048, activation="relu"))
        model.add(Dropout(0.3))

        # dense and dropout
        model.add(Dense(1024, activation="relu"))
        model.add(Dropout(0.3))
        # final dense
        model.add(Dense(3, activation=self.fin_opt))

        # model architecture
        model.summary()

        # model compile
        model.compile(loss="categorical_crossentropy", optimizer=self.optimizer, metrics=["accuracy",
                                                                                          metrics.categorical_accuracy,
                                                                                          metrics.Recall(),
                                                                                          metrics.Precision(),
                                                                                          metrics.AUC()])

        return model

    # fitter
    def model_fitter(self):

        # fitter
        history = self.model.fit(self.train, epochs=self.epochs, verbose=1,
                                 callbacks=[self.callback_2], shuffle=True, validation_data=self.val)

        self.model.save("distracted.h5")

        dataframe = pd.DataFrame(history.history).plot()

        plt.show()


# class
c = Distracted()
c.model_fitter()
