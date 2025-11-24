import numpy as np
from keras import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split

from Sub_Functions.Evaluate import main_est_parameters
import os

from Sub_Functions.Load_data import train_test_splitter, train_test_splitter1


def Deep_CNN(x_train, x_test, y_train, y_test, epochs):
    classes = len(np.unique(y_train))

    # Reshape: (N, 13) -> (N, 13, 1)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    model = Sequential()

    # ----------- Deep CNN Block 1 -----------
    model.add(Conv1D(64, kernel_size=3, activation="relu", padding="same", input_shape=(13, 1)))
    model.add(BatchNormalization())
    model.add(Conv1D(64, kernel_size=3, activation="relu", padding="same"))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))

    # ----------- Deep CNN Block 2 -----------
    model.add(Conv1D(128, kernel_size=3, activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(Conv1D(128, kernel_size=3, activation="relu", padding="same"))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))

    # ----------- Deep CNN Block 3 -----------
    model.add(Conv1D(256, kernel_size=3, activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(Conv1D(256, kernel_size=3, activation="relu", padding="same"))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.4))

    # ----------- Dense Layers -----------
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.4))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(classes, activation="softmax"))

    # Compile
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    # Train
    model.fit(x_train, y_train, epochs=epochs,
              validation_data=(x_test, y_test),
              verbose=1)

    # Predict
    pred = model.predict(x_test)
    pred = np.argmax(pred, axis=1)

    # Evaluate
    metrics = main_est_parameters(y_test, pred)

    # ----------- SAVE MODEL -------------
    save_path = "Saved_model"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model.save(os.path.join(save_path, "DB1.h5"))

    return metrics


feat = np.load("data_loader/DB2/Features.npy")
labels = np.load("data_loader/DB2/Labels.npy")
x_train, x_test, y_train, y_test = train_test_splitter1("DB2", 0.6)
metrics = Deep_CNN(x_train, x_test, y_train, y_test, epochs=100)
print(metrics)


