import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Input, GlobalMaxPooling1D
from keras.src.layers import Lambda
from keras import Model
from keras.src.optimizers import Adam
from Sub_Functions.Evaluate import main_est_parameters
from transformer import PositionalEmbedding
from transformer import Encoder
from transformer import Decoder, MultiheadAttention
import tensorflow as tf
from QCNN import QCNN, n_qubits

import tensorflow as tf
from tensorflow.keras import layers


class CBAMBLOCK(tf.keras.layers.Layer):
    def __init__(self, ratio=8):
        super(CBAMBLOCK, self).__init__()
        self.ratio = ratio

    def build(self, input_shape):
        channel = input_shape[-1]

        # channel attention
        self.shared_dense_one = Dense(channel // self.ratio, activation='relu')
        self.shared_dense_two = Dense(channel)

        # spatial attention
        self.conv1 = Conv1D(1, kernel_size=7, padding='same', activation='sigmoid')

    def call(self, x):
        # -------- Channel Attention --------
        avg_pool = tf.reduce_mean(x, axis=1, keepdims=True)
        max_pool = tf.reduce_max(x, axis=1, keepdims=True)

        avg_out = self.shared_dense_two(self.shared_dense_one(avg_pool))
        max_out = self.shared_dense_two(self.shared_dense_one(max_pool))

        channel_att = tf.nn.sigmoid(avg_out + max_out)
        x = x * channel_att

        # -------- Spatial Attention --------
        avg_pool_spatial = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_pool_spatial = tf.reduce_max(x, axis=-1, keepdims=True)
        spatial = tf.concat([avg_pool_spatial, max_pool_spatial], axis=-1)

        spatial_att = self.conv1(spatial)
        x = x * spatial_att

        return x


def proposed_model(x_train, x_test, y_train, y_test, Training_percentage, DB, embedding_size=64):
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    num_classes = len(np.unique(x_train))
    input = Input(shape=(13, 1))

    x = Conv1D(64, 3, padding="same", activation="relu")(input)
    # now shape = (Batch, 13, 64)

    x = CBAMBLOCK()(x)

    pos = PositionalEmbedding(words=13, embedding_size=64)

    out = pos(x)

    encoder = Encoder(d_k=8, attention_function=MultiheadAttention, model_embedding=embedding_size)

    attention_out, ff_out = encoder(out)

    concat = tf.keras.layers.Concatenate(axis=-1)([attention_out, ff_out])

    pooled = GlobalMaxPooling1D()(concat)

    q_input = Dense(n_qubits, activation='tanh')(pooled)
    q_input = Lambda(lambda x: x * np.pi)(q_input)

    qcnn = QCNN(num_filters=2, filter_size=4, num_params=1)
    output = qcnn(q_input)

    output = Dense(num_classes, activation="softmax")(output)

    model =Model(input= input,output=output)

    model.compile(optimizer=Adam(learning_rate=0.01),loss="sparse_categorical_crossentropy",metrics=["Accuracy"])

    model.fit(x_train,x_test,batch_size=32,epochs=2,validation_split=0.2)

    y_pred = model.predict(x_test)

    metrics = main_est_parameters(x_test,y_pred)

    return metrics








