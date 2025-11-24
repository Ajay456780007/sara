from keras.src.layers import Flatten
from tensorflow.keras.layers import Dense, Lambda, Concatenate
import numpy as np
import tensorflow as tf
import pennylane as qml
# your QCNN code
filter_size = 2
n_qubits = filter_size ** 2   # = 4

dev = qml.device("lightning.qubit", wires=n_qubits, shots=None)

def encoding(pixel):
    [qml.RY(pixel[i], wires=i) for i in range(n_qubits)]

def circuit_1(params):
    qml.BasicEntanglerLayers(weights=params, wires=range(n_qubits))

@qml.qnode(dev)
def q_filter(inputs, params):
    encoding(inputs)
    circuit_1(params)
    return qml.expval(qml.PauliZ(0))  # output is 1 value

weight_shapes = {"params": (1, n_qubits)}
qlayer = qml.qnn.KerasLayer(q_filter, weight_shapes, output_dim=1)


class QCNN(tf.keras.Model):

    def __init__(self, num_filters, filter_size, num_params):
        super(QCNN, self).__init__()
        self.num_filters = num_filters
        self.filter_size = filter_size
        # using two filter now
        self.quantum_filter_1 = qml.qnn.KerasLayer(q_filter, weight_shapes, output_dim=1, name='quantum_filter')
        self.quantum_filter_2 = qml.qnn.KerasLayer(q_filter, weight_shapes, output_dim=1, name='quantum_filter')
        self.hidden = Dense(64, activation='relu')
        self.flatten = Flatten()
        self.dense = Dense(10, activation='softmax')

    def call(self, inputs):
        # The output length of one sample after the convolution operation with stride = 1
        output_length = inputs.shape[1] - self.filter_size + 1
        num_sample = inputs.shape[0]
        output_all = []
        quantum_filter_list = [self.quantum_filter_1, self.quantum_filter_2]

        outputs = []

        for b in range(num_sample):
            sample = inputs[b]  # shape: (features,)
            row_output = []

            # slide 1D window: i:i+filter_size
            for i in range(output_length):
                sub_input = sample[i: i + self.filter_size]  # shape (filter_size,)

                # apply each quantum filter
                filter_outputs = []
                for q_filter_layer in self.quantum_filters:
                    filter_outputs.append(q_filter_layer(sub_input))

                # combine filters into a vector
                row_output.append(tf.stack(filter_outputs))

            outputs.append(tf.stack(row_output))  # (output_length, num_filters)

        outputs = tf.stack(outputs)

        x = self.flatten(output_all)
        x = self.dense(x)
        return x