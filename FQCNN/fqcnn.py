import tensorflow as tf
import tensorflow_quantum as tfq
from tensorflow.keras.layers import Lambda
import sympy

import cirq
from cirq.contrib.svg import SVGCircuit
from cirq.circuits.qasm_output import QasmUGate

import scipy.io
import os
import numpy as np
from PIL import Image
import random
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


inputSize = 32
lr = 0.003
batch_size = 50
setting = 2

rawdata = scipy.io.loadmat('data_5fold_5classes.mat')

data = rawdata['setting'+str(setting)]
train_x = data['train_x'][0][0]
train_y = data['train_y'][0][0][0]

test_x = data['test_x'][0][0]
test_y = data['test_y'][0][0][0]


print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)


def resize_img(image, size):
    image = (255*(image - np.amin(image)) / (np.amax(image)-np.amin(image)))
    image = image.astype(np.uint8)
    img = Image.fromarray(image)
    img.thumbnail((size, size), Image.LANCZOS)
    return np.asarray(img)

def normalize(img):
    return (img - np.min(img))/(np.max(img)-np.min(img))


def iqr(image):
    for i in range(image.shape[2]):
        boundry1, boundry2 = np.percentile(image[:,:,i], [2 ,98])
        image[:,:,i] = np.clip(image[:,:,i], boundry1, boundry2)
    return image


def data_process(patches, labels):
    processed_img = []
    for (patch, name) in patches:
        temp = np.zeros((64, 4))
        img = iqr(patch)
        img = resize_img(img, 8)
        img = normalize(img)
        img = img*np.pi/2
        temp[:, 0] = img[:, :, 0].flatten()
        temp[:, 1] = img[:, :, 1].flatten()
        temp[:, 2] = img[:, :, 2].flatten()
        temp[:, 3] = img[:, :, 3].flatten()
        processed_img.append(temp)
    (unique, counts) = np.unique(labels, return_counts=True)
    print(len(labels), unique, counts)
    
    processed_img = np.array(processed_img)
    processed_label = LabelBinarizer().fit_transform(labels)
    
    return processed_img, processed_label


train_x, train_y = data_process(train_x, train_y)
test_x, test_y = data_process(test_x, test_y)

print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)


# Circuit: Image Encoding Circuit for FQCNN model
def friq_representor(symbols, qubits):
    circuit = cirq.Circuit()
    loc = qubits[:6]
    target = qubits[6]
    circuit.append(cirq.H.on_each(loc))
    for i in range(8):
        for j in range(8):
            if symbols[8 * i + j] != 0:
                row = [int(binary) for binary in format(i, '03b')]
                column = [int(binary) for binary in format(j, '03b')]
                ctrl_state = row + column
                circuit.append(cirq.ry(2 * symbols[8 * i + j]).on(target).controlled_by(loc[5], loc[4], loc[3], loc[2], loc[1], loc[0], control_values=ctrl_state))
    return circuit


# Circuit: Controlled U3 gate for FQCNN model
def cu3(circ, theta, phi, lam, target, cs, ctrl_state):
    circ.append(cirq.rz(lam).on(target).controlled_by(cs[0], cs[1], cs[2], cs[3], control_values=ctrl_state))
    circ.append(cirq.rx(np.pi / 2).on(target).controlled_by(cs[0], cs[1], cs[2], cs[3], control_values=ctrl_state))
    circ.append(cirq.rz(theta).on(target).controlled_by(cs[0], cs[1], cs[2], cs[3], control_values=ctrl_state))
    circ.append(cirq.rx(-np.pi / 2).on(target).controlled_by(cs[0], cs[1], cs[2], cs[3], control_values=ctrl_state))
    circ.append(cirq.rz(phi).on(target).controlled_by(cs[0], cs[1], cs[2], cs[3], control_values=ctrl_state))
    return circ


# Circuit: Quantum Convolution Layer for FQCNN model
def kernel_prepare(circ, xloc, yloc, target, kernel, readout, symbols0, symbols1, symbols2, ctrl):
    loc_states = [[0, 0], [1, 0], [0, 1], [1, 1]]
    for i, loc_state in enumerate(loc_states):
        ctrl_state = loc_state + [1] + ctrl
        circ = cu3(circ, symbols0[i], symbols1[i], symbols2[i], readout, [yloc[0], xloc[0], target, kernel[0]],
                   ctrl_state)

    return circ


# Circuit: Quantum Convolution Layer for FQCNN model
def conv_layer(circ, xloc, yloc, target, kernel, readout, symbols0, symbols1, symbols2):
    ctrls = []
    if len(kernel) > 0:
        for i in range(2 ** len(kernel)):
            states = [int(binary) for binary in format(i, '0' + str(len(kernel)) + 'b')]
            ctrls.append(states)
    else:
        ctrls.append([])
    for i, ctrl in enumerate(ctrls):
        circ = kernel_prepare(circ, xloc, yloc, target, kernel, readout, symbols0[4 * i:4 * i + 4],
                              symbols1[4 * i:4 * i + 4], symbols2[4 * i:4 * i + 4], ctrl)
    return circ


# circuit: FQCNN model
def qdcnn(qubits):
    circ = cirq.Circuit()

    symbols0 = sympy.symbols('a:16')
    symbols1 = sympy.symbols('b:16')
    symbols2 = sympy.symbols('c:16')

    color = qubits[6]
    kernel = qubits[7]
    readout = qubits[8:10]

    circ.append(cirq.H.on_each(kernel))

    circ = conv_layer(circ, qubits[3:6], qubits[:3], color, [kernel], readout[0], symbols0[0:8], symbols1[0:8],
                      symbols2[0:8])
    circ = conv_layer(circ, qubits[4:6], qubits[1:3], readout[0], [kernel], readout[1], symbols0[8:16], symbols1[8:16],
                      symbols2[8:16])

    return circ


class Encoding(tf.keras.layers.Layer):
    def __init__(self, qubits):
        super(Encoding, self).__init__()
        self.qubits = qubits

    def build(self, input_shape):
        self.symbols = sympy.symbols('img:64')
        self.circuit = friq_representor(self.symbols, self.qubits)

    def call(self, inputs):
        circuit = tfq.convert_to_tensor([self.circuit])
        circuits = tf.tile(circuit, [len(inputs)])
        symbols = tf.convert_to_tensor([str(x) for x in self.symbols])
        return tfq.resolve_parameters(circuits, symbols, inputs)


# Measurement
def readout(loc1, loc2, color, kernel1):
    imgsx = []
    imgsx.append((1 + cirq.X(loc1)) * (1 + cirq.X(loc2)) * (1 - cirq.X(color)))
    imgsx.append((1 - cirq.X(loc1)) * (1 + cirq.X(loc2)) * (1 - cirq.X(color)))
    imgsx.append((1 + cirq.X(loc1)) * (1 - cirq.X(loc2)) * (1 - cirq.X(color)))
    imgsx.append((1 - cirq.X(loc1)) * (1 - cirq.X(loc2)) * (1 - cirq.X(color)))

    imgsy = []
    imgsy.append((1 + cirq.Y(loc1)) * (1 + cirq.Y(loc2)) * (1 - cirq.Y(color)))
    imgsy.append((1 - cirq.Y(loc1)) * (1 + cirq.Y(loc2)) * (1 - cirq.Y(color)))
    imgsy.append((1 + cirq.Y(loc1)) * (1 - cirq.Y(loc2)) * (1 - cirq.Y(color)))
    imgsy.append((1 - cirq.Y(loc1)) * (1 - cirq.Y(loc2)) * (1 - cirq.Y(color)))

    imgsz = []
    imgsz.append((1 + cirq.Z(loc1)) * (1 + cirq.Z(loc2)) * (1 - cirq.Z(color)))
    imgsz.append((1 - cirq.Z(loc1)) * (1 + cirq.Z(loc2)) * (1 - cirq.Z(color)))
    imgsz.append((1 + cirq.Z(loc1)) * (1 - cirq.Z(loc2)) * (1 - cirq.Z(color)))
    imgsz.append((1 - cirq.Z(loc1)) * (1 - cirq.Z(loc2)) * (1 - cirq.Z(color)))

    kernelsx = []
    kernelsx.append((1 + cirq.X(kernel1)))
    kernelsx.append((1 - cirq.X(kernel1)))

    kernelsy = []
    kernelsy.append((1 + cirq.Y(kernel1)))
    kernelsy.append((1 - cirq.Y(kernel1)))

    kernelsz = []
    kernelsz.append((1 + cirq.Z(kernel1)))
    kernelsz.append((1 - cirq.Z(kernel1)))

    output = []
    for img in imgsx:
        for kernel in kernelsx:
            output.append(img * kernel)

    for img in imgsy:
        for kernel in kernelsy:
            output.append(img * kernel)

    for img in imgsz:
        for kernel in kernelsz:
            output.append(img * kernel)

    return output

# model build
def fqcnn():
    input_qubits = cirq.GridQubit.rect(1, 10)
    readout_operators = readout(input_qubits[2], input_qubits[5], input_qubits[9], input_qubits[7])

    input = tf.keras.Input(shape=(64, 4))

    r = Lambda(lambda x: x[:, :, 0])(input)
    g = Lambda(lambda x: x[:, :, 1])(input)
    b = Lambda(lambda x: x[:, :, 2])(input)
    x = Lambda(lambda x: x[:, :, 3])(input)

    encoding_layerR = Encoding(input_qubits)(r)
    qpc_layerR = tfq.layers.PQC(qdcnn(input_qubits), readout_operators)(encoding_layerR)

    encoding_layerG = Encoding(input_qubits)(g)
    qpc_layerG = tfq.layers.PQC(qdcnn(input_qubits), readout_operators)(encoding_layerG)

    encoding_layerB = Encoding(input_qubits)(b)
    qpc_layerB = tfq.layers.PQC(qdcnn(input_qubits), readout_operators)(encoding_layerB)

    encoding_layerX = Encoding(input_qubits)(x)
    qpc_layerX = tfq.layers.PQC(qdcnn(input_qubits), readout_operators)(encoding_layerX)

    concat_out = tf.keras.layers.concatenate([qpc_layerR, qpc_layerG, qpc_layerB, qpc_layerX])
    dense = tf.keras.layers.Dense(train_y.shape[1], activation='softmax', name='dense')(concat_out)

    qcnn_model = tf.keras.Model(inputs=[input], outputs=[dense])

    print(qcnn_model.summary())
    return qcnn_model


qcnn_model = fqcnn()
qcnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                   loss='categorical_crossentropy', metrics=['accuracy'])


# history = qcnn_model.fit(x=train_x, y=train_y,
#                          batch_size=batch_size,
#                          epochs=200,
#                          verbose=0,
#                          validation_data=(test_x, test_y))


qcnn_model.load_weights('fqcnn_model.h5')

_, acc = qcnn_model.evaluate(train_x, train_y)
print('train acc', acc)
_, acc = qcnn_model.evaluate(test_x, test_y)
print('test acc', acc)

