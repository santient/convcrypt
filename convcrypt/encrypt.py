import click
import keras
import math
import numpy


def read_bits(file_path):
    bits = []
    with open(file_path, 'rb') as f:
        bs = (ord(chr(b)) for b in f.read())
        for b in bs:
            for i in reversed(range(8)):
                bits.append((b >> i) & 1)
    return numpy.array(bits)


def next_even_square(n):
    return int((math.ceil(n ** (1 / 2) / 2) * 2) ** 2)


def next_even_cube(n):
    return int((math.ceil(n ** (1 / 3) / 2) * 2) ** 3)


def init_model_3d(shape):
    input_layer = keras.layers.Input(shape)
    output_layer = keras.layers.Conv3D(8, (shape[0] / 2 + 1, shape[1] / 2 + 1, shape[2] / 2 + 1), activation='sigmoid')
    model = keras.models.Model(input_layer, output_layer)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model


def cubify(bits):
    pad_size = next_even_cube(len(bits)) - len(bits)
    pad_array = numpy.zeros((pad_size,))
    return numpy.concatenate([bits, pad_array])


def training_data_3d(key, data):
    size = int(numpy.round(len(data) ** (1 / 3)))
    x = key.reshape(1, size, size, size, 1)
    y = data.reshape(1, size, size, size, 1)
    return x, y


def fit(model, x, y):
    acc = 0
    while acc < 1:
        model.fit(x, y, batch_size=1, epochs=1, verbose=0)
        acc = model.evaluate(x, y, batch_size=1, verbose=0)[1]
        print("Accuracy:", acc)
    return model


if __name__ == '__main__':
    data_bits = read_bits(file_path)
    data_cube = cubify(bit_data)
    
