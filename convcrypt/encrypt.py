import click
import keras
import math
import numpy
import os
import shutil
import tarfile
import tensorflow
import tqdm


def read_bits(file_path):
    bits = []
    with open(file_path, 'rb') as f:
        bs = (ord(chr(b)) for b in f.read())
        for b in bs:
            for i in reversed(range(8)):
                bits.append((b >> i) & 1)
    return numpy.array(bits)


def next_size_3d(n, block_size):
    return int(math.ceil(n / block_size ** 3) * block_size ** 3)


def random_pad_3d(bits, block_size):
    pad_size = next_size_3d(len(bits), block_size) - len(bits)
    pad_array = numpy.random.randint(2, size=(pad_size,))
    return numpy.concatenate([bits, pad_array])[numpy.newaxis, :], pad_size


def num_blocks_3d(data, block_size):
    return data.shape[1] // block_size ** 3


def generate_key_cube(block_size): 
    return numpy.random.randint(2, size=(1, block_size, block_size, block_size, 1))


def split_data(data, blocks):
    return data.reshape(blocks, 1, data.shape[1] // blocks)


def init_model_3d(input_shape):
    input_layer = keras.layers.Input(input_shape)
    x = keras.layers.Conv3D(8, (input_shape[0] // 2 + 1, input_shape[1] // 2 + 1, input_shape[2] // 2 + 1))(input_layer)
    x = keras.layers.Flatten()(x)
    output_layer = keras.layers.Activation('sigmoid')(x)
    model = keras.models.Model(input_layer, output_layer)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def fit_models_3d(key, data):
    models = []
    for block in tqdm.tqdm(data):
        model = init_model_3d(key.shape[1:])
        acc = 0
        while acc < 1:
            model.fit(key, block, batch_size=1, epochs=1, verbose=0)
            acc = model.evaluate(key, block, batch_size=1, verbose=0)[1]
        models.append(model)
    return models


def test_models(key, data, models):
    for i, model in enumerate(models):
        result = int(numpy.round(model.predict(key, batch_size=1)))
        numpy.testing.assert_equal(result, data[i, :, :])


def compress_tar(source_dir, output_filename):
    with tarfile.open(output_filename, 'w:gz') as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))


def save_data(file_path, models, block_size, pad_size, dimensions):
    temp_dir = file_path + '#'
    os.makedirs(temp_dir)
    meta_file = os.path.join(temp_dir, 'meta')
    with open(meta_file, 'w') as meta:
        meta.write('block_size={}\n'.format(block_size))
        meta.write('pad_size={}\n'.format(pad_size))
        meta.write('dimensions={}\n'.format(dimensions))
    for i, model in enumerate(models):
        temp_file = os.path.join(temp_dir, 'block{}'.format(i))
        model.save_weights(temp_file)
    compress_tar(temp_dir, file_path)
    shutil.rmtree(temp_dir)


def save_key(file_path, key):
    numpy.save(file_path, key)
    os.rename(file_path + '.npy', file_path)


@click.command()
@click.option('--input_path', help="Name of file to encrypt.")
@click.option('--output_path', help="Name of encrypted file to write.")
@click.option('--key_path', help="Name of key file to write.")
@click.option('--dimensions', default=3, help="Dimensions of convolutional layer, between 1 and 3.")
@click.option('--block_size', default=32, help="Size of all dimensions for each data block.")
def encrypt(input_path, output_path, key_path, dimensions, block_size):
    if dimensions not in [1, 2, 3]:
        raise ValueError("Only 1, 2, or 3 dimensions are supported.")
    data_bits = read_bits(input_path)
    data, pad_size = random_pad_3d(data_bits, block_size)
    blocks = num_blocks_3d(data, block_size)
    key = generate_key_cube(block_size)
    dataset = split_data(data, blocks)
    models = fit_models_3d(key, dataset)
    test_models(key, dataset, models)
    save_data(output_path, models, block_size, pad_size, dimensions)
    save_key(key_path, key)
    print("Encryption complete.")


if __name__ == '__main__':
    encrypt()
