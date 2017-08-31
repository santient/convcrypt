import click
import keras
import math
import numpy
import os
import shutil
import tarfile
import tqdm


def load_data(file_path):
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
        result = numpy.round(model.predict(key, batch_size=1)).astype(numpy.uint8)
        numpy.testing.assert_equal(result, data[i, :, :])


def compress(source_dir, output_filename):
    with tarfile.open(output_filename, 'w:gz') as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))


def save_encrypted(file_path, models, blocks, block_size, pad_size, dimensions):
    temp_dir = file_path + '#'
    os.makedirs(temp_dir)
    meta_file = os.path.join(temp_dir, 'meta')
    with open(meta_file, 'w') as meta:
        meta.write(str(blocks) + '\n')
        meta.write(str(block_size) + '\n')
        meta.write(str(pad_size) + '\n')
        meta.write(str(dimensions) + '\n')
    for i, model in enumerate(models):
        temp_file = os.path.join(temp_dir, 'block{}'.format(i))
        model.save_weights(temp_file)
    compress(temp_dir, file_path)
    shutil.rmtree(temp_dir)


def to_bytes(bits):
    bs = []
    for i in range(len(bits) // 8):
        bs.append(int(''.join(bits[i*8:(i+1)*8].astype(str)), 2))
    return bytes(bs)


def save_key(file_path, key):
    bits = key.flatten()
    key_data = to_bytes(bits)
    with open(file_path, 'wb') as f:
        f.write(key_data)


@click.command()
@click.option('--input_file', help="Name of file to encrypt.")
@click.option('--output_file', help="Name of encrypted file to write.")
@click.option('--key_file', help="Name of key file to write.")
@click.option('--dimensions', default=3, help="Dimensions of convolutional layer, between 1 and 3.")
@click.option('--block_size', default=32, help="Size of all dimensions for each data block.")
def encrypt(input_file, output_file, key_file, dimensions, block_size):
    """Encrypts the specified file using the ConvCrypt algorithm."""
    if input_file is None:
        raise ValueError("Please specify file input path.")
    if output_file is None:
        raise ValueError("Please specify file output path.")
    if key_file is None:
        raise ValueError("Please specify key output path.")
    if dimensions not in [1, 2, 3]:
        raise ValueError("Only 1, 2, or 3 dimensions are supported.")
    if block_size not in [8, 16, 32]:
        raise ValueError("Block size must be either 8, 16, or 32.")
    data_bits = load_data(input_file)
    data, pad_size = random_pad_3d(data_bits, block_size)
    blocks = num_blocks_3d(data, block_size)
    key = generate_key_cube(block_size)
    data_blocks = split_data(data, blocks)
    models = fit_models_3d(key, data_blocks)
    test_models(key, data_blocks, models)
    save_encrypted(output_file, models, blocks, block_size, pad_size, dimensions)
    save_key(key_file, key)
    print("Encryption complete.")


if __name__ == '__main__':
    encrypt()
