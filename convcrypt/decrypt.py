import click
import keras
import math
import numpy
import os
import shutil
import tarfile
import tqdm


def init_model_3d(input_shape):
    input_layer = keras.layers.Input(input_shape)
    x = keras.layers.Conv3D(8, (input_shape[0] // 2 + 1, input_shape[1] // 2 + 1, input_shape[2] // 2 + 1))(input_layer)
    x = keras.layers.Flatten()(x)
    output_layer = keras.layers.Activation('sigmoid')(x)
    model = keras.models.Model(input_layer, output_layer)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def load_models(file_path):
    models = []
    with tarfile.open(file_path, 'r:gz') as tar:
        for member in tar.getmembers():
            f = tar.extractfile(member)
            if f.name == 'meta':
                metadata = f.read().split('\n')
                block_size = metadata[0]
                pad_size = metadata[1]
                dimensions = metadata[2]
            if 'block' in f.name:
                model = init_model_3d((block_size, block_size, block_size, 1))
                model.load_weights(f.name)
                models.append(model)
    return models, block_size, pad_size, dimensions
# TODO: actually extract files so that weights can be loaded


def load_key(file_path):
    pass


def cubify(bits):
    pass


def decrypt_data(models, key):
    pass


def reconstruct(data, pad_size):
    pass


def save_data(file_path, data):
    pass


@click.command()
@click.option('--input_path', help="Name of file to decrypt.")
@click.option('--output_path', help="Name of decrypted file to write.")
@click.option('--key_path', help="Name of key file to use.")
def decrypt(input_path, output_path, key_path):
    """Decrypts the specified file using the ConvCrypt algorithm."""
    models, block_size, pad_size, dimensions = load_models(input_path)
    key_bits = load_key(key_path)
    key = cubify(key_bits)
    data_blocks = decrypt_data(models, key)
    data = reconstruct(data_blocks, pad_size)
    save_data(output_path, data)
    print("Decryption complete.")


if __name__ == '__main__':
    decrypt()
