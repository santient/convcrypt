import click
import keras
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


def load_data(file_path):
    models = []
    temp_dir = file_path + '#'
    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall(path=temp_dir)
        with open(os.path.join(temp_dir, 'meta'), 'r') as meta:
            metadata = meta.read().split('\n')
            blocks = int(metadata[0])
            block_size = int(metadata[1])
            pad_size = int(metadata[2])
            dimensions = int(metadata[3])
        for i in range(blocks):
            path = os.path.join(temp_dir, 'block{}'.format(i))
            model = init_model_3d((block_size, block_size, block_size, 1))
            model.load_weights(path)
            models.append(model)
    shutil.rmtree(temp_dir)
    return models, blocks, block_size, pad_size, dimensions


def load_key(file_path):
    bits = []
    with open(file_path, 'rb') as f:
        bs = (ord(chr(b)) for b in f.read())
        for b in bs:
            for i in reversed(range(8)):
                bits.append((b >> i) & 1)
    return numpy.array(bits)


def cubify(bits):
    size = int(numpy.round(len(bits) ** (1 / 3)))
    return bits.reshape(1, size, size, size, 1)


def decrypt_data(models, key):
    data_blocks = []
    for model in tqdm.tqdm(models):
        data_blocks.append(model.predict(key, batch_size=1))
    return numpy.round(numpy.array(data_blocks)).astype(numpy.uint8).flatten()


def unpad(data, pad_size):
    return data[:-pad_size]


def save_data(file_path, data):
    with open(file_path, 'wb') as f:
        f.write(data)


@click.command()
@click.option('--input_path', help="Name of file to decrypt.")
@click.option('--output_path', help="Name of decrypted file to write.")
@click.option('--key_path', help="Name of key file to use.")
def decrypt(input_path, output_path, key_path):
    """Decrypts the specified file using the ConvCrypt algorithm."""
    models, blocks, block_size, pad_size, dimensions = load_data(input_path)
    key_bits = load_key(key_path)
    key = cubify(key_bits)
    data_blocks = decrypt_data(models, key)
    data = unpad(data_blocks, pad_size)
    save_data(output_path, data)
    print("Decryption complete.")


if __name__ == '__main__':
    decrypt()
