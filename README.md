# ConvCrypt
### By Santiago Benoit
ConCrypt is an experimental encryption algorithm that uses n-dimensional convolutional neural networks. It currently supports only 3-dimensional convolutions. Please note that this encryption algorithm is a proof-of-concept for experimental purposes only. It is by no means practical, and should not be used to encrypt important files.

## Usage
python3 [encrypt/decrypt].py [--parameters]

Encryption parameters:
- --input-file: Name of file to encrypt.
- --output-file: Name of encrypted file to write.
- --key-file: Name of key file to write.
- --dimensions: Dimensions of convolutional layer (only 3 are currently supported).
- --block-size: Size of all dimensions for each data block (only 8, 16, or 32 are supported).

Decryption parameters:
- --input-file: Name of file to decrypt.
- --output-file: Name of decrypted file to write.
- --key-file: Name of key file to use.

## How it Works
The first step in the algorithm is to separate the data into blocks of the specified size. The number of bits in each blocks is the block-size raised to the specified dimensionality of the convolution (which would currently be only 3). The last data block is randomly padded to fit this size. Then, the key is generated: for 3D convolutions, it is a randomly generated cube of bits with the same size as a data block. Lastly, a convolutional neural network is trained to convolve the key into each data block, so that each data block gets its own trained network. In a way, the networks are purposely overfit to the data. The resulting encrypted data is the weights of each of the networks (the values of the kernel tensors).

![2D Convolution](/images/3D_Convolution_Animation.gif?raw=true)

Visualization of a 2D Convolution. (Source: Wikimedia Commons)

## Drawbacks
Due to the nature of this algorithm, keys that are very similar to each other tend to encode for the same data. Also, this algorithm takes a very long time for large files because there are more data blocks to reconstruct, and hence more networks to train. The resulting encrypted file is always bigger than the input file.

## Future Directions
In the future, I plan to convolve byte values rather than individual bits so that the encrypted data is smaller and takes less time to create. It might be possible to create a training algorithm for the neural networks optimized for this encryption algorithm, or to abandon neural networks entirely for this purpose. I also plan to test the security of this algorithm and to add the option to use 1D or 2D convolutions.
