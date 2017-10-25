# ConvCrypt
### By Santiago Benoit
ConCrypt is an experimental data encryption using n-dimensional convolutional neural networks. It currently supports only 3-dimensional convolutions. Please note that this encryption algorithm is a proof-of-concept for experimental purposes only. It is by no means practical, and should not be used to encrypt important files.

## Usage
python3 [encrypt/decrypt].py [--parameters]

Encryption parameters:
- --input-file: Name of file to encrypt.
- -- output-file: Name of encrypted file to write.
- --key-file: Name of key file to write.
- --dimensions: Dimensions of convolutional layer (only 3 are currently supported).
- --block-size: Size of all dimensions for each data block (only 8, 16, or 32 are supported).

Decryption parameters:
- --input-file: Name of file to decrypt.
- --output-file: Name of decrypted file to write.
- --key-file: Name of key file to use.

## How it works
TODO
