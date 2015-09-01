import numpy as np

import network

class ParseException(Exception):
    pass

def kth_standard_basis_vector(k):
    z = np.zeros((10, 1))
    z[k] = 1.0
    return z

def read_label_file(filename, vectorize_labels=False):
    with open(filename, 'rb') as f:
        magic = f.read(4)
        magic_int = int.from_bytes(magic, byteorder='big')
        if magic_int != 0x00000801:
            raise ParseException("Magic number in label file is not 2049.")

        n = int.from_bytes(f.read(4), byteorder='big')

        labels = []
        for i in range(0, n):
            label = f.read(1)
            if vectorize_labels:
                labels.append(kth_standard_basis_vector(label[0]))
            else:
                labels.append(label[0])

    assert len(labels) == n
    return labels

def read_image_file(filename):
    with open(filename, 'rb') as f:
        magic = f.read(4)
        magic_int = int.from_bytes(magic, byteorder='big')
        if magic_int != 0x00000803:
            raise ParseException("Magic number in label file is not 2051.")

        n = int.from_bytes(f.read(4), byteorder='big')
        rows = int.from_bytes(f.read(4), byteorder='big')
        cols = int.from_bytes(f.read(4), byteorder='big')
        num_pixels = rows*cols

        images = []
        for i in range(0, n):
            pixel_vec = np.fromstring(f.read(num_pixels), dtype=np.uint8)
            images.append(pixel_vec.reshape(len(pixel_vec), 1))

    assert len(images) == n
    return images

def read_training_data():
    image_file_name = 'data/train-images-idx3-ubyte'
    label_file_name = 'data/train-labels-idx1-ubyte'

    return list(zip(read_image_file(image_file_name),
                    read_label_file(label_file_name, vectorize_labels=True)))

def read_test_data():
    image_file_name = 'data/t10k-images-idx3-ubyte'
    label_file_name = 'data/t10k-labels-idx1-ubyte'

    return list(zip(read_image_file(image_file_name),
                    read_label_file(label_file_name)))


training_data = read_training_data()
test_data = read_test_data()

nn = network.NeuralNetwork([28*28, 30, 10])
num_epochs = 30
batch_size = 10
nn.sgd(training_data, num_epochs, batch_size, 3.0, test_data)

