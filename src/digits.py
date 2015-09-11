import numpy as np

import network

class ParseException(Exception):
    pass

def kth_standard_basis_vector(k):
    z = np.zeros((10, 1))
    z[k] = 1.0
    return z

def read_label_file(filename, hold_out=None):
    with open(filename, 'rb') as f:
        magic = f.read(4)
        magic_int = int.from_bytes(magic, byteorder='big')
        if magic_int != 0x00000801:
            raise ParseException("Magic number in label file is not 2049.")

        n = int.from_bytes(f.read(4), byteorder='big')

        if hold_out is None:
            labels = []
            for i in range(0, n):
                label = f.read(1)
                labels.append(label[0])
            assert len(labels) == n
            return labels
        else:
            labels1 = []
            for i in range(0, n - hold_out):
                label = f.read(1)
                labels1.append(label[0])

            labels2 = []
            for i in range(0, hold_out):
                label = f.read(1)
                labels2.append(label[0])

            assert len(labels1) == n - hold_out
            assert len(labels2) == hold_out
            return labels1, labels2


def read_image_file(filename, hold_out=None):
    with open(filename, 'rb') as f:
        magic = f.read(4)
        magic_int = int.from_bytes(magic, byteorder='big')
        if magic_int != 0x00000803:
            raise ParseException("Magic number in label file is not 2051.")

        n = int.from_bytes(f.read(4), byteorder='big')
        rows = int.from_bytes(f.read(4), byteorder='big')
        cols = int.from_bytes(f.read(4), byteorder='big')
        num_pixels = rows*cols

        # reads `num_images` from f, each being a list of `num_pixels` bytes
        # converts them to floats between 0 and 1 and appends to `images_list`
        def append_images(images_list, num_images, num_pixels):
            for i in range(0, num_images):
                pixel_vec = np.fromstring(f.read(num_pixels), dtype=np.uint8)
                float_pixel_vec = np.zeros((len(pixel_vec), 1))
                for j in range(0, len(pixel_vec)):
                    float_pixel_vec[j] = float(pixel_vec[j])/255.0

                images_list.append(float_pixel_vec)

        if hold_out is None:
            images = []
            append_images(images, n, num_pixels)
            assert len(images) == n
            return images
        else:
            images1, images2 = [], []
            append_images(images1, n - hold_out, num_pixels)
            append_images(images2, hold_out, num_pixels)
            assert len(images1) == n - hold_out
            assert len(images2) == hold_out
            return images1, images2


def load_data():
    train_image_file = 'data/train-images-idx3-ubyte'
    train_label_file = 'data/train-labels-idx1-ubyte'
    test_image_file = 'data/t10k-images-idx3-ubyte'
    test_label_file = 'data/t10k-labels-idx1-ubyte'

    training_images, validation_images = read_image_file(train_image_file, hold_out=10000)
    training_labels, validation_labels = read_label_file(train_label_file, hold_out=10000)

    training_labels = [kth_standard_basis_vector(y) for y in training_labels]

    training_data = list(zip(training_images, training_labels))
    validation_data = list(zip(validation_images, validation_labels))
    test_data = list(zip(read_image_file(test_image_file),
                         read_label_file(test_label_file)))

    return (training_data, validation_data, test_data)



training_data, validation_data, test_data = load_data()

nn = network.NeuralNetwork([28*28, 30, 10])
num_epochs = 30
batch_size = 10
nn.sgd(training_data, num_epochs, batch_size, 3.0, test_data)

