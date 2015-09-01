import random
import numpy as np

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

class NeuralNetwork:

    def __init__(self, sizes):
        """
        `sizes` is a list of the number of units in each layer.
        `sizes[0]` is the number of input units.

        This method initializes biases and weights randomly.
        This is a total ripoff of Michael Nielsen's `Network`
        class in his book "Neural Networks and Deep Learning".
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, inp):
        """
        Calculate the feed-forward output given input `inp`.
        """
        a = inp
        for (weights, biases) in zip(self.weights, self.biases):
            z = np.dot(weights, a) + biases
            a = sigmoid(z)

        return a

    def sgd(self, training_data, num_epochs, batch_size, eta, test_data=None):
        """
        Runs stochastic gradient descent to train the network.

        An *epoch* is when all items in `training_data` are used once. In each
        epoch, training data is broken up into blocks of size `batch_size` at
        random. Each `batch_size` collection of data is used to take a step
        under gradient descent. Keep taking steps until all the training data
        is used, and then the epoch is complete.

        `eta` is the learning rate used by gradient descent.
        """

        n = len(training_data)
        if test_data is not None: n_test = len(test_data)

        for i in range(0, num_epochs):
            random.shuffle(training_data)
            for j in range(0, n, batch_size):
                batch = training_data[j:j+batch_size]
                self.update_batch(batch, eta)

            if test_data is not None:
                print("Epoch {}: {}/{}".format(i, self.evaluate(test_data), n_test))
            else:
                print("Epoch {} complete.".format(i))


    def update_batch(self, batch, eta):
        """
        Take a step under gradient descent using a batch of examples.
        """
        # To calculate the gradient for the cost function corresponding to
        # the whole batch, we can average the gradients for the individual
        # cost functions corresponding to each example.
        batch_grad_b = [np.zeros(b.shape) for b in self.biases]
        batch_grad_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in batch:
            ex_grad_b, ex_grad_w = self.backpropagation(x, y)
            batch_grad_b = [batch_b + ex_b for batch_b, ex_b in zip(batch_grad_b, ex_grad_b)]
            batch_grad_w = [batch_w + ex_w for batch_w, ex_w in zip(batch_grad_w, ex_grad_w)]

        m = len(batch)
        self.weights = [w - (eta/m) * grad_w
                        for w, grad_w in zip(self.weights, batch_grad_w)]
        self.biases = [b - (eta/m) * grad_b
                        for b, grad_b in zip(self.biases, batch_grad_b)]



    def backpropagation(self, x, y):
        """
        Compute the gradient of the network's cost function according to a
        single training example.

        Steps:
        
          - compute output layer error
          - "backpropagate" the error to compute errors in previous layers
          - compute gradient from the layer-by-layer errors in previous steps

        We're using the quadratic cost function here.
        """
        # the activations at each layer are needed for the gradient
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]

        activations = [x]
        a = x
        for (weights, biases) in zip(self.weights, self.biases):
            wa = np.dot(weights, a)
            z = wa + biases
            a = sigmoid(z)
            activations.append(a)

        # output layer error, = grad C(a) (*Hadamard product*) sigmoid'(z)
        # note sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z)) = a * (1 - a)
        delta = (activations[-1] - y) * activations[-1] * (1 - activations[-1])

        grad_b[-1] = delta
        grad_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            a = activations[-l]
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * a * (1 - a)
            grad_b[-l] = delta
            grad_w[-l] = np.dot(delta, activations[-l - 1].transpose())

        return grad_b, grad_w


    def evaluate(self, test_data):
        correct = 0
        for x, y in test_data:
            out = np.argmax(self.feedforward(x))
            if out == y: correct += 1
        return correct
