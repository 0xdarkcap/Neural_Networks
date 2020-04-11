import numpy as np

class NN:

    def __init__(self, input_neurons=2, hidden_neurons=[3, 3], output_neurons=1):

        self.input_neurons = input_neurons
        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons

        self.layers = [input_neurons] + hidden_neurons + [output_neurons]

        # initialize weights
        self.weights = np.array(
            [np.random.rand(self.layers[i + 1], self.layers[i]) for i in range(len(self.layers) - 1)])

        # initialize bias and activation
        self.bias = np.array([np.random.rand(self.layers[i], 1) for i in range(len(self.layers))])
        self.activations = np.array([np.zeros((self.layers[i], 1)) for i in range(len(self.layers))])
        self.z = np.array([np.zeros((self.layers[i], 1)) for i in range(len(self.layers))])
        self.error = np.array([np.zeros((self.layers[i], 1)) for i in range(len(self.layers))])
        # print("bias {}\nactivation {}\n z {}\n error {}\n".format(self.bias[1].shape, self.activations[1].shape, self.z[1].shape, self.error[1].shape))

    def forward_pass(self, inputs):
        # first layer activations are activations itself.
        inputs = np.array(inputs)
        self.activations[0] = inputs.reshape(len(inputs), 1)
        self.z[0] = inputs.reshape(len(inputs), 1)
        for i in range(len(self.layers) - 1):
            self.z[i + 1] = np.dot(self.weights[i], self.activations[i]).reshape(self.layers[i + 1], 1) + self.bias[
                i + 1]
            self.activations[i + 1] = self.sigmoid(self.z[i + 1])
            # print(self.z[i].shape)
        return self.activations[-1]

    def back_propagation(self, x, y, lr):
        x = np.array(x)
        y = np.array(y)
        """
        Four equations of backpropagation

        1.) Error in neurons of last layer L
            E[L] = dC/dZ[L], or
            E[L] = dC/da[L] * s`(z[L])

        2.) Error of a layer in terms of error in next layers.
            E[l] = dC/dz[l], or (applying chain rule)
            E[l] = ((w[l+1].T) * E[l+1]) * s`(z[l])

        3.) Change of cost wrt bias
            dC/db[l] = E[l]

        4.) Change of cost function wrt weight of any neuron
            dC/dw[l] = a[l-1] * E[l]

        L = last layer of network
        l = Any layer of network
        C = Cost function
        Z = inputs * weights + bias
        a = activations of a layer l or s(Z[l])
        s = sigmoid function, s` is derivative of sigmoid
        b = bias
        w = weights

        Let's go
        """
        # for given x output of network at current weights and biases

        # Calculate error in last layer (Using equation 1)
        self.error[-1] = (self.forward_pass(x) - y) * self.sigmoid_prime(self.activations[-1])

        # Calculate error in every layer moving backwards (Using equation 2)

        for i in reversed(range(len(self.layers) - 1)):
            self.error[i] = np.dot(self.weights[i].T, self.error[i + 1]) * self.sigmoid_prime(self.activations[i])

        # Calculate gradient of cost function wrt weights and biases (Using equation 3 and 4)
        delta_bias = np.array([self.error[i + 1] for i in range(len(self.layers) - 1)])
        delta_weight = np.array([self.error[i + 1] * self.activations[i].T for i in range(len(self.layers) - 1)])

        self.weights -= lr * delta_weight
        self.bias[1:] -= lr * delta_bias

    def train(self, x, y, epochs, lr):
        cost = []
        for i in range(epochs):
            for j in range(len(x)):
                self.back_propagation(x[j], y[j], lr)
                cost.append(np.mean(self.cost(x[j], y[j])))
            print("for epoch {}, cost = {}".format(i, np.mean(cost)))

    def predict(self, x):
        return np.argmax(self.forward_pass(x))

    def cost(self, x, y):
        return 1 / 2 * (self.forward_pass(x) - y) ** 2

    def sigmoid_prime(self, x):
        return x * (1 - x)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))


"""
if __name__ == "__main__":
    nn = NN()
    x = np.array([1, 2])
    outputs = nn.forward_pass(x)
    print(outputs)

    x1 = np.array([1, 1, 0, 0])
    x2 = np.array([1, 0, 1, 0])
    y = np.array([1, 0, 0, 0])
    for i in range(100):
        for j in range(4):
            nn.back_propagation([x1[j], x2[j]], [y[j]])
            print(nn.cost([x1[j], x2[j]], [y[j]]))
"""
