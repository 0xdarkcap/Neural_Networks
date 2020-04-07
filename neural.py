import numpy as np

class NN:

    def __init__(self, input_neurons=2, hidden_neurons=[3,3], output_neurons=2):

        self.input_neurons = input_neurons
        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons

        self.layers = [input_neurons] + hidden_neurons + [output_neurons]

        # initialize weights
        weights = []
        for i in range(len(self.layers)-1):
            w = np.random.rand(self.layers[i], self.layers[i+1])
            weights.append(w.T)
        self.weights = weights

        # initialize bias and activation
        bias = []
        activations = []
        for i in range(len(self.layers)):
            bias.append(np.zeros(self.layers[i]).reshape(self.layers[i],1))
            activations.append(np.zeros(self.layers[i]).reshape(self.layers[i],1))
        self.bias = bias
        self.activations = activations
        print(self.weights[0].shape, self.activations[0].shape)


    def forward_pass(self, inputs):
        self.activations[0] = inputs
        for i in range(len(self.layers)-1):
            self.activations[i+1] = self.sigmoid(self.bias[i+1] + np.dot(self.weights[i], self.activations[i]))
        return self.activations[len(self.layers)-1]

    def back_propagation(self):
        pass

    def cost(self):
        pass

    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    def sigmoid_prime(self):
        pass

if __name__ =="__main__":
    nn = NN()
    outputs = nn.forward_pass([1,2])
    print(outputs)