# Neural_Networks
### Todo list
- [x] Initialize weights, baises, activations
- [x] Forward Propagation
- [x] Back Propagation
- [ ] Training method
- [ ] Refactorisation
- [ ] Update Readme

# Table of Contents
1. Introduction
2. Before we start
    
# Introduction
Hello, hive community it’s been awhile since I wrote anything on steemit in fact I think it’s been way above year, my friend told me about hive and what happened with statement I’m excited about hive, happy to be a part of community. Now I have been learning programming for a while now and machine learning is one of the computer sciences concepts that took my attention as I was pursuing my computer science degree. In my university there is no module related to machine learning. So, I self-taught myself and I went to University of Nottingham, Ningbo as an exchange student and completed a machine learning module. Machine learning itself has a lot of concepts, algorithms, maths and it can get quite intimidating but once you give enough time and take everything step-by-step then you enjoy learning the so-called hard concepts around machine learning and deep learning. Because of the coronavirus we have been locked down into our houses, me personally I have been locked down for a month now the meantime I decided to make neural networks from scratch with just using numpy, which is a Python library for the array manipulation, it’s faster than Python lists and it has various functions that are quite useful as you will see later. So in this blog I will try my best to show you how I made artificial neural networks and you can too.

# Before we start
Before we start
Sweet little warning before I start, I’m still learning as a student. I might not use professional have the maths symbols but I’ll try my best to make you   punderstand what’s going on and if you find any mistakes, I encourage you to point them out so that we can improve the quality of this blog post as time progresses. Now before we start, there are a couple prerequisites that will help you understand the text and if you want to follow along the blog then would recommend that you have some understanding of the following topics- 
* Python – this would be our language. Although you can use any language since we are going to make everything from scratch and I think you will be able to follow along in any popular programming language. But I think it is much easier to do in python. Now you don’t have to be a professional in Python. You should know loops, functions, classes and syntax. 
    * Language Syntax
    * Loops
    * Classes
* Maths – To understand how neural networks learn to have to have understanding of some mathematical topics and these topics include
    * Calculus – You don’t need to know the whole calculus; we will use chain rule and partial derivatives when we do the back-propagation algorithm.
        * Partial Derivatives
        * Chain Rule
    * Matrices – We will store all the weights and biases in the matrix represented by NumPy arrays for easy mathematical operations. Matrices will make our life a lot easier and once you get familiar with the notation and representation of a matrix then it will be easy to generalize the calculation deep and heavy neural networks. We will use the transpose property of Matrices, I’ll just give you a hint to change rows with columns but I urge you to understand what that really means. Also you will need matrix multiplication, there are some laws that go into how you multiply Matrices.
        * Matrix operations (Addition, Multiplication etc.)
        * Transpose of a Matrix
# Neural Networks
Neural Networks
You can think of neural networks as a big function that takes in input values and gives output and some magic happens in the middle that makes our output logical to the problem we are trying to solve, now we don’t hardcode the logic the neural network learns the logic for itself by using data. However, in this blog we are going to make the neural networks from scratch. Now if you are familiar with how neural networks work, their structure and mathematics then, you can go ahead to next section. But if you are new to this topic that I would highly recommend you to watch a [video]( https://youtu.be/TalSpaMJ_2k) from my favorite mathematics YouTube channel [3Blue1Brown]( https://www.youtube.com/channel/UCYO_jab_esuFRV4b17AJtAw) about neural networks.

We are going to learn the theory and code as we move along and before we start I’ll give you a brief structure of theory and code. That will give you an idea of what we are going to do how we are going to achieve that. 
#### single neuron
you can imagine a single neuron being a kind of function, which takes in some inputs and multiplies them with corresponding weights and then passes the result through in this case a sigmoid function will talk about sigmoid function later in this post. Let us just look at the code for the single neuron.
```python
import math

import numpy as np

# activation function
def sigmoid(x):
    return 1.0 / (1 + math.exp(-x))

def activate(inputs, weights):
    h = 0
    for x, w in zip(inputs, weights):
        h += x * w
    # perform activation
    return sigmoid(h)

#entry point
if __name__ == "__main__": 
    inputs = np.array([0.5, 0.3, 0.2])
    weights = np.array([0.4, 0.7, 0.2])
    output = activate(inputs, weights)
    print(output)
```

As you can see the neuron takes in multiple inputs, multiplies the inputs with corresponding weights, passes the result through sigmoid and returns the output. 
#### activation function
As you can see the neuron takes in multiple inputs, multiplies the inputs with corresponding weights, passes the result through sigmoid and returns the output. 
The output of a neuron will depend upon the activation function, the activation function takes in weighted sum from previous layer ( in single neuron, weighted sum of inputs ) and outputs a value between a certain upper limit and the lower limit in this case we are going to use Sigmoid. As you can see from the figure the range of the sigmoid function is (0 to 1). So as you can see if the input of a sigmoid function is a very negative number then the sigmoid function’s output will be close to 0 and if the input is a very big positive no. then the output will be close to 1. 

## Structure

Before we start coding, you’ll need a couple of things. Now if you are familiar with Python just install numpy. If you do not have Python installed in your system then [click here]( https://www.python.org/downloads/) to download it. Next up we need numpy, it’s a library of Python and we will use it for array manipulation.

To install the numpy open up your terminal and type the following command
```python
pip install numpy
```
Now open up your text editor and let’s start coding.
```python
import numpy as np
```
First of all we are going to make a class and our class will represent the whole neural network. Also you can find the whole code on github. 
```python
class NN:
```
Now inside this class we will make constructor which in Python made by __init__ function it will take in some values which will be input Neurons, hidden neurons and output neurons and based on the size of every parameter that has been passed we will initialize our weights basises, activations etc.
```python
def __init__(self, input_neurons=2, hidden_neurons=[3, 3], output_neurons=1):
```

| Parameter | Type | Purpose |
|-------------------|:-----:|:----------|
| input_neurons | `int` | It's the number of neurons in first layer or number of dimensions.|
| hidden_neurons | `list` | The size of list represents the number of hidden layers and the value at each index will represent number of neurons in that layer.
| output_neurons | `int` | It represents the number of neurons in last layer or output layer

We will use all these parameters to initialise weights and  because we are making a fully connected neural network every neuron will be connected to every other neuron of next layer.  and since we have the information of how many neurons in each layer are  we can initialize the weights, biases and activations etc. So let's do that 

First we will make a list, that will represent our whole neural network.   it will be helpful initialising parameters such as weights  baises etc. for the whole network  since right now we have different parameters as input Neurons, hidden neurons and output neurons. 

```python
self.layers = [input_neurons] + hidden_neurons + [output_neurons]
```
 here we used array concatenation property of python list  notice how we didn't put `[ ]`  around hidden_neurons because it's already a Python list
 
 Now let's start with initializing everything.
```python
self.weights = np.array([np.random.rand(self.layers[i + 1], self.layers[i]) for i in range(len(self.layers) - 1)])

# initialize bias and activation
self.bias = np.array([np.random.rand(self.layers[i], 1) for i in range(len(self.layers))])
self.activations = np.array([np.zeros((self.layers[i], 1)) for i in range(len(self.layers))])
self.z = np.array([np.zeros((self.layers[i], 1)) for i in range(len(self.layers))])
self.derivative = np.array([np.random.zeros(self.layers[i + 1], self.layers[i]) for i in range(len(self.layers) - 1)])
```


# Feed forward

# Backpropagation

# References

# code