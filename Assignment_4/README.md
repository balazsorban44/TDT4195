# Image Processing


##  Numerical examples

- Numerical example of forward/backward propagation
- Visualization of weights in a fully-trained convolutional neural network


<!-- ### Neuron
  A neuron, or perceptron is the building block of any neural networks. A single neuron to work needs the following:
  - an activation function
  - a cost function
  - weight updating
  - learning rate

### Network of neurons -->

Let's take an example of a _fully-connected neural network_ with **1 hidden layer** and **2 neurons** in each layer.

![image](https://user-images.githubusercontent.com/18369201/47603440-3df86b80-d9ec-11e8-96e9-df2ce7e5a3d9.png)
_<center>Figure 1</center>_

#### EXAMPLE:
For our example we are going to create a neural network to predict if for a given background color we should use either a light or dark text color that is easily readable.

https://www.youtube.com/watch?v=9Hz3P1VgLz4
##### Prerequisites
We choose `sigmoid` as our _activation function_:
```python
def sigmoid(z):
    return 1/(1+math.exp(-z))
```

For `weight initialization`, we use random values from _normal distribution_. (We use NumPy to make life easier 👌):

```python
def initialize_weights(input_size, hidden_layer_size, output_size):
  #numpy random normal distribution
  W1 = np.random.normal(0, 1/np.sqrt(input_size), (hidden_layer_size, input_size))
  W2 = np.random.normal(0, 1/np.sqrt(hidden_layer_size), (output_size, hidden_layer_size))

  #weights dictionary  
  parameters = {"W1": W1, "W2": W2}
    
  return parameters
```


##### Data

```python
# (r,g,b)
input = [
  (0,0,0) , # black
  (255,255,255) # white
]


# 1 means light, 0 means dark
expected_output = [
  1,
  0
]

weights = [
  -0.2, 0.34,
   0.1,-0.4,
]

```

z<sup>0</sup><sub>0</sub> = W<sup>0</sup><sub>0,0</sub> * input<sup>0</sup><sub>0</sub> + W<sup>0</sup><sub>1,0</sub> * input<sup>0</sup><sub>1</sub>

z<sup>0</sup><sub>1</sub> = W<sup>0</sup><sub>0,1</sub> * input<sup>0</sup><sub>0</sub> + W<sup>0</sup><sub>1,1</sub> * input<sup>0</sup><sub>1</sub>

input<sup>1</sup><sub>0</sub> = sigmoid(z<sup>0</sup><sub>0</sub>)

input<sup>1</sup><sub>1</sub> = sigmoid(z<sup>0</sup><sub>1</sub>)

z<sup>1</sup><sub>0</sub> = W<sup>1</sup><sub>0,0</sub> * input<sup>1</sup><sub>0</sub> +  W<sup>1</sup><sub>1,0</sub> * input<sup>1</sup><sub>1</sub>

z<sup>1</sup><sub>1</sub> = W<sup>1</sup><sub>0,1</sub> * input<sup>1</sup><sub>0</sub> +  W<sup>1</sup><sub>1,1</sub> * input<sup>1</sup><sub>1</sub>

input<sup>2</sup><sub>0</sub> = sigmoid(z<sup>1</sup><sub>0</sub>)

input<sup>2</sup><sub>1</sub> = sigmoid(z<sup>1</sup><sub>1</sub>)