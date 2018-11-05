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

Let's take an example of a _fully-connected neural network_ with **2 hidden layers** and **2 neurons** in each layer.

![image](https://user-images.githubusercontent.com/18369201/47615122-351a9f00-daab-11e8-925d-75e41b537317.png)

_<center>Figure 1</center>_

#### EXAMPLE:
For our example we are going to create a neural network to predict if for a given height the person is an adult or a child.

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
# height in cm
data = [156, 140, 180, 192, 172, 162]

# 1 means adult, 0 means child
expected_output = [1, 0, 1, 1, 0, 1]

# Define some random weights
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


For our numerical example, we chose to train on RGB colors, and predict if the written color above them should be black or white.

To do so, we had to implement both the forward propagation and the backward propagation.
First of all, the forward propagation compute the weighted sum for each layer, for the initialized random weights and the input for each layer.

```Sum to write ``` (1)

But by computing the weighted sum, the obtained values have a range from minus infinity to infinity... That is why we are using one function to range them between 0 and 1. This function is called the activation function.
We chose to use only the sigmoid function as activation function since its derivative is easy to get :

```python
  def sigmoid(self, beforeAct, derivative = False):
   #activation function
    #sigmoid function
    if not derivative:
      return 1/(1+np.exp(-beforeAct))
     else:
      #derivative formula
      output = self.sigmoid(beforeAct)
    return output*(1 - output)))
```

Note that the boolean here is used to get the derivative when needed.
We then have successively output values from the weighted sum which are, after the activate those values, the input for the next layer. Eventually, we obtain an output for each input (example) from the input data.

Those first values should not be really good. That is why we are going to train our model, by modifying the weights according to each layer output. This is the backward propagation.





## DATA ANALYSIS


How many examples do I have in my dataset?
--> 70000 (60000 training + 10000 test)
What is the shape of my input images?
--> 28*28 of grayscale imaging
How many classes do I have?
10 classes
What is the class distribution in my dataset?
--> code
Is the testing set similar to the training set?
--> yes, also grayscale imaging, same dimension ...

softmax regression

https://dzone.com/articles/exploring-handwritten-digit-classification-a-tidy
Exploring pixel data:
  - How much gray is there in the set of images?
  - Average of the images
  - Which digits have more variability on average?
  - Find the images which are the least similar to the average, and nearest
  - Pixels that distinguish pairs of images


## Data pre-processing

### Loss function
To compile a model we have to use a loss function that returns a scalar for each data-point. This loss function takes two arguments : true labels and predictions. It measures the performance of a classification model whose output is a probability value between 0 and 1.

#### Cross entropy
Cross entropy loss could also be called log loss. If the predicted probability diverges from the actual observation label, cross entropy increases. Then a perfect model, which means that the predicted probability is 1, would have a cross entropy loss of 0 (log loss : log(1) = 0).

![Getting started](./cross_entropy.png)

_<center>Figure 2</center>_

From this figure, one can see that as predicted probability decreases log loss increases rapidely and, however, as predicted probability increases log loss decreases slowly.



#### Sparse cross entropy

### Different optimization algorithm
Optimization algorithms help us to minimize (or maximize) the error function, which is our most important objective while doing back propagation. This function depends on the learnable parameters of the network : the weights and the bias.
#### Standard stochastic Gradient Descent (SGD)
#### Adam optimizer
#### RMSProp optimizer
#### Nesterov Adam optimizer


### Metrics
When a model is compiled, we can put a special metric function which will judge the performance of the model.
#### Categorical accuracy
#### Sparse categorical accuracy
#### Top k categorical accuracy

## Model construction

### Implementation of the given network
