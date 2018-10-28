import numpy as np
import math

def sigmoid(z):
  return 1/(1 + math.exp(-z))


def error(expected_output, output):
  error = np.sum(np.subtract(expected_output, output) ** 2)
  error = np.squeeze(error)
  return error

def initialize_weights(n_x, n_h, n_y):
  W1 = np.random.normal(0, 1/np.sqrt(n_x), (n_h, n_x))
  W2 = np.random.normal(0, 1/np.sqrt(n_h), (n_y, n_h))
  assert (W1.shape == (n_h, n_x))
  assert (W2.shape == (n_y, n_h))
  
  parameters = {"W1": W1,
                "W2": W2}
  return parameters
  
def weighted_sum(weights, data):
  return np.matmul(weights, data)


def forward_propagation(input_1st_layer, weights):
  sum_1st_neuron_1st_layer = weighted_sum(weights["W1"], input_1st_layer)[0]
  sum_2nd_neuron_1st_layer = weighted_sum(weights["W1"], input_1st_layer)[1]
  
  ouput_1st_neuron = sigmoid(sum_1st_neuron_1st_layer)
  ouput_2nd_neuron = sigmoid(sum_2nd_neuron_1st_layer)

  input_2nd_layer = np.array([ouput_1st_neuron, ouput_2nd_neuron])
  sum_1st_neuron_2nd_layer = weighted_sum(weights["W2"], input_2nd_layer)[0]
  sum_2nd_neuron_2nd_layer = weighted_sum(weights["W2"], input_2nd_layer)[1]
  
  ouput_1st_neuron = sigmoid(sum_1st_neuron_2nd_layer)
  ouput_2nd_neuron = sigmoid(sum_2nd_neuron_2nd_layer)

  return np.array([ouput_1st_neuron, ouput_2nd_neuron])

if __name__ == '__main__':
  data = np.array([156, 140, 180, 192, 172, 162])
  expected_output = np.array([1, 0, 1, 1, 0, 1])
  output = None


  weights = initialize_weights(np.size(data), 2, 2)

  for i in range(10):
    output = forward_propagation(data, weights)
    weights = error(expected_output, output)
    