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

  parameters = {"w1": W1,
                "w2": W2}
  return parameters


def initialize_parameters(n_x, n_h, n_y):
  b1 = np.zeros((n_h, 1))
  b2 = np.zeros((n_y, 1))
  weights = initialize_weights(n_x, n_h, n_y)
  parameters = {**weights,
                "b1": b1,
                "b2": b2}
  return parameters

def weighted_sum(weights, data):
  return np.matmul(weights, data)


def forward_propagation(input_1st_layer, parameters):

  sum_1st_neuron_1st_layer = np.add(weighted_sum(parameters["w1"], input_1st_layer)[0], parameters["b1"][0])
  sum_2nd_neuron_1st_layer = np.add(weighted_sum(parameters["w1"], input_1st_layer)[1], parameters["b1"][1])

  ouput_1st_neuron_1st_layer = sigmoid(sum_1st_neuron_1st_layer)
  ouput_2nd_neuron_1st_layer = sigmoid(sum_2nd_neuron_1st_layer)

  input_2nd_layer = np.array([ouput_1st_neuron_1st_layer, ouput_2nd_neuron_1st_layer])
  sum_1st_neuron_2nd_layer = np.add(weighted_sum(parameters["w2"], input_2nd_layer)[0], parameters["b2"][0])
  sum_2nd_neuron_2nd_layer = np.add(weighted_sum(parameters["w2"], input_2nd_layer)[1], parameters["b2"][1])

  ouput_1st_neuron_2nd_layer = sigmoid(sum_1st_neuron_2nd_layer)
  ouput_2nd_neuron_2nd_layer = sigmoid(sum_2nd_neuron_2nd_layer)

  cache = {
    "s11" : sum_1st_neuron_1st_layer,
    "s21" : sum_2nd_neuron_1st_layer,
    "o11" : ouput_1st_neuron_1st_layer,
    "o21" : ouput_2nd_neuron_1st_layer,
    "s12" : sum_1st_neuron_2nd_layer,
    "s22" : sum_2nd_neuron_2nd_layer,
    "o12" : ouput_1st_neuron_2nd_layer,
    "o22" : ouput_2nd_neuron_2nd_layer,
  }

  return np.array([ouput_1st_neuron_2nd_layer, ouput_2nd_neuron_2nd_layer]), cache


def backward_propagation(parameters, cache, X, Y):
  O12 = cache["o12"]
  O22 = cache["o22"]
  S12 = cache["s12"]
  S22 = cache["s22"]
  S11 = cache["s11"]
  S21 = cache["s21"]

  W12 = np.transpose(parameters["w2"])[0]
  W22 =  np.transpose(parameters["w2"])[1]
  print(X)
  d12 = (O12 - Y) * sigmoid(S12) * (1 - sigmoid(S12))
  d22 = (O22 - Y) * sigmoid(S22) * (1 - sigmoid(S22))


  #d11 = W12.T * d12 * sigmoid(S11) * (1 - sigmoid(S11))

  d11 = W12 * d12 * np.multiply(sigmoid(S11),(1 - sigmoid(S11)))


  d21 = W22 * d22 * np.multiply(sigmoid(S21),(1 - sigmoid(S21)))


  partial_derivative12 = d12 * S21
  partial_derivative22 = d22 * S11
  partial_derivative11 = d11 * X
  partial_derivative21 = d21 * np.transpose(X)

  return {
    "pd12": partial_derivative12,
    "pd22": partial_derivative22,
    "pd11": partial_derivative11,
    "pd21": partial_derivative21
  }

def update_parameters(parameters, grads, learning_rate = 0.5):

    W11 = parameters["w1"][0]
    W21 = parameters["w1"][1]
    W12 = parameters["w2"][0]
    W22 = parameters["w2"][1]
    b1 = parameters["b1"]
    b2 = parameters["b2"]

    dW11 = grads["pd11"]
    dW21 = grads["pd21"]
    dW12 = grads["pd12"]
    dW22 = grads["pd22"]

    dB1 = 0
    dB2 = 0
    W11 = W11 - learning_rate * dW11
    W21 = W21 - learning_rate * dW21
    W12 = W12 - learning_rate * dW12
    W22 = W22 - learning_rate * dW22
    b1 = b1 - learning_rate * dB1
    b2 = b2 - learning_rate * dB2

    new_parameters = {"w1": np.array([W11,W21]),
                    "w2": np.array([W12,W22]),
                    "b1": b1,
                    "b2": b2}

    return new_parameters

if __name__ == '__main__':
  data = np.array([0.1, 0.90, 0.2, 0.4, 0.6])
  # expected_output = np.array([1, 0])
  expected_output = np.array([0, 1,0,1,1])
  output = None

  cache = None

  parameters = initialize_parameters(np.size(data), 2, 5)



  for i in range(5):
    output, cache = forward_propagation(data, parameters)
    grads = backward_propagation(parameters, cache, output, expected_output)
    parameters = update_parameters(parameters,grads)
    print(parameters)
print(output)
