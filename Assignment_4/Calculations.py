import numpy as np

def sigmoid(z):
    return 1/(1+math.exp(-z))

def initialize_weights(input_size, hidden_layer_size, output_size):
  #numpy random normal distribution
  W1 = np.random.normal(0, 1/np.sqrt(input_size), (hidden_layer_size, input_size))
  W2 = np.random.normal(0, 1/np.sqrt(hidden_layer_size), (output_size, hidden_layer_size))

  #weights dictionary  
  parameters = {"W1": W1, "W2": W2}
    
  return parameters
    
def computeNeuronSum(weights, data):
    return np.matmul(weights, data)



if __name__ == '__main__':
    data = np.array([156, 140, 180, 192, 172, 162])
    expected_output = np.array([1, 0, 1, 1, 0, 1])
    
    net_first_neuron_first_layer = computeNeuronSum(weights["W1"], data)
    print("Total net input of neuron 1 layer 1 :" + str(net_first_neuron_first_layer))
    net_snd_neuron_first_layer = computeNeuronSum(weights["W2"], data)
    print("Total net input of neuron 2 layer 1 :" + str(net_snd_neuron_first_layer))                                                            
