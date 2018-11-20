from random import seed
from random import random
import math

# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
	network = list()
	hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	network.append(output_layer)
	return network


# Calculate neuron activation for an input
def activate(weights, inputs):
	activation = weights[-1]
    #weighted sum
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation

# Transfer neuron activation
def transfer(activation):
    #sigmoid function
	return 1.0 / (1.0 + math.exp(-activation))

# Forward propagate input to a network output
def forward_propagate(network, data):
	inputs = data
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
            #neuron's output is the sigmoid applied to the weighted sum
			neuron['output'] = transfer(activation)
            #concatenation of all outputs, which will become inputs
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs

# test forward propagation
#network = [[{'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
#		[{'weights': [0.2550690257394217, 0.49543508709194095]}, {'weights': [0.4494910647887381, 0.651592972722763]}]]
#row = [1, 0, None]
#output = forward_propagate(network, row)
#print(output)



# Calculate the derivative of an neuron output
def transfer_derivative(output):
	return output * (1.0 - output)



# test backpropagation of error
#network = [[{'output': 0.7105668883115941, 'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
#		[{'output': 0.6213859615555266, 'weights': [0.2550690257394217, 0.49543508709194095]}, {'output': 0.6573693455986976, 'weights': [0.4494910647887381, 0.651592972722763]}]]
#expected = [0, 1]
#backward_propagate_error(network, expected)
#for layer in network:
#	print(layer)

# Calculate the derivative of an neuron output
def transfer_derivative(output):
	return output * (1.0 - output)

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
    #going back to forward
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
        #if this is not the output layer
		if i != len(network)-1:
            #for each neuron of the considering layer
			for j in range(len(layer)):
				error = 0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['error'])
                #storing neuron error in errors list
                errors.append(error)
		else:
            #if this is the output layer we are computing the error with the
            #expected output
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(expected[j] - neuron['output'])
		for j in range(len(layer)):
			neuron = layer[j]
            #update of the neuron error for each of them
			neuron['error'] = errors[j] * transfer_derivative(neuron['output'])

# test backpropagation of error
#network = [[{'output': 0.7105668883115941, 'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
#		[{'output': 0.6213859615555266, 'weights': [0.2550690257394217, 0.49543508709194095]}, {'output': 0.6573693455986976, 'weights': [0.4494910647887381, 0.651592972722763]}]]
#expected = [0, 1]
#backward_propagate_error(network, expected)
#for layer in network:
#	print(layer)

# Update network weights with error
def update_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] += l_rate * neuron['error'] * inputs[j]
			neuron['weights'][-1] += l_rate * neuron['error']

            
# Make a prediction with a network
def predict(network, row):
	outputs = forward_propagate(network, row)
	return outputs.index(max(outputs))

# Test making predictions with the network
dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]
network = [[{'weights': [-1.482313569067226, 1.8308790073202204, 1.078381922048799]}, {'weights': [0.23244990332399884, 0.3621998343835864, 0.40289821191094327]}],
	[{'weights': [2.5001872433501404, 0.7887233511355132, -1.1026649757805829]}, {'weights': [-2.429350576245497, 0.8357651039198697, 1.0699217181280656]}]]
for row in dataset:
	prediction = predict(network, row)
	print('Expected=%d, Got=%d' % (row[-1], prediction))
