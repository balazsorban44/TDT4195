import numpy as np


class Neural_Network():

    layer_count = 0
    shape = None
    weights = []

    #init
    def __init__(self, layer):

        #we don't include the input layer
        self.layer_count = len(layer) - 1
        self.shape = layer
        self._layer_input = []
        self._layer_output = []

        #weights initialization
        #zip : creates a couple (layer 1,layer 2) between which will be the weight
        #ie: have to consider from 1st layer to the penultimate layer and from
        #the 2nd to the last
        for (layer1, layer2) in zip(layer[:-1],layer[1:]):
            #adding of random weights
            #output shape : layer2 * layer1 + 1 (+1 for the bias)
            self.weights.append(np.random.normal(scale=0.15,size = (layer2, layer1+1)))

    def run(self, input):
        print(self.forwardPropagation(input))

    def sigmoid(self, beforeAct, derivative = False):
    #activation function
        #sigmoid function
        if not derivative:
            return 1/(1+np.exp(-beforeAct))
        else:
            #derivative formula
            output = self.sigmoid(beforeAct)
            return output*(1 - output)

    def forwardPropagation(self, input):
        number_of_input = input.shape[0]
        self._layer_input = []
        self._layer_output = []
        for i in range(self.layer_count):
            #if 1st layer (input layer)
            if i == 0:
                #self.weights[0] : weights between the first and the second (hidden) layer
                #product between the corresponding weights and the transposed input matrix
                #adding ones vector : for the bias addition after the weighted sum
                layerinput = self.weights[0].dot(np.vstack([input.T, np.ones([1,number_of_input])]))

            else:
                #same with taking the output of the previous layer (-1 takes the last one)
                #each time we will have a output vector for each input (ones of size of number of inputs)

                layerinput = self.weights[i].dot(np.vstack([self._layer_output[-1], np.ones([1,number_of_input])]))
            #storing the layer input (vector because for each neuron): before activation
            self._layer_input.append(layerinput)
            #storing the output for each neuron (after activation --> sigmoid)
            self._layer_output.append(self.sigmoid(layerinput))

        return self._layer_output[-1].T


    def backpropagation(self, input, output, learning_rate = 0.2):
        len_input = input.shape[0]
        #we need the ouput --> forward propagation
        fwd_prog = self.forwardPropagation(input)
        #delta calculation
        delta = []
        #we are going from the end to the beginning of the neural network
        for i in reversed(range(self.layer_count)):
            #for the penultimate layer
            if i == self.layer_count -1:
                #obtained output from forward propagation minus expected output
                #output is stored in columns vectors --> transpose of the expected output
                diff = self._layer_output[i] - output.T
                #total squared error
                error = np.sum(diff**2)
                #difference times the derivative of pre activation value of the layer
                #-->delta for the output layer
                delta_k = diff * (self.sigmoid(self._layer_input[i], True))
                #print(delta_k)
                #storing of delta_k
                delta.append(delta_k)
            else:
                #hidden layer needs the product of the previous layer’s delta
                #with the previous layer’s weights (previous which is actually the next one in fw)
                #chain rule : to obtain the derivative in function of a weight
                #we have to multiply previous delta and
                #d(inputvalue before activation)/d(weight) = weight
                delta_j = self.weights[i+1].T.dot(delta[-1])
                #without the last line which corresponds to the bias
                #sigmoid derivative corresponds to:
                #d(outputHiddenLayer)/d(inputHiddenLayer) =
                #d(outputAfterActivation)/d(inputBeforeActivation)
                #--> product with the gradient of the act function evaluated at the current layer
                delta.append(delta_j[:-1,:] * self.sigmoid(self._layer_input[i], True))

        #weights calculation
        #for all the layers we update the weights
        for i in range(self.layer_count):
            #to update weights between 2 layers we need to use the output of
            #the first one and the deltas of 2nd layer, ie: we have to store both corresponding indices

            delta_i = self.layer_count -1 - i
            #we have to get the output for each layer :
            #i = 0 input layer (even if layer 0 is actually the first hidden layer in self._layer_output)
            if i == 0:
                #with adding the line for biases
                #with vstack all the input will be in the same matrix of the array
                layeroutput = np.vstack([input.T, np.ones([1, len_input])])
            else:
                #the layer output with bias is the output of considering layer
                #with adding bias with corresponding size
                #i-1 because i=0 is the input layer, and in self._layer_output
                #we begin the indices at 0 : first hidden layer
                layeroutput = np.vstack([self._layer_output[i-1], np.ones([1, self._layer_output[i-1].shape[1]])])

            #we want the layer output of the first layer * the delta of the 2nd
            #[None,:,:] : first dimension empty and 1st and 2nd with rows and columns of the data
            #with transpose, we have a matrix where inputs are separatly stored
            #same thing with deltas, because we want to multiply the output of the layer 1 with
            #the delta of layer 2 for each input (ie: we have to sort it by inputs!)
            #we put other axis to enable the matrix product

            weight_delta = np.sum(layeroutput[None,:,:].transpose(2,0,1) * delta[delta_i][None,:,:].transpose(2,1,0), axis=0)

            #update of the weights of layer i
            self.weights[i] -= learning_rate * weight_delta

        return error

if __name__ == '__main__':
    #input with number of neurons in each layer
    A = (3,5,2,1)
    net = Neural_Network(A)
    #print(bpn.weights[1])

    def hexToNormRGB(hex):
        return(list(int(hex[1:][i:i+2], 16)/255 for i in (0, 2 ,4)))

    input = np.array([
        hexToNormRGB("#00ff00"),
        hexToNormRGB("#00ffff"),
        hexToNormRGB("#ff0000"),
        hexToNormRGB("#f3fefe"),
        hexToNormRGB("#eeeeee"),
        hexToNormRGB("#e7a6cd"),
        hexToNormRGB("#000000"),
        hexToNormRGB("#dedecd"),
        hexToNormRGB("#4444ee"),
        hexToNormRGB("#ede213")
    ])

    #print(np.vstack([input.T, np.ones([1,input.shape[0]])]))
    #output = bpn.forwardPropagation(input)
    #print(output)
    expected_output = np.array([
      1,1,0,0,0,0,1,0,1,0
    ])

    max_iteration = 200000
    min_error = 1e-3
    for i in range(max_iteration):
        err = net.backpropagation(input,expected_output)
        if i%1000 == 0:
            print("Iteration {0}\t error: {1:0.8f} ".format(i,err), end="\r")

        #if we reach the objective: out of the for
        if err <= min_error:
            print("Minimum error reached at iteration {0}".format(i))
            break

    test = np.array([
        hexToNormRGB("#ffffff"),
        hexToNormRGB("#ed64fe"),
        hexToNormRGB("#cdade0"),
        hexToNormRGB("#599567"),
    ])
    # should return 0, 0, 0, 1
    net.run(test)
