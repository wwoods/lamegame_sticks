
import copy
import math
import numpy as np
import random

class Neuron(object):
    def __init__(self, weights):
        """Weights - initial scalars for previous layer as input to this
        neuron.
        """
        self.weights = np.asarray(weights)
        self.weights_doc = """Array of length len(previousLayer.neurons).  The
                scalar for each input from that layer."""
        self.deltas = np.zeros(len(weights))
        self.deltas_doc = """Array of deltas for last training session"""
        self.age = 0
        self.age_doc = """Count of training cycles endured"""


    def absorb(self, prevLayerNeurons, prevLayerInputs):
        """Absorbs all neurons in the previous layer, as they will be removed.
        Our input count will become prevLayerInputs.
        """
        newWeights = []
        for j in range(prevLayerInputs):
            wt = 0.0
            for i, pn in enumerate(prevLayerNeurons):
                wt += pn.weights[j] * self.weights[i]
            newWeights.append(wt)
        self.weights = np.asarray(newWeights)
        self.deltas = np.zeros(len(self.weights))


class NeuronLayer(object):

    kernel = math.tanh
    kernelD = lambda self, y: 1.0 - y*y
    kernelD_doc = """Derivative of kernel for a given Y"""

    def __init__(self):
        """Standard neuron layer - sigmoidal via tanh."""
        self._neurons = []
        self._inputs = 0


    def __len__(self):
        return len(self._neurons)


    def _randomWeight(self):
        return random.random() * 2 - 1


    def absorbLayer(self, prevLayer):
        """Called when prevLayer is being removed and we need to absorb its
        weights and adjust our input count to their input count.
        """
        for n in self._neurons:
            n.absorb(prevLayer._neurons, prevLayer._inputs)
        self._inputs = prevLayer._inputs


    def addInput(self):
        """Means that we are adding a new input at the tail.
        """
        self._inputs += 1
        for n in self._neurons:
            n.weights = np.append(n.weights, self._randomWeight())
            n.deltas = np.append(n.deltas, 0.0)


    def addNeuron(self):
        wts = []
        if self._inputs > 0:
            totals = [ 0.0 ] * self._inputs
            for n in self._neurons:
                for i, wt in enumerate(n.weights):
                    wtSqr = wt * wt
                    totals[i] += wtSqr
            maxTotal = max(totals)
            for i in range(self._inputs):
                iDiff = math.sqrt(maxTotal - totals[i])
                # iDiff is now the amount of this input NOT being used at
                # this layer, so try to use it a little bit more in a new
                # neuron.  Turns out maybe this is a bad idea, turned off
                scale = 1.0 + iDiff * 0.0
                wts.append(self._randomWeight() * scale)
        self._neurons.append(Neuron(wts))


    @classmethod
    def directTransfer(cls, prevLayer):
        """Returns an instance of NeuronLayer whose neurons directly transfer
        the result from the previous layer.
        """
        layer = cls()
        layer._inputs = len(prevLayer._neurons)
        for i, n in enumerate(prevLayer._neurons):
            wts = [0.0] * layer._inputs
            wts[i] = 1.0
            layer._neurons.append(Neuron(wts))
        return layer


    def eval(self, inputs):
        """Produces an array for the next row...
        """
        outputs = np.asarray([ self.kernel(inputs.dot(n.weights))
                for n in self._neurons ])
        self._lastOutputs = outputs
        return outputs


    def getLastOutputs(self):
        return self._lastOutputs


    def mutate(self, nextLayer, rate):
        # nextLayer is None means we're the output layer
        partLeft = random.random()
        if nextLayer is not None and partLeft <= 0.20:
            # Add a neuron!
            self.addNeuron()
            nextLayer.addInput()
        elif (nextLayer is not None 
                and len(self._neurons) > 1
                and partLeft <= 0.35):
            # Remove a neuron!
            n = random.randint(0, len(self._neurons) - 1)
            self._neurons.pop(n)
            nextLayer.removeInput(n)
        elif partLeft <= 0.45:
            # Cut a neuron's age in half (it'll learn faster)
            random.choice(self._neurons).age = 0
        elif partLeft <= 0.80:
            # Minor neuron disturbance
            if random.random() < 0.4:
                # Swap!
                n = random.choice(self._neurons)
                w1 = random.randint(0, len(n.weights) - 1)
                w2 = random.randint(0, len(n.weights) - 1)
                t = n.weights[w1]
                n.weights[w1] = n.weights[w2]
                n.weights[w2] = t * (1.0 + random.random() - 0.5)
            else:
                # Mutate!
                # By which we mean shut someone down
                n = random.choice(self._neurons)
                for i in xrange(len(n.weights)):
                    if random.random() < 0.75:
                        n.weights[i] = 0.0
                n.weights[random.randint(0, len(n.weights) - 1)] = \
                        random.random() * 10 - 5.0
                #w = random.randint(0, len(n.weights) - 1)
                #n.weights[w] *= 1.0 + random.random() - 0.5
        else:
            # Complete neuron reset
            n = random.choice(self._neurons)
            for i in range(len(n.weights)):
                n.weights[i] = self._randomWeight()


    def train(self, lastInputs, outputDeltas, outputWeights, rate):
        """Trains this layer according to needed changes in the next layer's
        inputs and the weights of each of this layer's nodes on each of the
        next layer's output.

        Returns outputDeltas and outputWeights for the row before this one.
        """
        myOldInputs = np.array([ n.weights for n in self._neurons ])
        inputDeltas = np.zeros(len(self._neurons))

        for i, n in enumerate(self._neurons):
            # Take my part in all of the outputs, summed.  This is about how
            # much I need to change.
            myPart = outputDeltas.dot(outputWeights[:, i])
            inputDeltas[i] = myPart

            # Use my derivative (combined with age) to transform myPart into
            # how much my inputs need to change, approximately.  Except the main
            # purpose is confidence, so if the derivative is low, keep how much
            # we move low.
            myD = self.kernelD(self._lastOutputs[i])
            a = 40.0
            myD += 0.5 * a / (a + n.age)

            # Compare how I take input to my inputs; adjust weights by rate
            lastEffects = n.weights * lastInputs
            # Grow those that point towards myPart, shrink others away.  The
            # larger a weight is, the slower it should change.
            weightChangeRate = np.abs(n.weights) + 1.0
            n.weights += weightChangeRate * (rate * myD * myPart)
            n.age += 1

        return inputDeltas, myOldInputs

        ##OLD METHOD; please merge docs before deleting
        # Calculate delta at each of our neurons
        for i, n in enumerate(self._neurons):
            for j, v in enumerate(outputDeltas):
                inputDeltas[i] += v * outputWeights[j][i]
            # Each of our neurons has an effect on the next layer, so divide
            # by our neuron count
            # False - this seems to make it slower.
            #inputDeltas[i] *= 1.0 / len(self._neurons)

            outputDerivative = self.kernelD(self._lastOutputs[i])
            # Add some bias - greatly improves learning rate, with the downside
            # that we must "age" our neurons to get them to even out at good
            # values
            a = 40.0
            outputDerivative += 0.5 * a / (a + n.age)
            b = 5000000.0
            change = rate * inputDeltas[i] * outputDerivative * (
                    b / (b + n.age))
            for j, w in enumerate(n.weights):
                # Different direction values:
                ma = 0.4  # Momentum of new input
                m = 0.35  # Momentum of last change
                newChange = change * lastInputs[j]
                if newChange * n.deltas[j] >= 0.0:
                    # Same direction; this actually means MORE momentum since
                    # we are confirming our opinion
                    ma = 1.0
                    m = 0.35

                n.deltas[j] = (ma * newChange + n.deltas[j] * m)
                n.weights[j] += n.deltas[j]

            # Age the neuron!
            n.age += 1


        return inputDeltas, myOldInputs


    def removeInput(self, index):
        """Removes an input to this layer (with the specified index)"""
        for n in self._neurons:
            n.weights = np.delete(n.weights, index)
            n.deltas = np.delete(n.deltas, index)
        self._inputs -= 1


class LinearLayer(NeuronLayer):
    kernel = lambda self, x: x
    kernelD = lambda self, y: 1.0 / (1.0 + y ** 2)


class InputLayer(NeuronLayer):
    """Input layer - rather than a weight for every input, each weights are a
    set of scale / offset.
    """

    def __init__(self):
        NeuronLayer.__init__(self)
        self._neurons = np.ndarray((0, 2), dtype = np.float64)


    def addInput(self):
        raise NotImplementedError()


    def addNeuron(self):
        oldSize = self._neurons.shape[0]
        self._neurons.resize((oldSize + 1, 2))
        self._neurons[oldSize] = [ 1.0, 0.0 ]


    def eval(self, inputs):
        outputs = np.asarray([ n[0] * inputs[i] + n[1]
                for i, n in enumerate(self._neurons) ])
        self._lastOutputs = outputs
        return outputs


    def mutate(self, nextLayer, rate):
        pass


    def train(self, lastInputs, outputDeltas,
            outputWeights, rate):
        # Skip input training for now.....
        return [], []


class NeuralNet(object):
    def __init__(self):
        """Initializes a brand new neural network with the specified number
        of nodes.  Note that the point here is that the structure will evolve
        though, so these shouldn't matter...
        """
        self._layers = []


    def _init(self):
        """Sets up a 2-layer, no neuron network.
        """
        self._layers = [ InputLayer(), NeuronLayer() ]


    def _addNodeToLayer(self, layerIndex):
        """Adds a node to the specified layer.  If called with layer 0, it
        means we are adding an input to the system.
        """
        self._layers[layerIndex].addNeuron()
        if layerIndex + 1 < len(self._layers):
            self._layers[layerIndex + 1].addInput()


    def copy(self):
        return copy.deepcopy(self)


    def eval(self, values, outputCount):
        """Takes an array of values and a count of expected outputs, and 
        initializes the neural network if needed.  Evaluates with the given
        values.

        Returns an array of outputs.
        """
        if len(self._layers) < 2:
            self._init()
        while len(self._layers[0]) < len(values):
            self._addNodeToLayer(0)
        while len(self._layers[-1]) < outputCount:
            self._addNodeToLayer(len(self._layers) - 1)

        output = values
        for layer in self._layers:
            output = layer.eval(output)
        return output


    def mutate(self, rate):
        """Performs a mutation according to a distribution that varies based
        on severity.  rate is analogous to train() rate
        """
        if random.random() < max(0.005, 0.05 / (len(self._layers))):
            # Add a layer!
            layerToDupe = random.randint(0, len(self._layers) - 2)
            layer = NeuronLayer.directTransfer(
                    self._layers[layerToDupe])
            self._layers.insert(layerToDupe + 1, layer)
            layer.mutate(self._layers[layerToDupe + 2], rate)
        elif len(self._layers) > 2 and random.random() < 0.025:
            # Remove a layer!  Other than the input or output, of course
            layerToAxe = random.randint(1, len(self._layers) - 2)
            self._layers[layerToAxe + 1].absorbLayer(self._layers[layerToAxe])
            self._layers.pop(layerToAxe)
        else:
            # Random layer mutation!
            # Note that we don't consider input or output layers as valid for
            # point mutation, since they're not structural
            if len(self._layers) <= 2 or random.random() < 0.45:
                # Modify output layer; less frequent since it's not a 
                # structural modification, and weights can be updated by
                # backpropagation.
                layerToMutate = len(self._layers) - 1
            else:
                layerToMutate = random.randint(1, len(self._layers) - 2)
            nextLayer = None
            if layerToMutate + 1 < len(self._layers):
                nextLayer = self._layers[layerToMutate + 1]
            self._layers[layerToMutate].mutate(nextLayer, rate)


    def train(self, values, outputs, rate = 0.01):
        """Takes an array of values and outputs and trains the network
        via backpropagation.  Also updates self.confidence.

        If rate is 0, do not update the network.

        outputs -- Desired outputs

        Returns the sum of squares error of the original output with the
        expected.
        """
        outputs = np.asarray(outputs)
        realOutputs = self.eval(values, outputs.shape[0])

        # Squared error with original output
        beforeError = np.sum((outputs - realOutputs) ** 2)

        if rate > 0:
            inputDeltas = outputs - realOutputs
            layerWeights = np.eye(outputs.shape[0], outputs.shape[0])

            for i in range(len(self._layers) - 1, -1, -1):
                layer = self._layers[i]
                if i > 0:
                    lastInputs = self._layers[i - 1].getLastOutputs()
                else:
                    lastInputs = values
                inputDeltas, layerWeights = layer.train(
                        lastInputs = lastInputs,
                        outputDeltas = inputDeltas,
                        outputWeights = layerWeights,
                        rate = rate)

        return beforeError

