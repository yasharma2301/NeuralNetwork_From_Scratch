import random
from random import seed
import math
random.seed(1)

# ----------------- STEP 1: -----------------
# Network initialisation:
def makeNetwork(inputs,neuronsCount,output):
    hiddenLayer = [{'weights':[random.random() for i in range(inputs+1)]} for i in range(neuronsCount)]
    outputLayer = [{'weights':[random.random() for i in range(neuronsCount+1)]} for i in range(output)]
    neuralNetwork = []
    neuralNetwork.append(hiddenLayer)
    neuralNetwork.append(outputLayer)
    return neuralNetwork

# ----------------- STEP 2: -----------------
# Forward Propogation
    # - Neuron Activation
    # - Neuron Transfer
    # - Neuron Propagation

# - Neuron Activation. Formula, activation = Σ(weight*input)+bias.    
def activate(weights, inputs):
    # bias
	activation = weights[-1]
    # weight*input
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation

# - Neuron Transfer. Here, sigmoid function.
def transfer(activation):
	return 1.0 / (1 + math.exp(-activation))

# - Neuron Propagation
def forwardPropagate(neuralNetwork, dataRow):
	inputs = dataRow
	for layer in neuralNetwork:
		inputForNextLayer = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
			inputForNextLayer.append(neuron['output'])
		inputs = inputForNextLayer
	return inputs

# ----------------- STEP 3: -----------------
# Back Propagation
    # - Transfer Derivative
    # - Error Backpropagation

# - Transfer Derivative OR simply derivative(sigmoid)
def transferDerivative(output):
    return output*(1.0-output)

# - Error Backpropagation
    # - Error for each neuron in outlayer: error = (expected - output) * transferDerivative(output)
    # - Error for neurons in hidden layer: error = (weight-i * error-j) * transferDerivative(output)

def backPropagateError(neuralNetwork, expected):
    for i in range(len(neuralNetwork)-1,-1,-1):
        currentLayer = neuralNetwork[i]
        errors = []

        if i == len(neuralNetwork)-1:
            for j in range(len(currentLayer)):
                neuron = currentLayer[j]
                errors.append(expected[j] - neuron['output'])
        else:
            for j in range(len(currentLayer)):
                error = 0
                for neuron in neuralNetwork[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)

        for j in range(len(currentLayer)):
            neuron = currentLayer[j]
            neuron['delta'] = errors[j] * transferDerivative(neuron['output']) 


# ----------------- STEP 4: -----------------
# Training the network
    # - Update Weights
    # - Train Netwwork
# weightNew = weightOld + learning_rate * error * input

# - Update Weights
def updateWts(neuralNetwork, dataRow, learningRate):
	for i in range(len(neuralNetwork)):
		inputs = dataRow[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in neuralNetwork[i - 1]]
		for neuron in neuralNetwork[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] += learningRate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] += learningRate * neuron['delta']

# - Train Netwwork
def trainTheNetwork(neuralNetwork,trainData,learningRate,epochs,outputsCount):
    print('---> learning rate=%.4f' % learningRate)
    for epoch in range(epochs):
        totalError = 0
        for eachRow in trainData:
            outputs = forwardPropagate(neuralNetwork,eachRow)
            # encode data
            expectedValues = [0 for i in range(outputsCount)]
            expectedValues[eachRow[-1]]=1

            # error calculation for one row of data
            totalError+=sum([(expectedValues[i]-outputs[i])**2 for i in range(len(expectedValues))])
            
            # backpropagate and update weights
            backPropagateError(neuralNetwork,expectedValues)
            updateWts(neuralNetwork,eachRow,learningRate)

        print('---> epoch=%d, error=%.4f' % (epoch, totalError))


# ----------------- STEP 5: -----------------
# Predictions

def predict(neuralNetwork, row):
	outputs = forwardPropagate(neuralNetwork, row)
    # select the class with max probability or in other words arg max function
	return outputs.index(max(outputs))


from csv import reader
# ----------------- STEP 6: -----------------
# Testing our algorithm with a dataset:

# load CSV
def loadCSVfile(filename):
    data = []
    with open(filename,'r') as file:
        csv_reader = reader(file)
        for eachRow in csv_reader:
            if not eachRow:
                continue
            else:
                data.append(eachRow)
    return data

# string to float
def strColumnToFloat(data,column):
    for eachRow in data:
        eachRow[column] = float(eachRow[column].strip())

# string to int
def strColumnToInt(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	dic = {}
	for i, value in enumerate(unique):
		dic[value] = i
	for row in dataset:
		row[column] = dic[row[column]]
	return dic

# min max stats from data
def minMaxFromData(data):
    stats = [[min(column), max(column)] for column in zip(*data)]
    return stats

# normalizeData
def normalizeDataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)-1):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

# k-fold cross validation
def crossValidation(data,folds):
    dataSplit = []
    dataCopy = list(data)
    foldSize = int(len(data)/folds)
    for _ in range(folds):
        fold = []
        while len(fold)<foldSize:
            index = random.randrange(len(dataCopy))
            fold.append(dataCopy.pop(index))
        dataSplit.append(fold)
    return dataSplit

# accuracy score
def accuracyMetric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100

# run algo using cross validation
def runAlgorithm(data,algorithm,foldCount,*args):
    folds = crossValidation(data, foldCount)
    scores = []
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = []
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracyMetric(actual, predicted)
        scores.append(accuracy)
    return scores

# Wrapper function of neural network
def backPropagation(train, test, learningRate, numberOfEpochs, numberOfHiddenLayers):
	numberOfInputs = len(train[0]) - 1
	numberOfOutputs = len(set([row[-1] for row in train]))
	neuralNetwork = makeNetwork(numberOfInputs, numberOfHiddenLayers, numberOfOutputs)
	trainTheNetwork(neuralNetwork, train, learningRate, numberOfEpochs, numberOfOutputs)
	predictions = []
	for row in test:
		prediction = predict(neuralNetwork, row)
		predictions.append(prediction)
	return predictions

# loading the CSV
filename = 'seeds_dataset.csv'
dataset = loadCSVfile(filename)

# convert some colums to floats
for i in range(len(dataset[0])-1):
	strColumnToFloat(dataset, i)

# convert some columns to integers
strColumnToInt(dataset, len(dataset[0])-1)

# normalize the data
minmax = minMaxFromData(dataset)
normalizeDataset(dataset, minmax)

# evaluate algorithm
numberOfFolds = 5
learningRate = 0.3
numberOfEpochs = 500
numberOfHiddenLayers = 5

scores = runAlgorithm(dataset, backPropagation, numberOfFolds, learningRate, numberOfEpochs, numberOfHiddenLayers)
print('Scores List: %s' % scores)
print('Accuracy mean: %.4f%%' % (sum(scores)/float(len(scores))))







