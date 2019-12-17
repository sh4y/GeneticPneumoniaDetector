from sklearn.neural_network import MLPClassifier
import numpy as np
import random

import Data

SOLVERS = ['lbfgs', 'sgd', 'adam']
ACTIVATION = ['identity', 'logistic', 'tanh', 'relu']
MAX_HIDDEN_LAYERS = 3
MAX_NEURONS_PER_LAYER = 4

INITIAL_NUMBER_NEURAL_NETS = 1

def create_hidden_layer_properties():
    layers = list(np.ones(random.randint(1, MAX_HIDDEN_LAYERS)))
    for i in range(len(layers)):
        layers[i] = random.randint(1, MAX_NEURONS_PER_LAYER)
    return tuple(layers)

def create_random_nn():
    hidden_layer = create_hidden_layer_properties()
    activator = random.choice(ACTIVATION)
    solver = random.choice(SOLVERS)
    print("Created classifier with hidden layer sizes of {0}, {1} as solver, and {2} as activation function.".format(str(hidden_layer), solver, activator))
    classifier = MLPClassifier(hidden_layer_sizes=hidden_layer, max_iter = 300, activation=activator, solver=solver)
    return classifier

def initialize_nn_population(n=INITIAL_NUMBER_NEURAL_NETS):
    print("Creating initial neural net population of {0}".format(n))
    classifiers = []
    for x in range(n):
        classifier = create_random_nn()
        classifiers.append(classifier)

    return classifiers

def train_classifiers(classifiers, xtrain, ytrain):
    for x in range(len(classifiers)):
        print("Training classifier number " + str(x))
        classifiers[x].fit(xtrain, ytrain)

def test_classifiers(classifiers, xtest, ytest):
    for x in range(len(classifiers)):
        print("Testing classifier number " + str(x))
        prediction = classifiers[x].predict(xtest)
        test_accuracy = np.sum(prediction == ytest.flatten()) / float(prediction.shape[0])
        print(test_accuracy)


xtrain,ytrain,xtest,ytest = Data.load_data(50, 50)
print("Loaded xtrain, ytrain with {0} points of data".format(xtrain.shape))

classifiers = initialize_nn_population()
train_classifiers(classifiers, xtrain, ytrain)

#xtest,ytest = Data.load_data('test', 50, 50)
test_classifiers(classifiers, xtest, ytest)