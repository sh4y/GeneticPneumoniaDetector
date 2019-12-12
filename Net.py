from sklearn.neural_network import MLPClassifier
import numpy as np
import random

import Data

SOLVERS = ['lbfgs', 'sgd', 'adam']
ACTIVATION = ['identity', 'logistic', 'tanh', 'relu']
MAX_HIDDEN_LAYERS = 10
MAX_NEURONS_PER_LAYER = 15

INITIAL_NUMBER_NEURAL_NETS = 5

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
        
print('Loading + padding train data.')
xtrain, ytrain = Data.load_data('train', 50, 50)
print("Loaded xtrain with shape {0} and ytrain with shape {1}".format(xtrain.shape, ytrain.shape))
classifiers = initialize_nn_population()
train_classifiers(classifiers, xtrain, ytrain)
