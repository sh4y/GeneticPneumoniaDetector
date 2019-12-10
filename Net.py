from sklearn.neural_network import MLPClassifier
import numpy as np
import random

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
    classifier = MLPClassifier(hidden_layer_sizes=hidden_layer, max_iter = 300, activation=activator, solver=solver)
    return classifier

def initialize_nn_population():
    classifiers = []
    for x in range(INITIAL_NUMBER_NEURAL_NETS):
        classifier = create_random_nn()
        classifiers.append(classifier)

    return classifiers