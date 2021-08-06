from __future__ import print_function
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop

from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np

def sinc_data():
    X = 10 * np.pi * np.random.random(200) - 5 * np.pi
    X = X[np.nonzero(X)]
    Y = np.sin(X)/X + 0.04 * X

    indeces = np.arange(len(X))
    np.random.shuffle(indeces)
    split = int(0.9 * len(indeces))
    X_train, Y_train, X_test, Y_test = X[indeces[:split]], Y[indeces[:split]], X[indeces[split:]], Y[indeces[split:]]
    return X_train, Y_train, X_test, Y_test

def mnist_data():
    '''
    Data providing function:

    This function is separated from model() so that hyperopt
    won't reload data for each evaluation run.
    '''
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    nb_classes = 10
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    return X_train[:30000], Y_train[:30000], X_test[:5000], Y_test[:5000]

def gz_tsne_data():
    '''
    Data providing function:

    This function is separated from model() so that hyperopt
    won't reload data for each evaluation run.
    '''
    X, Y = np.load('Data/gz_small/training.npy'), np.load('Data/gz_small/solutions.npy')
    # define an anonymous function that will split and return data after it has been loaded from the paths given above.
    indeces = np.arange(len(X))
    np.random.shuffle(indeces)
    split = int(0.9 * len(indeces))
    X_train, Y_train, X_test, Y_test = X[indeces[:split]], Y[indeces[:split]], X[indeces[split:]], Y[indeces[split:]]
    return X_train, Y_train, X_test, Y_test

def dark_matter_data():
    '''
    Data providing function:

    This function is separated from model() so that hyperopt
    won't reload data for each evaluation run.
    '''
    X, Y = np.load('Data/dark_matter/training_gal.npy'), np.load('Data/dark_matter/solutions.npy')
    # define an anonymous function that will split and return data after it has been loaded from the paths given above.
    indeces = np.arange(len(X))
    np.random.shuffle(indeces)
    split = int(0.9 * len(indeces))
    X_train, Y_train, X_test, Y_test = X[indeces[:split]], Y[indeces[:split]], X[indeces[split:]], Y[indeces[split:]]
    return X_train, Y_train, X_test, Y_test


def model(X_train, Y_train, X_test, Y_test):
    '''
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    '''
    model = Sequential()
    model.add(Dense({{choice([64, 128, 256, 512, 1024])}}, input_shape=(X_test.shape[1],)))
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense({{choice([64, 128, 256, 512, 1024])}}))
    model.add(Activation({{choice(['relu', 'sigmoid'])}}))
    model.add(Dropout({{uniform(0, 1)}}))

    # If we choose 'four', add an additional fourth layer
    if conditional({{choice(['three', 'four'])}}) == 'four':
        model.add(Dense(100))
        model.add({{choice([Dropout(0.5), Activation('linear')])}})
        model.add(Activation('relu'))

    model.add(Dense(10))
    model.add(Activation('sigmoid'))

    model.compile(loss='categorical_crossentropy',
                  optimizer={{choice(['rmsprop', 'adam', 'sgd'])}},
                  metrics=['accuracy'])

    model.fit(X_train, Y_train,
              batch_size={{choice([32, 64, 128])}},
              nb_epoch=10,
              verbose=2,
              validation_data=(X_test, Y_test))
    score, acc = model.evaluate(X_test, Y_test, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

import itertools
import matplotlib.pyplot as plt
def plot_confusion_matrix(cm, classes,
                      normalize=False,
                      title='Confusion matrix',
                      cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

if __name__ == '__main__':

    data = mnist_data

    X_train, Y_train, X_test, Y_test = data()
    trials_store = Trials()

    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=trials_store)

    predictions = best_model.predict(X_test)
    class_names = ['Digit 1','Digit 2','Digit 3', 'Digit 4', 'Digit 5', 'Digit 6', 'Digit 7' , 'Digit 8', 'Digit 9']
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, predictions)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')

    plt.show()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
