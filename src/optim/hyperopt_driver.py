from hyperopt import tpe, rand
from definitions.hyperopt_components import HyperoptEstimator, any_regressor, any_classifier
import numpy as np
from multiprocessing import freeze_support # to assist with some OpenBLAS threads hanging in HyperOpt
from definitions.hyperopt_components.components import random_forest_regression, svr_rbf, svr_poly, svr_sigmoid, knn_regression, gradient_boosting_regression
from definitions.hyperopt_components.components import random_forest, svc_rbf, svc_poly, svc_sigmoid, knn, gradient_boosting, ada_boost
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
import argparse
from sklearn import datasets
import matplotlib.pyplot as plt
import itertools


from data_set import mnist_data, neal_data, gz_data

# Convenience dictionaries for accessing ML algorithms for HPO
dict_regressors = { 0: any_regressor('rgs'),
                    1: random_forest_regression('rfr'),
                    2: svr_rbf('svrrbf'), 3: svr_sigmoid('svrsig'),
                    4: knn_regression('knn'),
                    5: gradient_boosting_regression('gbr') }

dict_classifiers = {0: any_classifier('clf'),
                    1: random_forest('rf'),
                    2: svc_rbf('svcrbf'),
                    3: svc_sigmoid('svcsig'),
                    4: knn('knn'),
                    5: gradient_boosting('gb')
                    }

def main():
    parser = argparse.ArgumentParser(description='Optimize hyperparameters for scikit-learn regression algorithms and perform model selection for given input-output data pairs')
    parser.add_argument('--algorithm_type', metavar='A', type=int, default=0, help='Integer to select type of regression/classification algorithm to optimize over, 0 is for any regressor/classifier, 1 is for SVM, and so on. Choices range from 0 to 6. Default is 0 (all regressors/classifiers)')
    parser.add_argument('--optimizer', metavar='OP', type=str, default='TPE', help='HP optimizer to use. Options are random (for random search) and TPE (for Hyperopt TPE). Default is TPE')
    parser.add_argument('--experiment_type', metavar='E', type=str, default='mnist', help='Experiment to run. Options are [mnist, gz, dark, toy, supernovae]. Default is mnist')
    parser.add_argument('--max_evals', metavar='MV', type=int, default=10, help='Number of evaluations to carry out. Default is 10')
    parser.add_argument('--data_location', metavar='DL', type=str, default='./', help='String indicating the path of the folder containing training.npy and solutions.npy. Default is current directory, or None for MNIST')
    parser.add_argument('--output_file', metavar='O', type=str, default='regressor_runs.p', help='Output file name. Default file is regressor_runs.p')
    args = parser.parse_args()

    f = open('classifier_info.data', 'w')
    algorithm = args.algorithm_type
    path = args.data_location
    experiment_type = args.experiment_type
    output_file = args.output_file
    optimizer = args.optimizer
    max_evals = args.max_evals

    if optimizer == 'random':
        optimizer = rand.suggest
    else:
        optimizer = tpe.suggest

    classification = False
    if experiment_type == 'mnist':
        data = mnist_data
        dictionary = dict_classifiers
        loss_function = accuracy_score
        classification = True
    elif experiment_type == 'galaxy':
        data = gz_data
        dictionary = dict_regressors
        loss_function = lambda x,y: np.sqrt(mean_squared_error(x,y))
    elif experiment_type == 'neal':
        data = neal_data
        dictionary = dict_classifiers
        loss_function = accuracy_score
        classification = True

    # intialize estimator from Hyperopt
    estim= HyperoptEstimator(
        preprocessing=(),
        #preprocessing=hpsklearn.components.any_preprocessing('pp'),
        regressor=dictionary[algorithm],
        algo=tpe.suggest,
        max_evals = max_evals,
        #loss_fn=loss_function,
        verbose=True,
        fit_increment_dump_filename=output_file,
        )

        # get dataset for experiment specified
    X_train, y_train, X_test, y_test = data(path)
    # train on training data
    estim.fit( X_train, y_train )
    # get predicitons
    predictions = estim.predict(X_test)

    # evaluate best model
    model_values = estim.best_model()
    model = model_values['learner']
    model.fit( X_train, y_train )
    predictions = model.predict(X_test)
    print('Loss: ' + str(loss_function(predictions, y_test)))
    if classification:
        print('confusion_matrix: ' + str(confusion_matrix(predictions, y_test)))
    # print and write to file
    print(str(model_values))
    f.write(str(model_values))
    f.write('\n')

    f.close()

    if classification:
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

def optimize(algorithm, data_location, optimizer='suggest', max_evals=10, output_file='results/optimization', do_plotting = False):
    '''
    Helper/Wrapper function around Hyperopt to run experiments using the 5 datasets above and to evaluate performance. Finally,

    algorithm_type: Integer to select type of regression/classification algorithm to optimize over, 0 is for any regressor/classifier, 1 is for SVM, and so on.
                    Choices range from 0 to 6. Default is 0 (all regressors/classifiers)
    optimizer: Optimizer for hyperopt to use. Options are random (for random search) and TPE (for Hyperopt TPE). Default is TPE
    experiment_type: Experiment to run. Options are [mnist, gz, dark, toy, supernovae]. Default is mnist
    max_evals: Number of evaluations to carry out. Default is 10
    data_location: String indicating the path of the folder containing training.npy and solutions.npy. Default is current directory, or None for MNIST
    output_file: Output file name. Default file is regressor_runs.p
    '''

    if algorithm == None:
        algorithm = 0

        # choice of optimizer between random and TPE
    if optimizer == 'random':
        optimizer = rand.suggest
    else:
        optimizer = tpe.suggest

    classification = False
    if experiment_type == 'mnist':
        data = mnist_data
        dictionary = dict_classifiers
        loss_function = accuracy_score # use the accuracy score for classification tasks
        classification = True
    elif experiment_type == 'galaxy':
        data = gz_data
        dictionary = dict_regressors
        loss_function = lambda x,y: np.sqrt(mean_squared_error(x,y)) # lambda function representing RMSE error. x and y are array-like objects
    elif experiment_type == 'neal':
        data = neal_data
        dictionary = dict_classifiers
        loss_function = accuracy_score
        classification = True

    # initialize the estimator object from Hyperopt
    estim= HyperoptEstimator(
        preprocessing=(),
        #preprocessing=hpsklearn.components.any_preprocessing('pp'),
        regressor=dictionary[algorithm],
        algo=optimizer,
        max_evals = max_evals,
        loss_fn=loss_function,
        verbose=True,
        fit_increment_dump_filename=output_file,
        )

    # retrieve the dataset specified for the experiment
    X_train, y_train, X_test, y_test = data(path)

    estim.fit( X_train, y_train )
    predictions = estim.predict(X_test)

    # score model on test data, print RMS
    model_values = estim.best_model()
    model = model_values['learner']
    model.fit( X_train, y_train )
    predictions = model.predict(X_test)
    print('Loss: ' + str(loss_function(predictions, y_test)))

    if classification:
        print('Confusion_matrix: ' + str(confusion_matrix(predictions, y_test)))

    # print and write to file the best model
    print(str(model_values))
    f.write(str(model_values))
    f.write('\n')

    f.close()

    if do_plotting:
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

    return estim

def rmse(predicted, true):
    return np.sqrt(mean_squared_error(predicted, true))

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    Parameters:
    -----------
    cm: confusion matrix return value computed by sklearn.metrics.confusion_matrix
    classes: labels for the classes as a Python list objectives
    normalize: Boolean parameter specifyhing whther or not to normalize histograms
    title: title to use for the plot generated
    cmap: colour map to use. should be a matplotlib.pyplot colour map object
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

if __name__ == "__main__":
    freeze_support()
    main()
