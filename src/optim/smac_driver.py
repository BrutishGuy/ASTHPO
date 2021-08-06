import logging
import os
import inspect

import numpy as np
from sklearn.metrics import make_scorer
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from data_set import *

from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter

from smac.tae.execute_func import ExecuteTAFuncDict
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC

from definitions.smac_components.components import ml, ml_cfg

def get_config_space():

    # Build Configuration Space which defines all parameters and their ranges
    cs = ConfigurationSpace()
    algorithm = CategoricalHyperparameter("algorithm", ["nn", "svm", "rf"], default="rf")

    do_bootstrapping = CategoricalHyperparameter("do_bootstrapping", ["true", "false"], default="true")
    num_trees = UniformIntegerHyperparameter("num_trees", 10, 50, default=10)
    max_features = UniformIntegerHyperparameter("max_features", 1, boston.data.shape[1], default=1)
    min_weight_frac_leaf = UniformFloatHyperparameter("min_weight_frac_leaf", 0.0, 0.5, default=0.0)
    criterion = CategoricalHyperparameter("criterion", ["mse", "mae"], default="mse")
    min_samples_to_split = UniformIntegerHyperparameter("min_samples_to_split", 2, 20, default=2)
    min_samples_in_leaf = UniformIntegerHyperparameter("min_samples_in_leaf", 1, 20, default=1)
    max_leaf_nodes = UniformIntegerHyperparameter("max_leaf_nodes", 10, 1000, default=100)

    use_do_bootstrapping = InCondition(child=do_bootstrapping, parent=algorithm, values=["rf"])
    use_num_trees = InCondition(child=num_trees, parent=algorithm, values=["rf"])
    use_max_features = InCondition(child=max_features, parent=algorithm, values=["rf"])
    use_min_wieght_frac_leaf = InCondition(child=min_weight_frac_leaf, parent=algorithm, values=["rf"])
    use_criterion = InCondition(child=criterion, parent=algorithm, values=["rf"])
    use_min_samples_in_leaf= InCondition(child=min_samples_in_leaf, parent=algorithm, values=["rf"])
    use_min_samples_to_split = InCondition(child=min_samples_to_split, parent=algorithm, values=["rf"])
    use_max_leaf_nodes = InCondition(child=max_leaf_nodes, parent=algorithm, values=["rf"])

    cs.add_hyperparameters([do_bootstrapping, num_trees, min_weight_frac_leaf, criterion,max_features, min_samples_to_split, min_samples_in_leaf, max_leaf_nodes])
    cs.add_conditions([use_criterion, use_num_trees,use_max_features, use_max_leaf_nodes, use_do_bootstrapping, use_min_samples_in_leaf, use_min_samples_to_split,use_min_wieght_frac_leaf])
    # We define a few possible types of SVM-kernels and add them as "kernel" to our cs
    kernel = CategoricalHyperparameter("kernel", ["linear", "rbf", "poly", "sigmoid"], default="poly")
    cs.add_hyperparameter(kernel)

    # There are some hyperparameters shared by all kernels
    C = UniformFloatHyperparameter("C", 0.001, 1000.0, default=1.0)
    use_C = InCondition(child=C, parent=algorithm, values=["svm"])
    shrinking = CategoricalHyperparameter("shrinking", ["true", "false"], default="true")
    use_shrinking = InCondition(child=shrinking, parent=algorithm, values=["svm"])
    cs.add_hyperparameters([C, shrinking])
    cs.add_conditions([use_C, use_shrinking])

    # Others are kernel-specific, so we can add conditions to limit the searchspace
    degree = UniformIntegerHyperparameter("degree", 1, 5, default=3)     # Only used by kernel poly
    coef0 = UniformFloatHyperparameter("coef0", 0.0, 10.0, default=0.0)  # poly, sigmoid
    cs.add_hyperparameters([degree, coef0])
    use_degree = InCondition(child=degree, parent=kernel, values=["poly"])
    use_coef0 = InCondition(child=coef0, parent=kernel, values=["poly", "sigmoid"])
    cs.add_conditions([use_degree, use_coef0])

    use_use_coef = InCondition(child=use_coef0, parent=algorithm, values=["svm"])
    use_use_degree = InCondition(child=use_degree, parent=algorithm, values=["svm"])
    cs.add_conditions([use_use_degree, use_use_coef0])
    # This also works for parameters that are a mix of categorical and values from a range of numbers
    # For example, gamma can be either "auto" or a fixed float
    gamma = CategoricalHyperparameter("gamma", ["auto", "value"], default="auto")  # only rbf, poly, sigmoid
    gamma_value = UniformFloatHyperparameter("gamma_value", 0.0001, 8, default=1)
    cs.add_hyperparameters([gamma, gamma_value])
    # We only activate gamma_value if gamma is set to "value"
    use_gamma_value = InCondition(child=gamma_value, parent=gamma, values=["value"])
    # And again we can restrict the use of gamma in general to the choice of the kernel
    use_gamma = InCondition(child=gamma, parent=kernel, values=["rbf", "poly", "sigmoid"])
    cs.add_conditions([use_gamma_value, use_gamma])
    use_use_gamma = InCondition(child=use_gamma, parent=algorithm, values=["svm"])
    use_use_gamma_value = InCondition(child=use_gamma_value, parent=algorithm, values=["svm"])
    cs.add_conditions([use_use_gamma_value, use_use_gamma])
    return cs

def main():
    parser = argparse.ArgumentParser(description='Optimize hyperparameters for scikit-learn regression algorithms and perform model selection for given input-output data pairs')
    parser.add_argument('--algorithm_type', metavar='A', type=int, default=0, help='Integer to select type of regression/classification algorithm to optimize over, 0 is for any regressor/classifier, 1 is for SVM, and so on. Choices range from 0 to 6. Default is 0 (all regressors/classifiers)')
    parser.add_argument('--experiment_type', metavar='E', type=str, default='mnist', help='Experiment to run. Options are [mnist, gz, dark, toy, supernovae]. Default is mnist')
    parser.add_argument('--max_evals', metavar='MV', type=int, default=10, help='Number of evaluations to carry out. Default is 10')
    parser.add_argument('--data_location', metavar='DL', type=str, default='./', help='String indicating the path of the folder containing training.npy and solutions.npy. Default is current directory, or None for MNIST')
    parser.add_argument('--output_file', metavar='O', type=str, default='regressor_runs.p', help='Output file name. Default file is regressor_runs.p')
    args = parser.parse_args()

    algorithm = args.algorithm_type
    path = args.data_location
    experiment_type = args.experiment_type
    output_file = args.output_file
    max_evals = args.max_evals

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

    #logger = logging.getLogger("SVMExample")
    logging.basicConfig(level=logging.INFO)  # logging.DEBUG for debug output
    cs = get_config_space()
    # Scenario object
    scenario = Scenario({"run_obj": "quality",   # we optimize quality (alternatively runtime)
                         "runcount-limit": max_evals,  # maximum function evaluations
                         "cs": cs,               # configuration space
                         "deterministic": "true"
                         })

    # Example call of the function
    # It returns: Status, Cost, Runtime, Additional Infos
    def_value = ml_cfg(cs.get_default_configuration())
    print("Default Value: %.2f" % (def_value))

    # Optimize, using a SMAC-object
    print("Optimizing! Depending on your machine, this might take a few minutes.")
    model = ml_cfg(data)
    smac = SMAC(scenario=scenario, rng=np.random.RandomState(42),
            tae_runner=model)

    incumbent = smac.optimize()

    inc_value = svm_from_cfg(incumbent)

    print("Optimized Value: %.2f" % (inc_value))

def optimize(algorithm, data_location, max_evals=10, output_file='results/optimization', do_plotting = False):
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

    #logger = logging.getLogger("SVMExample")
    logging.basicConfig(level=logging.INFO)  # logging.DEBUG for debug output
    cs = get_config_space()
    # Scenario object
    scenario = Scenario({"run_obj": "quality",   # we optimize quality (alternatively runtime)
                         "runcount-limit": max_evals,  # maximum function evaluations
                         "cs": cs,               # configuration space
                         "deterministic": "true"
                         })

    # Example call of the function
    # It returns: Status, Cost, Runtime, Additional Infos
    def_value = ml_cfg(cs.get_default_configuration())
    print("Default Value: %.2f" % (def_value))

    # Optimize, using a SMAC-object
    print("Optimizing! Depending on your machine, this might take a few minutes.")
    model = ml_cfg(data)
    smac = SMAC(scenario=scenario, rng=np.random.RandomState(42),
            tae_runner=model)

    incumbent = smac.optimize()

    inc_value = svm_from_cfg(incumbent)

    print("Optimized Value: %.2f" % (inc_value))

if __name__ == "__main__":
    main()
