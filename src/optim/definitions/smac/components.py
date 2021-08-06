import logging
import os
import inspect

import numpy as np
from sklearn.metrics import make_scorer
from sklearn.cross_validation import cross_val_score

import numpy as np
import sklearn.svm
import sklearn.ensemble
import sklearn.tree
import sklearn.neighbors
import sklearn.decomposition
import sklearn.preprocessing
import sklearn.neural_network
import sklearn.linear_model
import sklearn.discriminant_analysis
import sklearn.feature_extraction.text
import sklearn.naive_bayes
import sklearn.multiclass

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter

from smac.tae.execute_func import ExecuteTAFuncDict
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC

def rfr_cfg(cfg, seed):
    """
        Creates a random forest regressor from sklearn and fits the given data on it.
        This is the function-call we try to optimize. Chosen values are stored in
        the configuration (cfg).

        Parameters:
        -----------
        cfg: Configuration
            configuration chosen by smac
        seed: int or RandomState
            used to initialize the rf's random generator

        Returns:
        -----------
        np.mean(rmses): sklearn.trees.RandomForestRegressor
            hyperparameterized RandomForest model
    """
    rfr = sklearn.ensemble.RandomForestRegressor(
        n_estimators=cfg["num_trees"],
        criterion=cfg["criterion"],
        min_samples_split=cfg["min_samples_to_split"],
        min_samples_leaf=cfg["min_samples_in_leaf"],
        min_weight_fraction_leaf=cfg["min_weight_frac_leaf"],
        max_features=cfg["max_features"],
        max_leaf_nodes=cfg["max_leaf_nodes"],
        bootstrap=cfg["do_bootstrapping"],
        random_state=seed)

    return rfr

def nn_cfg(cfg):
    n_layers=cfg["num_layers"]
    n_hidden=cfg
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(784,)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])
    return model


def rfc_cfg(cfg, seed=0):
    """
        Creates a random forest regressor from sklearn. Chosen values are stored in
        the configuration (cfg).

        Parameters:
        -----------
        cfg: Configuration
            configuration chosen by smac
        seed: int or RandomState
            used to initialize the rf's random generator

        Returns:
        -----------
        np.mean(rmses): sklearn.trees.RandomForestRegressor
            hyperparameterized RandomForest model
    """
    rfc = sklearn.ensemble.RandomForestClassifier(
        n_estimators=cfg["num_trees"],
        criterion=cfg["criterion"],
        min_samples_split=cfg["min_samples_to_split"],
        min_samples_leaf=cfg["min_samples_in_leaf"],
        min_weight_fraction_leaf=cfg["min_weight_frac_leaf"],
        max_features=cfg["max_features"],
        max_leaf_nodes=cfg["max_leaf_nodes"],
        bootstrap=cfg["do_bootstrapping"],
        random_state=seed)

    return rfc

def svc_cfg(cfg):
    """ Creates a SVM based on a configuration

    Parameters:
    -----------
    cfg: Configuration (ConfigSpace.ConfigurationSpace.Configuration)
        Configuration containing the parameters.
        Configurations are indexable!

    Returns:
    --------
    A crossvalidated mean score for the svm on the loaded data-set.
    """
    # For deactivated parameters, the configuration stores None-values.
    # This is not accepted by the SVM, so we remove them.
    cfg = {k : cfg[k] for k in cfg if cfg[k]}
    # We translate boolean values:
    cfg["shrinking"] = True if cfg["shrinking"] == "true" else False
    # And for gamma, we set it to a fixed value or to "auto" (if used)
    if "gamma" in cfg:
        cfg["gamma"] = cfg["gamma_value"] if cfg["gamma"] == "value" else "auto"
        cfg.pop("gamma_value", None)  # Remove "gamma_value"

    clf = sklearn.svm.SVC(**cfg, random_state=42)

    return clf

def ml(data, cfg):
    X = data[0]
    y = data[1]
    
    algorithm = cfg["algorithm"]
    if algorithm == "svm":
        clf = svm_cfg(cfg)
    elif algorithm == "rf":
        clf = rf_cfg(cfg)
    else:
        clf = rf_cfg(cfg)

    scores = cross_val_score(clf, X, y, cv=5)
    return 1-np.mean(scores)  # Minimize!

from functools import partial, wraps

def ml_cfg(data):
    return (lambda cfg: ml(data, cfg))
