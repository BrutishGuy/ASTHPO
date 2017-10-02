# ASTHPO

Hyperparameter Optimization for Astronomy Domain Problems

# Overview

ASTHPO is a tool for combined algorithm selection and hyperparameter optimization (CASH) of machine learning algorithms
with specific application to solving problems in the domain area of astronomy.
The main core consists of Bayesian Optimization (TPE, SMAC variants) but also uses implementations of Hyperband and random search.
Data efficient techniques such as Fabolas and MTBO are also featured.

For a detailed description of of the algorithms used in this tool, one is referred
to the following papers for details:

    Hutter, F. and Hoos, H. H. and Leyton-Brown, K.
    Sequential Model-Based Optimization for General Algorithm Configuration
    In: Proceedings of the conference on Learning and Intelligent OptimizatioN (LION 5)

    Bergstra, J., Yamins, D., & Cox, D. D. (2013).
    Hyperopt: A python library for optimizing the hyperparameters of machine learning algorithms.
    In Proceedings of the 12th Python in Science Conference (pp. 13-20).

    Klein, A., Falkner, S., Bartels, S., Hennig, P., & Hutter, F. (2016).
    Fast bayesian optimization of machine learning hyperparameters on large datasets.
    arXiv preprint arXiv:1605.07079.

    Swersky, K., Snoek, J., & Adams, R. P. (2013).
    Multi-task bayesian optimization.
    In Advances in neural information processing systems (pp. 2004-2012).

    Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A., & Talwalkar, A. (2016).
    Hyperband: A novel bandit-based approach to hyperparameter optimization.
    arXiv preprint arXiv:1603.06560.

# Tools used for ASTHPO

SMAC v3 is written in python3 and continuously tested with python3.4 and python3.5.
Its [Random Forest](https://bitbucket.org/aadfreiburg/random_forest_run) is written in C++.

RoBO (Robust Bayesian Optimization) is a package implementing API for Fabolas, MTBO, and Hyperband. It is developed by the [ML4AAD Group of the University of Freiburg](http://www.ml4aad.org/)

Hyperopt is a tool implementing TPE for Bayesian optimization in Python. It is a functioning tool for both

# Installation

Due to requirements of both SMAC3 and RoBO, ASTHPO requires python>=3.4. The RoBO tool also requires that DIRECT be installed. Unfortunately, development for DIRECT has stagnated and the tool is only operational on Linux platforms. As a result, ASTHPO can only be built on Linux operating systems. Future development will aim to develop ASTHPO in a container system and to also create native implementations of the tools described above to allow ASTHPO to become a more cross-platform tool.

Besides the listed requirements (see `requirements.txt`), the random forest used in SMAC3 requires SWIG (>= 3.0). RoBO also requires gfortran for Fortran code compilation and Eigen, a C++ library for linear algebra applications.


	apt-get install gcc swig libeigen3-dev gfortran

    cat requirements.txt | xargs -n 1 -L 1 pip install

    python setup.py install

If you use Anaconda as your Python environment, you have to install two packages before you can install SMAC:

	conda install gcc swig

# Contact

SMAC v3 is developed by the [Centre for Artificial Intelligence Research](http://cair.za.net/) (CAIR) at the University of Cape Town.

If you found a bug, please report to https://github.com/BrutishGuy/ASTHPO
