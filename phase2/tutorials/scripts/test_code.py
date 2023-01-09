import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import auc, roc_curve

from sklearn.model_selection import train_test_split

df = pd.read_csv('mushrooms.csv')

print(df.head())

print('')

print(df.info())

pd.get_dummies(df['class'], drop_first=True)

df.drop(columns=['class'], axis=1)

pd.get_dummies(X, drop_first=True)

train_test_split(X, y, random_state=42)

logreg = LogisticRegression(fit_intercept=False, C=1000000000000.0, solver='liblinear')
# (class) LogisticRegression(penalty: str = "l2", *, dual: bool = False, tol: float = 0.0001, C: float = 1, fit_intercept: bool = True, intercept_scaling: int = 1, class_weight: Unknown | None = None, random_state: Unknown | None = None, solver: str = "lbfgs", max_iter: int = 100, multi_class: str = "auto", verbose: int = 0, warm_start: bool = False, n_jobs: Unknown | None = None, l1_ratio: Unknown | None = None)
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Logistic Regression (aka logit, MaxEnt) classifier.
#
# In the multiclass case, the training algorithm uses the one-vs-rest (OvR)
# scheme if the 'multi\_class' option is set to 'ovr', and uses the
# cross-entropy loss if the 'multi\_class' option is set to 'multinomial'.
# (Currently the 'multinomial' option is supported only by the 'lbfgs',
# 'sag', 'saga' and 'newton-cg' solvers.)
#
# This class implements regularized logistic regression using the
# 'liblinear' library, 'newton-cg', 'sag', 'saga' and 'lbfgs' solvers. \*\*Note
# that regularization is applied by default\*\*. It can handle both dense
# and sparse input. Use C-ordered arrays or CSR matrices containing 64-bit
# floats for optimal performance; any other input format will be converted
# (and copied).
#
# The 'newton-cg', 'sag', and 'lbfgs' solvers support only L2 regularization
# with primal formulation, or no regularization. The 'liblinear' solver
# supports both L1 and L2 regularization, with a dual formulation only for
# the L2 penalty. The Elastic-Net regularization is only supported by the
# 'saga' solver.
#
# Read more in the `User Guide <logistic_regression>`.
#
# Parameters
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# penalty : {'l1', 'l2', 'elasticnet', 'none'}, default='l2'  
# &nbsp;&nbsp;&nbsp;&nbsp;Specify the norm of the penalty:
#
#   - `'none'`: no penalty is added;
#   - `'l2'`: add a L2 penalty term and it is the default choice;
#   - `'l1'`: add a L1 penalty term;
#   - `'elasticnet'`: both L1 and L2 penalty terms are added.
#
# dual : bool, default=False  
# &nbsp;&nbsp;&nbsp;&nbsp;Dual or primal formulation. Dual formulation is only implemented for
# l2 penalty with liblinear solver. Prefer dual=False when
# n\_samples &gt; n\_features.
#
# tol : float, default=1e-4  
# &nbsp;&nbsp;&nbsp;&nbsp;Tolerance for stopping criteria.
#
# C : float, default=1.0  
# &nbsp;&nbsp;&nbsp;&nbsp;Inverse of regularization strength; must be a positive float.
# Like in support vector machines, smaller values specify stronger
# regularization.
#
# fit\_intercept : bool, default=True  
# &nbsp;&nbsp;&nbsp;&nbsp;Specifies if a constant (a.k.a. bias or intercept) should be
# added to the decision function.
#
# intercept\_scaling : float, default=1  
# &nbsp;&nbsp;&nbsp;&nbsp;Useful only when the solver 'liblinear' is used
# and self.fit\_intercept is set to True. In this case, x becomes
# \[x, self.intercept\_scaling\],
# i.e. a "synthetic" feature with constant value equal to
# intercept\_scaling is appended to the instance vector.
# The intercept becomes `intercept_scaling * synthetic_feature_weight`.
#
# &nbsp;&nbsp;&nbsp;&nbsp;Note! the synthetic feature weight is subject to l1/l2 regularization
# as all other features.
# To lessen the effect of regularization on synthetic feature weight
# (and therefore on the intercept) intercept\_scaling has to be increased.
#
# class\_weight : dict or 'balanced', default=None  
# &nbsp;&nbsp;&nbsp;&nbsp;Weights associated with classes in the form `{class_label: weight}`.
# If not given, all classes are supposed to have weight one.
#
# &nbsp;&nbsp;&nbsp;&nbsp;The "balanced" mode uses the values of y to automatically adjust
# weights inversely proportional to class frequencies in the input data
# as `n_samples / (n_classes * np.bincount(y))`.
#
# &nbsp;&nbsp;&nbsp;&nbsp;Note that these weights will be multiplied with sample\_weight (passed
# through the fit method) if sample\_weight is specified.
#
# random\_state : int, RandomState instance, default=None  
# &nbsp;&nbsp;&nbsp;&nbsp;Used when `solver` == 'sag', 'saga' or 'liblinear' to shuffle the
# data. See `Glossary <random_state>` for details.
#
# solver : {'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'},             default='lbfgs'
#
# &nbsp;&nbsp;&nbsp;&nbsp;Algorithm to use in the optimization problem. Default is 'lbfgs'.
# To choose a solver, you might want to consider the following aspects:
#
#     - For small datasets, 'liblinear' is a good choice, whereas 'sag'
# and 'saga' are faster for large ones;
#     - For multiclass problems, only 'newton-cg', 'sag', 'saga' and
# 'lbfgs' handle multinomial loss;
#     - 'liblinear' is limited to one-versus-rest schemes.
#
# max\_iter : int, default=100  
# &nbsp;&nbsp;&nbsp;&nbsp;Maximum number of iterations taken for the solvers to converge.
#
# multi\_class : {'auto', 'ovr', 'multinomial'}, default='auto'  
# &nbsp;&nbsp;&nbsp;&nbsp;If the option chosen is 'ovr', then a binary problem is fit for each
# label. For 'multinomial' the loss minimised is the multinomial loss fit
# across the entire probability distribution, \*even when the data is
# binary\*. 'multinomial' is unavailable when solver='liblinear'.
# 'auto' selects 'ovr' if the data is binary, or if solver='liblinear',
# and otherwise selects 'multinomial'.
#
# verbose : int, default=0  
# &nbsp;&nbsp;&nbsp;&nbsp;For the liblinear and lbfgs solvers set verbose to any positive
# number for verbosity.
#
# warm\_start : bool, default=False  
# &nbsp;&nbsp;&nbsp;&nbsp;When set to True, reuse the solution of the previous call to fit as
# initialization, otherwise, just erase the previous solution.
# Useless for liblinear solver. See `the Glossary <warm_start>`.
#
# n\_jobs : int, default=None  
# &nbsp;&nbsp;&nbsp;&nbsp;Number of CPU cores used when parallelizing over classes if
# multi\_class='ovr'". This parameter is ignored when the `solver` is
# set to 'liblinear' regardless of whether 'multi\_class' is specified or
# not. `None` means 1 unless in a `joblib.parallel_backend`
# context. `-1` means using all processors.
# See `Glossary <n_jobs>` for more details.
#
# l1\_ratio : float, default=None  
# &nbsp;&nbsp;&nbsp;&nbsp;The Elastic-Net mixing parameter, with `0 <= l1_ratio <= 1`. Only
# used if `penalty='elasticnet'`. Setting `l1_ratio=0` is equivalent
# to using `penalty='l2'`, while setting `l1_ratio=1` is equivalent
# to using `penalty='l1'`. For `0 < l1_ratio <1`, the penalty is a
# combination of L1 and L2.
#
# Attributes
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# classes\_ : ndarray of shape (n\_classes, )  
# &nbsp;&nbsp;&nbsp;&nbsp;A list of class labels known to the classifier.
#
# coef\_ : ndarray of shape (1, n\_features) or (n\_classes, n\_features)  
# &nbsp;&nbsp;&nbsp;&nbsp;Coefficient of the features in the decision function.
#
# &nbsp;&nbsp;&nbsp;&nbsp;`coef_` is of shape (1, n\_features) when the given problem is binary.
# In particular, when `multi_class='multinomial'`, `coef_` corresponds
# to outcome 1 (True) and `-coef_` corresponds to outcome 0 (False).
#
# intercept\_ : ndarray of shape (1,) or (n\_classes,)  
# &nbsp;&nbsp;&nbsp;&nbsp;Intercept (a.k.a. bias) added to the decision function.
#
# &nbsp;&nbsp;&nbsp;&nbsp;If `fit_intercept` is set to False, the intercept is set to zero.
# `intercept_` is of shape (1,) when the given problem is binary.
# In particular, when `multi_class='multinomial'`, `intercept_`
# corresponds to outcome 1 (True) and `-intercept_` corresponds to
# outcome 0 (False).
#
# n\_features\_in\_ : int  
# &nbsp;&nbsp;&nbsp;&nbsp;Number of features seen during `fit`.
#
# feature\_names\_in\_ : ndarray of shape (`n_features_in_`,)  
# &nbsp;&nbsp;&nbsp;&nbsp;Names of features seen during `fit`. Defined only when `X`
# has feature names that are all strings.
#
# n\_iter\_ : ndarray of shape (n\_classes,) or (1, )  
# &nbsp;&nbsp;&nbsp;&nbsp;Actual number of iterations for all classes. If binary or multinomial,
# it returns only 1 element. For liblinear solver, only the maximum
# number of iteration across all classes is given.
#
# See Also
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# SGDClassifier : Incrementally trained logistic regression (when given  
# &nbsp;&nbsp;&nbsp;&nbsp;the parameter `loss="log"`).  
# LogisticRegressionCV : Logistic regression with built-in cross validation.
#
# Notes
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# The underlying C implementation uses a random number generator to
# select features when fitting the model. It is thus not uncommon,
# to have slightly different results for the same input data. If
# that happens, try with a smaller tol parameter.
#
# Predict output may not match that of standalone liblinear in certain
# cases. See `differences from liblinear <liblinear_differences>`
# in the narrative documentation.
#
# References
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# L-BFGS-B -- Software for Large-scale Bound-constrained Optimization  
# &nbsp;&nbsp;&nbsp;&nbsp;Ciyou Zhu, Richard Byrd, Jorge Nocedal and Jose Luis Morales.  
# &nbsp;&nbsp;&nbsp;&nbsp;http://users.iems.northwestern.edu/\~nocedal/lbfgsb.html
#
# LIBLINEAR -- A Library for Large Linear Classification  
# &nbsp;&nbsp;&nbsp;&nbsp;https://www.csie.ntu.edu.tw/\~cjlin/liblinear/
#
# SAG -- Mark Schmidt, Nicolas Le Roux, and Francis Bach  
# &nbsp;&nbsp;&nbsp;&nbsp;Minimizing Finite Sums with the Stochastic Average Gradient  
# &nbsp;&nbsp;&nbsp;&nbsp;https://hal.inria.fr/hal-00860051/document
#
# SAGA -- Defazio, A., Bach F. & Lacoste-Julien S. (2014).  
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`"SAGA: A Fast Incremental Gradient Method With Support
# for Non-Strongly Convex Composite Objectives" <1407.0202>`
#
# Hsiang-Fu Yu, Fang-Lan Huang, Chih-Jen Lin (2011). Dual coordinate descent  
# &nbsp;&nbsp;&nbsp;&nbsp;methods for logistic regression and maximum entropy models.  
# &nbsp;&nbsp;&nbsp;&nbsp;Machine Learning 85(1-2):41-75.  
# &nbsp;&nbsp;&nbsp;&nbsp;https://www.csie.ntu.edu.tw/\~cjlin/papers/maxent\_dual.pdf
#
# Examples
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# >>> from sklearn.datasets import load_iris
# >>> from sklearn.linear_model import LogisticRegression
# >>> X, y = load_iris(return_X_y=True)
# >>> clf = LogisticRegression(random_state=0).fit(X, y)
# >>> clf.predict(X[:2, :])
# array([0, 0])
# >>> clf.predict_proba(X[:2, :])
# array([[9.8...e-01, 1.8...e-02, 1.4...e-08],
#        [9.7...e-01, 2.8...e-02, ...e-08]])
# >>> clf.score(X, y)
# 0.97...

logreg.fit(X_train, y_train)

logreg.predict(X_test)

model_log.decision_function(X_train)

roc_curve(y_train, y_train_score)

model_log.decision_function(X_test)

roc_curve(y_test, y_test_score)

sns.set_style('darkgrid', {'axes.facecolor': '0.9'})

plt.figure(figsize=(10, 8))

plt.plot(train_fpr, train_tpr, color='darkorange', lw=lw, label='ROC curve')

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# (module) plt
# ───────────────────────────────────────────────────────────────────────────
# `matplotlib.pyplot` is a state-based interface to matplotlib. It provides
# an implicit,  MATLAB-like, way of plotting.  It also opens figures on your
# screen, and acts as the figure GUI manager.
#
# pyplot is mainly intended for interactive plots and simple cases of
# programmatic plot generation:
#
# import numpy as np
# import matplotlib.pyplot as plt
#
# x = np.arange(0, 5, 0.1)
# y = np.sin(x)
# plt.plot(x, y)
# ───────────────────────────────────────────────────────────────────────────
# The explicit object-oriented API is recommended for complex plots, though
# pyplot is still usually used to create the figure and often the axes in the
# figure. See `.pyplot.figure`, `.pyplot.subplots`, and
# `.pyplot.subplot_mosaic` to create figures, and
# `Axes API </api/axes_api>` for the plotting methods on an Axes:
#
# import numpy as np
# import matplotlib.pyplot as plt
#
# x = np.arange(0, 5, 0.1)
# y = np.sin(x)
# fig, ax = plt.subplots()
# ax.plot(x, y)
# ───────────────────────────────────────────────────────────────────────────
# See `api_interfaces` for an explanation of the tradeoffs between the
# implicit and explicit interfaces.

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.yticks([i / 20.0 for i in range(21)])

plt.xticks([i / 20.0 for i in range(21)])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic (ROC) Curve for Training Set')

plt.legend(loc='lower right')

print('Training AUC: {}'.format(auc(train_fpr, train_tpr)))

plt.show()

plt.figure(figsize=(10, 8))

plt.plot(test_fpr, test_tpr, color='darkorange', lw=lw, label='ROC curve')

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.yticks([i / 20.0 for i in range(21)])

plt.xticks([i / 20.0 for i in range(21)])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic (ROC) Curve for Test Set')

plt.legend(loc='lower right')

print('Test AUC: {}'.format(auc(test_fpr, test_tpr)))

print('')

plt.show()

df.head()

df.info()

'Training AUC: {}'.format(auc(train_fpr, train_tpr))

'Test AUC: {}'.format(auc(test_fpr, test_tpr))

auc(train_fpr, train_tpr)

auc(test_fpr, test_tpr)

range(21)

range(21)

range(21)

range(21)

