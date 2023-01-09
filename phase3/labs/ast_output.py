from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error as _mse
from sklearn.preprocessing import PolynomialFeatures
from scipy import stats
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_moons
from sklearn import svm
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.impute import SimpleImputer
import statsmodels as sm
import itertools
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE, ADASYN
from tqdm.autonotebook import tqdm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import warnings
import csv
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
from matplotlib.patches import Polygon
import ride
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from scipy.spatial.distance import euclidean
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
import timeit
import seaborn as sns
import random
import scipy.stats as stats
import matplotlib as mpl
from school import School
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from pylab import rcParams
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error
import sys
from sklearn.linear_model import Lasso
from itertools import combinations
from sklearn.linear_model import *
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from ride import Ride
from math import log
from sklearn.linear_model import Ridge
import cvxpy as cp
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from tabulate import tabulate
from scipy.special import comb
from sklearn import tree
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from shopping_cart import ShoppingCart
from driver import Driver
from sklearn.metrics import plot_confusion_matrix
#----------------------------------------------------------------------------------------------------
# comb as comb
#----------------------------------------------------------------------------------------------------
y.append(comb(a + b, a))
comb(a + b, a)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# matplotlib.pyplot as plt
#----------------------------------------------------------------------------------------------------
plt.vlines(x=x_value + delta_x, ymin=y_val, ymax=y_val_max, color='darkorange', label=vline_lab)
plt.figure(figsize=(20, 10))
plt.title('Four Blobs')
plt.title('Four blobs')
plt.scatter(X_3[:, 0], X_3[:, 1], c=y_3, edgecolors='gray')
plt.ylabel('R-squared')
plt.plot(max_features, test_results, 'r', label='Test AUC')
plt.scatter(list(range(10, 95)), testing_precision, label='testing_precision')
plt.style.use('ggplot')
plt.xlim([0.0, 1.0])
plt.title('d= %r, gam= %r, r = %r , score = %r' % (d, gamma, r, round(clf.score(X_test, y_test), 2)))
plt.plot(min_samples_splits, mse_results, 'r', label='RMSE')
plt.subplots(figsize=(12, 5))
plt.xlabel('Tree Depth')
plt.scatter(y_test, poly_test_predictions, label='Model')
plt.ylim([np.floor(np.min([x[:, 1], y[:, 1]])), np.ceil(np.max([x[:, 1], y[:, 1]]))])
plt.axhline(y=0, color='lightgrey')
plt.legend(bbox_to_anchor=(1, 1))
plt.xlabel('max features')
plt.scatter(X_4[:, 0], X_4[:, 1], c=y_4, s=25)
plt.ylim([0.0, 1.05])
plt.ylabel('RMSE')
plt.plot(fpr, tpr, color=colors[n], lw=lw, label='ROC curve Regularization Weight: {}'.format(names[n]))
plt.title('Combination sample space of a 25 observation sample compared to various second sample sizes')
plt.plot(max_depths, mse_results, 'r', label='RMSE')
plt.plot(min_samples_splits, train_results, 'b', label='Train AUC')
plt.title('SVC, C=0.1')
plt.scatter(list(range(10, 95)), training_recall, label='training_recall')
plt.savefig('./decision_tree.png')
plt.title('Two Moons with Substantial Overlap')
plt.hlines(y=y_val, xmin=x_value, xmax=x_value + delta_x, color='lightgreen', label=hline_lab)
plt.scatter(X1, X2, c=y_test, edgecolors='gray')
plt.xticks([i / 20.0 for i in range(21)])
plt.title('Receiver operating characteristic (ROC) Curve')
plt.title('Two blobs')
plt.subplots(figsize=(10, 6))
plt.plot(train_fpr, train_tpr, color='red', lw=lw, label='Scikit learn Model 2 with intercept Train ROC curve')
plt.figure(figsize=(11, 11))
plt.title('Four blobs with Varying Separability')
plt.subplot(221)
plt.title('gam= %r, C= %r, score = %r' % (gamma, C, round(clf.score(X_4, y_4), 2)))
plt.xlabel('Size of second sample')
plt.gca()
plt.subplot(3, 3, k + 1)
plt.figure(figsize=(7, 6))
plt.ylabel('Classifier Accuracy')
plt.plot(max_depths, train_results, 'b', label='Train AUC')
plt.subplot(121)
plt.ylabel('Feature')
plt.plot([d1_min, d1_max], [sup_dn_at_mind1, sup_dn_at_maxd1], '-.', color='blue')
plt.title('Model vs data for test set')
plt.scatter(X_1[:, 0], X_1[:, 1], c=y_1, s=25)
plt.plot(train_fpr, train_tpr, color='darkorange', lw=lw, label='ROC curve')
plot_confusion_matrix(logreg, X_test, y_test, cmap=plt.cm.Blues)
plt.scatter(1.1124498053361267, rss(1.1124498053361267), c='red')
plt.ylabel('Gross Domestic Sales', fontsize=16)
plt.plot(x, y, '.b')
plt.plot([d1_min, d1_max], [d2_at_mind1, d2_at_maxd1], color='black')
plt.scatter(X_21, X_22, c=y_2)
plt.title(' gam= %r, r = %r , score = %r' % (gamma, r, round(clf.score(X_3, y_3), 2)))
plt.scatter(list(range(10, 95)), training_precision, label='training_precision')
plt.plot(x, pdf)
plt.title('d= %r, gam= %r, r = %r , score = %r' % (d, gamma, r, round(clf.score(X_train, y_train), 2)))
(fig, ax) = plt.subplots(figsize=(10, 7))
plt.plot(y_test, linestyle='-', marker='o', label='actual values')
plt.legend(loc='lower right')
plt.scatter(list(range(10, 95)), training_accuracy, label='training_accuracy')
plt.ylabel('True Positive Rate')
plt.plot(y_test, y_test, label='Actual data')
plt.figure(figsize=(12, 12))
plt.scatter(df['budget'], df['domgross'], label='Actual Data Points')
plt.title('SVC, C=1')
plt.figure(figsize=(10, 4))
plt.scatter(y_test, lm_test_predictions, label='Model')
plt.ylabel('AUC score')
plt.xlabel('Resting Blood Pressure')
plt.figure(figsize=(20, 5))
plt.plot(test_fpr, test_tpr, color='yellow', lw=lw, label='Scikit learn Model 1 Test ROC curve')
plt.title('LinearSVC')
plt.plot(x_values, y_values, label='3x^2 + 11')
plt.scatter(x[:, 0], x[:, 1], color='purple')
plt.title('Actual vs. predicted values')
plt.plot(x_values, function_values, label='f (x) = 3x^2\N{MINUS SIGN}11 ')
plt.xlabel('Standard Deviations Used for Integral Band Width')
plt.title('Conditional Probability of Resting Blood Pressure ~145 for Those With Heart Disease')
plt.title('Receiver operating characteristic (ROC) Curve for Training Set')
plt.figure(figsize=(8, 5))
plt.title('RSS Loss Function for Various Values of m, with minimum marked')
plt.plot(x_values, y_values, label='4x + 15')
plt.plot(max_depths, test_results, 'r', label='Test AUC')
plt.title('Two Blobs with Mild Overlap')
plt.ylabel('y', fontsize=14)
plt.title('Train and Test Accruaccy Versus Various Standard Deviation Bin Ranges for GNB')
plt.plot(test_fpr, test_tpr, color='darkorange', lw=lw, label='ROC curve')
plt.plot(min_samples_leafs, test_results, 'r', label='Test AUC')
plt.plot(x_values, y_values, label='3x^2 - 11')
plt.plot(max_depths, r2_results, 'b', label='R2')
plt.subplots(1, 2, figsize=(10, 5), sharey=True)
plt.plot(test_fpr, test_tpr, color='purple', lw=lw, label='Scikit learn Model 2 with intercept Test ROC curve')
plt.xlabel('Tree depth')
plt.plot(range_stds, train_accs, label='Train Accuracy')
plt.subplot(4, 2, k + 1)
plt.tight_layout()
plt.plot(x_values, derivative_values, color='darkorange', label="f '(x)")
plt.xlabel('Min. Sample Leafs')
plt.show()
plt.subplot(1, 3, i + 1)
plt.barh(range(n_features), model.feature_importances_, align='center')
plt.title('Box plot of all columns in dataset')
plt.figure(figsize=(12, 14))
plt.title('Four Blobs with Varying Separability')
plt.plot(range_stds, test_accs, label='Test Accuracy')
plt.title(col)
plt.scatter(y_train, lm_train_predictions, label='Model')
plt.subplots(figsize=(12, 6))
plt.yticks([i / 20.0 for i in range(21)])
plt.title('Receiver operating characteristic (ROC) Curve for Test Set')
plt.axvline(x=0, color='lightgrey')
plt.subplots()
plt.legend(loc='upper left', bbox_to_anchor=[0, 1], ncol=2, fancybox=True)
plt.title('d= %r, gam= %r, r = %r , score = %r' % (d, gamma, r, round(clf.score(X_3, y_3), 2)))
plt.xlabel('Budget', fontsize=16)
plt.plot(x, y)
plt.figure(figsize=(5, 5))
plt.scatter(X1, X2, c=y, edgecolors='k')
plt.scatter(y[:, 0], y[:, 1], color='yellow')
plt.title('Model vs data for training set')
plt.title('Gross Domestic Sales vs. Budget', fontsize=18)
plt.yticks(np.arange(n_features), data_train.columns.values)
plt.plot(max_features, train_results, 'b', label='Train AUC')
plt.subplots(figsize=(10, 7))
plt.subplots(figsize=(10, 4))
plt.subplots(4, 2, figsize=(15, 15))
plt.plot(x_values, derivative_values, color='darkorange', label="f '(x) = 6x")
plt.legend(loc=(1.01, 0.85))
plt.plot(min_samples_splits, test_results, 'r', label='Test AUC')
plt.title('Two interleaving half circles')
plt.plot(data[col], target, 'o')
plt.subplot(222)
plt.axis('tight')
plt.scatter(X[:, 0], X[:, 1], c=y, s=25)
plt.style.use('seaborn')
plt.ylabel('Prices')
plt.grid(False)
plt.figure(figsize=(10, 10))
plt.plot(train_fpr, train_tpr, color='blue', lw=lw, label='Custom Model Train ROC curve')
plt.xlabel('Feature importance')
plt.scatter(y_train, poly_train_predictions, label='Model')
plt.xticks(range(len(df.columns.values)), df.columns.values)
plt.scatter(X1, X2, c=y_train, edgecolors='gray')
plt.scatter(X_3[:, 0], X_3[:, 1], c=y_3, s=25)
plt.xlabel(col)
plt.plot(test_fpr, test_tpr, color='darkorange', lw=lw, label='Test ROC curve')
plt.plot(min_samples_splits, r2_results, 'b', label='R2')
plt.plot(min_samples_leafs, train_results, 'b', label='Train AUC')
plt.plot((145, 145), (0, stats.norm.pdf(145, loc=aggs['mean'], scale=aggs['std'])), linestyle='dotted')
plt.scatter(list(range(10, 95)), testing_recall, label='testing_recall')
plt.figure(figsize=(8, 8))
plt.scatter(list(range(10, 95)), training_f1, label='training_f1')
plt.subplot(224)
plt.xlabel('False Positive Rate')
plt.plot(test_fpr, test_tpr, color='darkorange', lw=lw, label='Custom Model Test ROC curve')
plt.subplot(223)
plt.scatter(X_4[:, 0], X_4[:, 1], c=y_4, edgecolors='gray')
plt.plot(tan_line['x_dev'], tan_line['tan'], color='yellow', label=tan_line['lab'])
plt.plot(x_values, function_values, label='f (x)')
plt.figure(figsize=(12, 12), dpi=500)
plt.plot(y_train, y_train, label='Actual data')
plt.plot(fpr, tpr, color=colors[n], lw=lw, label='ROC curve Normalization Weight: {}'.format(names[n]))
plt.title('NuSVC, nu=0.5')
plt.scatter(list(range(10, 95)), testing_accuracy, label='testing_accuracy')
plt.ylabel('Probability Density')
plt.plot(y_pred, linestyle='-', marker='o', label='predictions')
plt.subplots(nrows=1, ncols=1, figsize=(12, 12), dpi=300, tight_layout=True)
plt.plot([d1_min, d1_max], [sup_up_at_mind1, sup_up_at_maxd1], '-.', color='blue')
plt.boxplot([df[col] for col in df.columns])
plt.contourf(X1_C, X2_C, Z, alpha=1)
plt.scatter(list(range(10, 95)), testing_f1, label='testing_f1')
plt.figure(figsize=(12, 6))
plt.ylabel('Number of combinations for permutation test')
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve')
plt.title('Two Seperable Blobs')
plt.scatter(X_11, X_12, c=y_1)
plt.xlabel('x', fontsize=14)
plt.scatter(x, 1.331 * x, label='Median Ratio Model')
plt.scatter(x, 1.575 * x, label='Mean Ratio Model')
plt.figure(figsize=(10, 8))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.legend(loc='upper left')
plt.xlabel('Min. Sample splits')
plt.scatter(X_2[:, 0], X_2[:, 1], c=y_2, s=25)
plt.plot(train_fpr, train_tpr, color='blue', lw=lw, label='Train ROC curve')
plt.plot(train_fpr, train_tpr, color='gold', lw=lw, label='Scikit learn Model 1 Train ROC curve')
plt.legend()
plt.title('RSS Loss Function for Various Values of m')
plt.subplot(122)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=25)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# numpy as np
#----------------------------------------------------------------------------------------------------
np.isin(all_idx, training_idx)
np.arange(n_features)
rss_survey_region = [np.sqrt(rss(m)) for m in x_survey_region]
np.zeros((3, 3))
np.concatenate([X_test_cont, X_test_cat.todense()], axis=1)
np.meshgrid(x22_coord, x21_coord)
np.linspace(b - 1, b + 1, 100)
np.mean(b)
random_number = np.random.random()
plt.ylim([np.floor(np.min([x[:, 1], y[:, 1]])), np.ceil(np.max([x[:, 1], y[:, 1]]))])
np.array([2, 2])
m_gradient = np.zeros(len(m_current))
np.array(err).mean()
np.matrix([[1240, 276, 0]])
tree.plot_tree(clf, feature_names=df.columns, class_names=np.unique(y).astype('str'), filled=True)
np.min([x[:, 0], y[:, 0]])
np.power(np.sum(np.power(np.abs(np.array(a) - np.array(b)), c)), 1 / c)
np.transpose(x)
np.transpose(y)
np.mean(ai)
np.mean(dt_grid_search.cv_results_['mean_train_score'])
d1_min = np.min([x[:, 0], y[:, 0]])
np.linspace(X1_min, X1_max, 200)
np.random.choice(self.population)
np.linspace(0.1, 2, num=21)
FunctionTransformer(np.log, validate=True)
test_errs.append(np.mean(test_residuals.astype(float) ** 2))
np.array([0.01, 1, 10])
np.array([[4], [3]])
np.set_printoptions(formatter={'float_kind': '{:f}'.format})
np.shape(X_train_scaled)
predictions = sigmoid(np.dot(X, weights))
np.linalg.inv(A)
np.floor(np.min([x[:, 1], y[:, 1]]))
np.array([[4, 1], [15, 0]])
np.mean(bi)
np.dot(X.transpose(), error_vector)
np.array([[3], [5], [9]])
gradient = np.dot(X.transpose(), error_vector)
np.mean(y_hat - y)
np.random.rand(30, 1)
np.log(num / denom)
np.transpose(A.dot(B))
np.shape(function_terms)
np.random.choice(all_idx, size=round(546 * 0.8), replace=False)
np.mean(test_residuals.astype(float) ** 2)
np.unique(y)
np.array([constant, exponent])
np.array([[2, 13], [1, 4], [72, 6], [18, 12], [27, 5]])
np.zeros(len(m_current))
np.array([y, x])
np.log(p_classes[class_])
np.arange(2, 11)
np.transpose(B)
np.unique(y).astype('str')
np.matrix([[490, 448]])
np.zeros((4, 4))
np.array([0.1, 1])
np.shape(array_1)
np.linspace(X21_min, X21_max, 10)
sigmoid(np.dot(X_train, weights))
np.mean(y_hat)
int(np.shape(function_terms)[0])
np.transpose(y).dot(x)
np.mean(cross_val_score(rf_clf, X_train, y_train, cv=3))
test_mses.append(np.mean(inner_test_mses))
np.mean([yi ** 2 for yi in y_hat])
np.argmax(c_probs)
np.mean(a)
np.linspace(x_value - line_length / 2, x_value + line_length / 2, 50)
gradient = np.gradient(rss_survey_region)[50]
np.array([1, 3])
np.ones((X.shape[1], 1)).flatten()
np.transpose(A)
np.dot(X, weights)
np.linspace(0, 200, num=50)
np.log(ames_cont)
np.ones((X.shape[1], 1))
x_values = [np.linspace(b - 1, b + 1, 100) for b in b_vals]
np.zeros((len(b_values), 2))
np.mean(inner_test_mses)
np.linspace(X11_min, X11_max, 10)
np.array([4, 2, 3])
np.ceil(np.max([x[:, 1], y[:, 1]]))
np.array([x, y])
np.linspace(0.1, 0.5, 5, endpoint=True)
np.matrix([162, 122])
np.linspace(X1_min, X1_max, 500)
np.array(a)
diff_mu_ai_bi = np.mean(ai) - np.mean(bi)
np.array(err)
np.random.normal(0, 0.2, 100)
train_mses.append(np.mean(inner_train_mses))
np.array([[2, 3], [6, 5]])
x_dev = np.linspace(x_value - line_length / 2, x_value + line_length / 2, 50)
np.matrix([[12, 3], [8, 3]])
np.argmin(test_mse)
np.sqrt(mean_sq_err)
np.dot(Xt, y_train)
np.argmax(posteriors)
np.linspace(-30, 30, 100)
np.linalg.inv(XtX)
np.array([[3, 2], [-11, 0]])
np.array([[5, 3], [2, 2]])
np.array([0.1, 1, 100])
np.abs(y_train - y_hat_train)
np.linspace(0.5, 0.9, 10)
np.shape(array_of_terms)
np.array([3, 2])
np.array(b)
np.meshgrid(x2_coord, x1_coord)
range(int(np.shape(array_of_terms)[0]))
np.array([[5, 30], [22, 2]])
np.savetxt(sys.stdout, bval_rss, '%16.2f')
np.bincount(labels)
np.mean(dt_cv_score)
np.random.rand(30, 1).reshape(30)
np.concatenate([X_train_cont, X_train_cat.todense()], axis=1)
int(np.shape(array_of_terms)[0])
np.array([[5], [2]])
np.zeros(np.shape(function_terms))
der_array = np.zeros(np.shape(function_terms))
np.arange(0, 14.5, step=0.5)
np.mean(inner_train_mses)
np.abs(y_test - y_hat_test)
counts = np.bincount(labels)
np.zeros((SIZE, SIZE))
np.matrix([[1, 1, 1], [0.1, 0.2, 0.3], [2, 0, -1]])
p = np.log(p_classes[class_])
tree.plot_tree(classifier_2, feature_names=X.columns, class_names=np.unique(y).astype('str'), filled=True, rounded=True)
plt.yticks(np.arange(n_features), data_train.columns.values)
initial_weights = np.ones((X.shape[1], 1)).flatten()
np.sqrt(mean_squared_error(y_test, y_pred))
term_output(np.array([3, 2]), 2)
np.matrix([[29, 41], [23, 41]])
np.array([[1402, 191], [1371, 821], [949, 1437], [147, 1448]])
np.random.seed(42)
np.transpose(data)
np.random.choice(union, size=len(a), replace=False)
np.sqrt(rss(m))
np.array([0.1, 1, 10])
np.array([[1, 2, 3], [4, 5, 6]])
np.transpose(B).dot(np.transpose(A))
random_person = np.random.choice(self.population)
np.linspace(0.1, 1.0, 10, endpoint=True)
np.dot(Xt, x_train)
np.linspace(start=cur_x - previous_step_size, stop=cur_x + previous_step_size, num=101)
sigmoid(np.dot(X, weights))
sigmoid(np.dot(X_test, weights))
np.power(np.abs(np.array(a) - np.array(b)), c)
np.matrix([[7, 5.25, 0]])
np.random.random()
np.array([0.001, 0.01, 0.1])
np.array([[2], [6], [7]])
np.max([x[:, 0], y[:, 0]])
np.random.seed(0)
np.argmax(counts)
np.linalg.solve(A, B.T)
np.array([[4, 2], [4, 1], [-10, 0]])
np.random.seed(11)
np.linspace(temp.min(), temp.max(), num=10 ** 3)
np.array([[4, 3], [-3, 1]])
np.array([[4, 3], [11, 2]])
np.random.rand(SIZE, SIZE)
ai = np.random.choice(union, size=len(a), replace=False)
range(int(np.shape(function_terms)[0]))
x_survey_region = np.linspace(start=cur_x - previous_step_size, stop=cur_x + previous_step_size, num=101)
np.matrix([[1, 1, 1], [0.5, 0.75, 1.25], [-2, 1, 0]])
np.linspace(X12_min, X12_max, 10)
np.random.rand(100, 1)
np.linspace(X2_min, X2_max, 200)
d1_max = np.max([x[:, 0], y[:, 0]])
np.linspace(start=df['budget'].min(), stop=df['budget'].max(), num=10 ** 5)
np.array(data)
np.shape(X_train_poly)
np.array([[2, 3], [1, 4], [7, 6]])
np.round(y_hat_test, 2)
np.array([3, 4])
np.array(x)
np.arange(data.shape[0])
table = np.zeros((len(b_values), 2))
np.linspace(-10, 10, 100)
np.gradient(rss_survey_region)
np.dot(X_train, weights)
np.sum(np.power(np.abs(np.array(a) - np.array(b)), c))
x = np.array(x)
np.linspace(xi_lower, xi_upper)
np.shape(x11x12)
np.random.rand(100, 1).reshape(100)
np.transpose(x_train)
np.linspace(start=-3, stop=5, num=10 ** 3)
np.max([x[:, 1], y[:, 1]])
np.dot(X_test, weights)
np.linspace(X2_min, X2_max, 500)
np.meshgrid(x12_coord, x11_coord)
np.random.seed(225)
np.abs(np.array(a) - np.array(b))
np.array([y, x1, x2])
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))
np.linspace(0, 5, 100)
np.min([x[:, 1], y[:, 1]])
np.linspace(X22_min, X22_max, 10)
np.transpose(x).dot(y)
np.array([0.1, 2])
np.random.normal(0, 3, 30)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# ride as ride
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# Ride as Ride
#----------------------------------------------------------------------------------------------------
Ride()
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# Driver as Driver
#----------------------------------------------------------------------------------------------------
Driver()
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# load_iris as load_iris
#----------------------------------------------------------------------------------------------------
load_iris(return_X_y=True, as_frame=True)
load_iris()
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# MinMaxScaler as MinMaxScaler
#----------------------------------------------------------------------------------------------------
MinMaxScaler()
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# DecisionTreeClassifier as DecisionTreeClassifier
#----------------------------------------------------------------------------------------------------
DecisionTreeClassifier(criterion='entropy', max_features=6, max_depth=3, min_samples_split=0.7, min_samples_leaf=0.25, random_state=SEED)
DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, random_state=SEED)
dt = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, random_state=SEED)
dt = DecisionTreeClassifier(criterion='entropy', min_samples_split=min_samples_split, random_state=SEED)
dt = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=min_samples_leaf, random_state=SEED)
DecisionTreeClassifier(criterion='entropy', random_state=SEED)
DecisionTreeClassifier(criterion='entropy', max_features=max_feature, random_state=SEED)
DecisionTreeClassifier(criterion='entropy')
BaggingClassifier(DecisionTreeClassifier(criterion='gini', max_depth=5), n_estimators=20)
dt = DecisionTreeClassifier(criterion='entropy', max_features=max_feature, random_state=SEED)
DecisionTreeClassifier(criterion='gini', max_depth=5)
DecisionTreeClassifier(random_state=10)
DecisionTreeClassifier(criterion='entropy', min_samples_leaf=min_samples_leaf, random_state=SEED)
DecisionTreeClassifier(random_state=10, criterion='entropy')
DecisionTreeClassifier()
DecisionTreeClassifier(criterion='entropy', min_samples_split=min_samples_split, random_state=SEED)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# csv as csv
#----------------------------------------------------------------------------------------------------
csv.reader(f)
raw = csv.reader(f)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# rcParams as rcParams
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# sys as sys
#----------------------------------------------------------------------------------------------------
np.savetxt(sys.stdout, bval_rss, '%16.2f')
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# pandas as pd
#----------------------------------------------------------------------------------------------------
pd.Series(residuals)
pd.concat([X_train, X_test])
X_test = pd.concat([pd.DataFrame(log_transformer.transform(X_test[continuous]), index=X_test.index), pd.DataFrame(ohe.transform(X_test[categoricals]), index=X_test.index)], axis=1)
pd.read_csv('titanic.csv')
col_transformed_test = pd.DataFrame(poly.transform(features_test[[col]]))
features_test = pd.concat([features_test.drop(col, axis=1), col_transformed_test], axis=1)
pd.concat([y_train, y_test])
final_model.fit(pd.concat([X_train, X_test]), pd.concat([y_train, y_test]))
pd.read_csv('heart.csv')
pd.DataFrame(poly.transform(features_test[[col]]))
pd.read_csv('ames.csv')
pd.DataFrame(ohe.transform(X_train[categoricals]), index=X_train.index)
y_train = pd.concat([fold for (i, fold) in enumerate(y_folds) if i != n])
pd.Series(residuals).value_counts()
pd.Series(encoder.fit_transform(y_train))
pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
weights_col = pd.DataFrame(initial_weights)
pd.Series(y_train_resampled).value_counts()
X_val = pd.concat([X_val.drop(col, axis=1), col_transformed_val], axis=1)
pd.DataFrame(poly.fit_transform(X_train[[col]]), columns=poly.get_feature_names([col]))
pd.concat([X_test, y_test], axis=1)
pd.DataFrame(X_train_scaled, columns=X.columns)
pd.concat([weights_col, pd.DataFrame(weights)], axis=1)
pd.concat([fold for (i, fold) in enumerate(X_folds) if i != n])
pd.concat([features_train.drop(col, axis=1), col_transformed_train], axis=1)
pd.read_excel('movie_data.xlsx')
pd.concat([features_test.drop(col, axis=1), col_transformed_test], axis=1)
pd.get_dummies(df[relevant_columns], drop_first=True, dtype=float)
col_transformed_train = pd.DataFrame(poly.fit_transform(X_train[[col]]), columns=poly.get_feature_names([col]))
pd.read_csv('salaries_final.csv', index_col=0)
pd.concat([minority, undersampled_majority])
pd.plotting.scatter_matrix(df, figsize=(10, 10))
pd.DataFrame(ohe.transform(X_test[categoricals]), index=X_test.index)
pd.DataFrame(poly.transform(X_test[[col]]), columns=poly.get_feature_names([col]))
pd.read_csv('simulation.csv')
pd.Series(y_train_resampled)
pd.read_csv('pima-indians-diabetes.csv')
pd.concat([pd.DataFrame(log_transformer.transform(X_test[continuous]), index=X_test.index), pd.DataFrame(ohe.transform(X_test[categoricals]), index=X_test.index)], axis=1)
pd.DataFrame(poly.fit_transform(X_val[[col]]), columns=poly.get_feature_names([col]))
pd.read_csv('ames.csv', index_col=0)
print(pd.Series(y_train_resampled).value_counts())
pd.read_csv('mushrooms.csv')
pd.get_dummies(ames[categoricals], prefix=categoricals, drop_first=True)
pd.Series(encoder.transform(y_test))
pd.concat([X_val.drop(col, axis=1), col_transformed_val], axis=1)
pd.concat([pd.DataFrame(log_transformer.transform(X_train[continuous]), index=X_train.index), pd.DataFrame(ohe.transform(X_train[categoricals]), index=X_train.index)], axis=1)
pd.concat([ames_log_norm, ames_ohe], axis=1)
pd.DataFrame(scaled_data_train, columns=one_hot_df.columns)
pd.concat([fold for (i, fold) in enumerate(y_folds) if i != n])
pd.read_csv('./data_banknote_authentication.csv', header=None, names=['Variance', 'Skewness', 'Kurtosis', 'Entropy'])
pd.DataFrame(initial_weights)
pd.Series(residuals).value_counts(normalize=True)
pd.get_dummies(salaries)
pd.DataFrame(X_test_scaled, columns=X.columns)
X_train = pd.concat([X_train.drop(col, axis=1), col_transformed_train], axis=1)
pd.Series(y_resampled).value_counts()
pd.DataFrame(log_transformer.transform(X_test[continuous]), index=X_test.index)
pd.concat([X_train, y_train], axis=1)
pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'text'])
pd.DataFrame()
pd.get_dummies(X, drop_first=True)
self.time_step_statistics_df = pd.DataFrame()
print(pd.Series(residuals).value_counts())
print(pd.Series(y_resampled).value_counts())
pd.read_csv('housing_prices.csv', index_col=0)
pd.get_dummies(df)
pd.get_dummies(salaries[xcols], drop_first=True)
pd.concat([X_train.drop(col, axis=1), col_transformed_train], axis=1)
X_train = pd.concat([fold for (i, fold) in enumerate(X_folds) if i != n])
weights_col = pd.concat([weights_col, pd.DataFrame(weights)], axis=1)
features_train = pd.concat([features_train.drop(col, axis=1), col_transformed_train], axis=1)
col_transformed_train = pd.DataFrame(poly.fit_transform(features_train[[col]]))
pd.read_csv('./mushrooms.csv')
pd.crosstab(y_test, y_preds, rownames=['True'], colnames=['Predicted'], margins=True)
pd.read_excel('movie_data_detailed_with_ols.xlsx')
pd.read_csv('creditcard.csv.gz', compression='gzip')
X_train = pd.concat([pd.DataFrame(log_transformer.transform(X_train[continuous]), index=X_train.index), pd.DataFrame(ohe.transform(X_train[categoricals]), index=X_train.index)], axis=1)
X_test = pd.concat([X_test.drop(col, axis=1), col_transformed_test], axis=1)
pd.read_csv('data_banknote_authentication.csv', header=None)
print(pd.Series(residuals).value_counts(normalize=True))
pd.DataFrame(weights)
pd.get_dummies(salaries['Target'], drop_first=True)
pd.DataFrame(poly.fit_transform(features_train[[col]]))
pd.read_csv('petrol_consumption.csv')
pd.get_dummies(df['class'], drop_first=True)
pd.concat([X_test.drop(col, axis=1), col_transformed_test], axis=1)
pd.Series(y_resampled)
col_transformed_test = pd.DataFrame(poly.transform(X_test[[col]]), columns=poly.get_feature_names([col]))
pd.DataFrame(log_transformer.transform(X_train[continuous]), index=X_train.index)
pd.DataFrame(top_7_polynomials, columns=['Column', 'Degree', 'R^2'])
pd.read_csv('winequality-red.csv')
pd.DataFrame({'color': ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet']})
col_transformed_val = pd.DataFrame(poly.fit_transform(X_val[[col]]), columns=poly.get_feature_names([col]))
pd.DataFrame(X_val_scaled, columns=X.columns)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# FunctionTransformer as FunctionTransformer
#----------------------------------------------------------------------------------------------------
FunctionTransformer(np.log, validate=True)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# OneHotEncoder as OneHotEncoder
#----------------------------------------------------------------------------------------------------
OneHotEncoder(handle_unknown='ignore')
OneHotEncoder(drop='first', sparse=False)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# train_test_split as train_test_split
#----------------------------------------------------------------------------------------------------
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=t_size, random_state=i)
train_test_split(X, y, test_size=0.2, random_state=4)
train_test_split(X, y, test_size=0.25, random_state=42)
train_test_split(X, y, test_size=t_size, random_state=42)
train_test_split(df, target, test_size=0.25, random_state=42)
train_test_split(X, y, test_size=0.2, random_state=42)
train_test_split(X, y, test_size=i / 100.0)
train_test_split(X_resampled, y_resampled, random_state=0)
train_test_split(X, y, random_state=0)
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=None)
train_test_split(X, y, test_size=None)
train_test_split(features, target, test_size=0.2, random_state=42)
train_test_split(X, y, random_state=10)
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=t_size, random_state=42)
train_test_split(data, target, test_size=0.25, random_state=123)
train_test_split(X, y, test_size=0.25, random_state=22)
train_test_split(X_train, y_train, random_state=0)
train_test_split(X, y, random_state=17)
train_test_split(data, target, test_size=0.25, random_state=0)
train_test_split(X, y, random_state=1)
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=i / 100.0)
train_test_split(X, y, test_size=t_size, random_state=i)
train_test_split(X, y, test_size=0.3, random_state=SEED)
train_test_split(one_hot_df, labels, test_size=0.25, random_state=42)
train_test_split(X, y, random_state=42)
train_test_split(X, y, test_size=0.2, random_state=10)
train_test_split(X_3, y_3, test_size=0.33, random_state=123)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# LinearRegression as LinearRegression
#----------------------------------------------------------------------------------------------------
lr = LinearRegression()
rfe = RFE(LinearRegression(), n_features_to_select=n)
score = LinearRegression().fit(features_train, y_train).score(features_test, y_test)
LinearRegression().fit(features_train, y_train).score(features_test, y_test)
LinearRegression().fit(features_train, y_train)
RFE(LinearRegression(), n_features_to_select=n)
LinearRegression()
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# mean_squared_error as mean_squared_error
#----------------------------------------------------------------------------------------------------
print('Lasso, alpha=10:  ', mean_squared_error(y_test, lasso_10.predict(X_test_preprocessed)))
print('MSE:', mean_squared_error(y_val, final_model.predict(X_val)))
mean_squared_error(y_train, ridge.predict(X_train_preprocessed))
mean_squared_error(y_train, y_hat_train)
print('Training MSE:', mean_squared_error(y_train, ridge.predict(X_train_preprocessed)))
print('Lasso, alpha=1:   ', mean_squared_error(y_test, lasso.predict(X_test_preprocessed)))
train_mses.append(mean_squared_error(y_train, y_hat_train))
mean_squared_error(y_test, y_pred)
mean_squared_error(y_true, y_predict, squared=False)
mean_squared_error(y_train, ridge_10.predict(X_train_preprocessed))
print('Test MSE:    ', mean_squared_error(y_test, ridge.predict(X_test_preprocessed)))
mean_squared_error(y_train, linreg.predict(X_train_preprocessed))
print('Ridge, alpha=1:   ', mean_squared_error(y_test, ridge.predict(X_test_preprocessed)))
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('Test MSE:    ', mean_squared_error(y_test, linreg.predict(X_test_preprocessed)))
train_mse.append(mean_squared_error(y_train, train_preds))
mean_squared_error(y_test, lasso.predict(X_test_preprocessed))
mean_squared_error(y_train, lasso.predict(X_train_preprocessed))
mean_squared_error(y_test, lasso_10.predict(X_test_preprocessed))
print('Training MSE:', mean_squared_error(y_train, ridge_10.predict(X_train_preprocessed)))
print('Ridge, alpha=10:  ', mean_squared_error(y_test, ridge_10.predict(X_test_preprocessed)))
inner_train_mses.append(mean_squared_error(y_train, y_hat_train))
test_mse.append(mean_squared_error(y_test, test_preds))
mean_squared_error(y_test, ridge_10.predict(X_test_preprocessed))
np.sqrt(mean_squared_error(y_test, y_pred))
print('Linear Regression:', mean_squared_error(y_test, linreg.predict(X_test_preprocessed)))
test_mses.append(mean_squared_error(y_test, y_hat_test))
print('Training MSE:', mean_squared_error(y_train, lasso_10.predict(X_train_preprocessed)))
print('Training MSE:', mean_squared_error(y_train, lasso.predict(X_train_preprocessed)))
rmse = mean_squared_error(y_true, y_predict, squared=False)
print('Test MSE:    ', mean_squared_error(y_test, lasso_10.predict(X_test_preprocessed)))
mean_squared_error(y_train, lasso_10.predict(X_train_preprocessed))
mean_squared_error(y_val, final_model.predict(X_val))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))
mean_squared_error(y_train, train_preds)
mean_squared_error(y_test, y_hat_test)
mean_squared_error(y_test, linreg.predict(X_test_preprocessed))
mean_squared_error(y_test, ridge.predict(X_test_preprocessed))
inner_test_mses.append(mean_squared_error(y_test, y_hat_test))
print('Test MSE:    ', mean_squared_error(y_test, ridge_10.predict(X_test_preprocessed)))
print('Test MSE:    ', mean_squared_error(y_test, lasso.predict(X_test_preprocessed)))
print('Training MSE:', mean_squared_error(y_train, linreg.predict(X_train_preprocessed)))
mean_squared_error(y_test, test_preds)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# cross_val_score as cross_val_score
#----------------------------------------------------------------------------------------------------
cross_val_score(adaboost_clf, df, target, cv=5).mean()
np.mean(cross_val_score(rf_clf, X_train, y_train, cv=3))
cross_val_score(gbt_clf, df, target, cv=5)
print(cross_val_score(adaboost_clf, df, target, cv=5).mean())
cross_val_score(rf_clf, X_train, y_train, cv=3)
cross_val_score(dt_clf, X_train, y_train, cv=3)
cross_val_score(linreg, X, y, cv=5, scoring='neg_mean_squared_error')
cross_val_score(adaboost_clf, df, target, cv=5)
cross_val_score(gbt_clf, df, target, cv=5).mean()
print(cross_val_score(gbt_clf, df, target, cv=5).mean())
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# PolynomialFeatures as PolynomialFeatures
#----------------------------------------------------------------------------------------------------
PolynomialFeatures(3)
PolynomialFeatures(degree, include_bias=False)
poly = PolynomialFeatures(degree, include_bias=False)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# * as *
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# warnings as warnings
#----------------------------------------------------------------------------------------------------
warnings.filterwarnings('ignore')
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# SimpleImputer as SimpleImputer
#----------------------------------------------------------------------------------------------------
SimpleImputer(strategy='constant', fill_value='missing')
SimpleImputer(strategy='median')
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# Lasso as Lasso
#----------------------------------------------------------------------------------------------------
Lasso(alpha=10000)
Lasso()
lasso = Lasso(alpha=alpha)
Lasso(alpha=1)
Lasso(alpha=10)
Lasso(alpha=alpha)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# Ridge as Ridge
#----------------------------------------------------------------------------------------------------
Ridge()
Ridge(alpha=10)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# combinations as combinations
#----------------------------------------------------------------------------------------------------
combinations(X_train.columns, 2)
list(combinations(X_train.columns, 2))
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# RFE as RFE
#----------------------------------------------------------------------------------------------------
rfe = RFE(LinearRegression(), n_features_to_select=n)
RFE(LinearRegression(), n_features_to_select=n)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# StandardScaler as StandardScaler
#----------------------------------------------------------------------------------------------------
StandardScaler()
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# statsmodels.api as sm
#----------------------------------------------------------------------------------------------------
sm.tools.add_constant(X)
sm.Logit(y, X)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# LogisticRegression as LogisticRegression
#----------------------------------------------------------------------------------------------------
logreg = LogisticRegression(fit_intercept=False, C=1e+20, solver='liblinear')
logreg = LogisticRegression(fit_intercept=False, C=c, solver='liblinear')
LogisticRegression(fit_intercept=True, C=1.5 ** n, solver='liblinear')
LogisticRegression(fit_intercept=False, C=1000000000000.0, solver='liblinear')
LogisticRegression(fit_intercept=False, C=1e+16, solver='liblinear')
LogisticRegression(fit_intercept=True, C=1e+16, solver='liblinear')
logreg = LogisticRegression(fit_intercept=True, C=1.5 ** n, solver='liblinear')
LogisticRegression(fit_intercept=False, C=c, solver='liblinear')
LogisticRegression(fit_intercept=False, C=1e+20, solver='liblinear')
LogisticRegression(fit_intercept=False, solver='liblinear')
LogisticRegression(fit_intercept=False, C=1e+25, solver='liblinear')
LogisticRegression(C=1000000000000.0, fit_intercept=False, solver='liblinear')
logreg = LogisticRegression(fit_intercept=False, C=1e+25, solver='liblinear')
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# confusion_matrix as confusion_matrix
#----------------------------------------------------------------------------------------------------
confusion_matrix(y_test, adaboost_test_preds)
confusion_matrix(target_test, pred)
confusion_matrix(test_predictions, y_test)
confusion_matrix(y_test, gbt_clf_test_preds)
confusion_matrix(y_test, y_hat_test)
print(confusion_matrix(target_test, pred))
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# plot_confusion_matrix as plot_confusion_matrix
#----------------------------------------------------------------------------------------------------
plot_confusion_matrix(model_log, X_test, y_test)
plot_confusion_matrix(clf, X, y, values_format='.3g')
plot_confusion_matrix(logreg, X_test, y_test, display_labels=['not fraud', 'fraud'], values_format='.5g')
plot_confusion_matrix(logreg, X_test, y_test, cmap=plt.cm.Blues)
plot_confusion_matrix(classifier, X, y, values_format='.3g')
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# precision_score as precision_score
#----------------------------------------------------------------------------------------------------
print('Testing Precision: ', precision_score(y_test, y_hat_test))
'Precision Score: {}'.format(precision_score(labels, preds))
precision_score(y_train, y_hat_train)
precision_score(labels, preds)
print('Precision Score: {}'.format(precision_score(labels, preds)))
print('Training Precision: ', precision_score(y_train, y_hat_train))
precision_score(y_test, y_hat_test)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# recall_score as recall_score
#----------------------------------------------------------------------------------------------------
recall_score(labels, preds)
print('Recall Score: {}'.format(recall_score(labels, preds)))
'Recall Score: {}'.format(recall_score(labels, preds))
recall_score(y_train, y_hat_train)
recall_score(y_test, y_hat_test)
print('Testing Recall: ', recall_score(y_test, y_hat_test))
print('Training Recall: ', recall_score(y_train, y_hat_train))
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# accuracy_score as accuracy_score
#----------------------------------------------------------------------------------------------------
acc = accuracy_score(true, preds)
accuracy_score(y_test, y_hat_test)
'Testing Accuracy for Decision Tree Classifier: {:.4}%'.format(accuracy_score(target_test, pred) * 100)
print('Testing Accuracy: ', accuracy_score(y_test, y_hat_test))
print('Accuracy Score: {}'.format(accuracy_score(labels, preds)))
accuracy_score(y_test, y_pred)
'Testing Accuracy: {}'.format(accuracy_score(y_test, preds))
accuracy_score(y_train, y_hat_train)
accuracy_score(target_test, pred)
print('Testing Accuracy for Decision Tree Classifier: {:.4}%'.format(accuracy_score(target_test, pred) * 100))
accuracy_score(y_test, preds)
print('Testing Accuracy: {}'.format(accuracy_score(y_test, preds)))
accuracy_score(labels, preds)
accuracy_score(true, preds)
accuracy_score(y_test, test_preds)
print('Training Accuracy: ', accuracy_score(y_train, y_hat_train))
accuracy_score(y_train, training_preds)
'Accuracy Score: {}'.format(accuracy_score(labels, preds))
accuracy_score(y_test, y_preds)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# f1_score as f1_score
#----------------------------------------------------------------------------------------------------
print('Testing F1-Score: ', f1_score(y_test, y_hat_test))
f1_score(true, preds)
f1_score(y_test, y_hat_test)
f1_score(y_train, y_hat_train)
f1_score(y_test, preds)
print('Training F1-Score: ', f1_score(y_train, y_hat_train))
f1 = f1_score(y_test, preds)
'F1 Score: {}'.format(f1_score(labels, preds))
f1 = f1_score(true, preds)
f1_score(labels, preds)
print('F1 Score: {}'.format(f1_score(labels, preds)))
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# roc_curve as roc_curve
#----------------------------------------------------------------------------------------------------
(false_positive_rate, true_positive_rate, thresholds) = roc_curve(y_train, train_pred)
(test_fpr, test_tpr, test_thresholds) = roc_curve(y_test, y_test_score)
roc_curve(y_test, y_hat_test)
roc_curve(y_test, y_pred)
roc_curve(y_test, y_score)
(false_positive_rate, true_positive_rate, thresholds) = roc_curve(y_test, y_pred)
roc_curve(y_train, y_train_score)
(fpr, tpr, thresholds) = roc_curve(y_test, y_score)
roc_curve(y_test, y_preds)
roc_curve(y_train, train_pred)
roc_curve(y_test, y_test_score)
roc_curve(y_train, y_hat_train)
(train_fpr, train_tpr, train_thresholds) = roc_curve(y_train, y_train_score)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# auc as auc
#----------------------------------------------------------------------------------------------------
'AUC: {}'.format(auc(test_fpr, test_tpr))
print('Training AUC: {}'.format(auc(train_fpr, train_tpr)))
'Scikit-learn Model 2 with intercept Train AUC: {}'.format(auc(train_fpr, train_tpr))
auc(test_fpr, test_tpr)
roc_auc = auc(false_positive_rate, true_positive_rate)
'Custom Model Test AUC: {}'.format(auc(test_fpr, test_tpr))
'Custome Model Train AUC: {}'.format(auc(train_fpr, train_tpr))
print('Scikit-learn Model 2 with intercept Train AUC: {}'.format(auc(train_fpr, train_tpr)))
print('Custome Model Train AUC: {}'.format(auc(train_fpr, train_tpr)))
'Scikit-learn Model 2 with intercept Test AUC: {}'.format(auc(test_fpr, test_tpr))
print('Test AUC: {}'.format(auc(test_fpr, test_tpr)))
test_auc = auc(test_fpr, test_tpr)
print('Scikit-learn Model 2 with intercept Test AUC: {}'.format(auc(test_fpr, test_tpr)))
print('Custom Model Test AUC: {}'.format(auc(test_fpr, test_tpr)))
'Train AUC: {}'.format(auc(train_fpr, train_tpr))
'Test AUC: {}'.format(auc(test_fpr, test_tpr))
print('Scikit-learn Model 1 Test AUC: {}'.format(auc(test_fpr, test_tpr)))
train_auc = auc(train_fpr, train_tpr)
print('AUC: {}'.format(auc(train_fpr, train_tpr)))
print('AUC: {}'.format(auc(fpr, tpr)))
'AUC: {}'.format(auc(fpr, tpr))
auc(train_fpr, train_tpr)
auc(false_positive_rate, true_positive_rate)
print('Train AUC: {}'.format(auc(train_fpr, train_tpr)))
print('AUC: {}'.format(auc(test_fpr, test_tpr)))
print('AUC for {}: {}'.format(names[n], auc(fpr, tpr)))
'Training AUC: {}'.format(auc(train_fpr, train_tpr))
'AUC for {}: {}'.format(names[n], auc(fpr, tpr))
'AUC: {}'.format(auc(train_fpr, train_tpr))
'Scikit-learn Model 1 Test AUC: {}'.format(auc(test_fpr, test_tpr))
'Scikit-learn Model 1 Train AUC: {}'.format(auc(train_fpr, train_tpr))
print('Scikit-learn Model 1 Train AUC: {}'.format(auc(train_fpr, train_tpr)))
auc(fpr, tpr)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# seaborn as sns
#----------------------------------------------------------------------------------------------------
sns.set_style('darkgrid', {'axes.facecolor': '0.9'})
sns.set_style('white')
sns.color_palette('Set2')
sns.color_palette('Set2', n_colors=len(names))
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# itertools as itertools
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# SMOTE as SMOTE
#----------------------------------------------------------------------------------------------------
SMOTE().fit_resample(X_train, y_train)
SMOTE()
SMOTE().fit_resample(X, y)
SMOTE().fit_sample(X, y)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# ADASYN as ADASYN
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# log as log
#----------------------------------------------------------------------------------------------------
log(p, 2)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# tree as tree
#----------------------------------------------------------------------------------------------------
tree.plot_tree(clf, feature_names=df.columns, class_names=np.unique(y).astype('str'), filled=True)
tree.plot_tree(classifier_2, feature_names=X.columns, class_names=np.unique(y).astype('str'), filled=True, rounded=True)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# tabulate as tabulate
#----------------------------------------------------------------------------------------------------
tabulate(output, tablefmt='fancy_grid')
print(tabulate([['Entropy']], tablefmt='fancy_grid'))
print(tabulate(output, tablefmt='fancy_grid'))
tabulate([['Entropy']], tablefmt='fancy_grid')
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# matplotlib as mpl
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# mean_absolute_error as mean_absolute_error
#----------------------------------------------------------------------------------------------------
mean_absolute_error(y_test, y_pred)
print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# DecisionTreeRegressor as DecisionTreeRegressor
#----------------------------------------------------------------------------------------------------
DecisionTreeRegressor(min_samples_split=5, max_depth=7, random_state=45)
DecisionTreeRegressor(random_state=42)
regressor = DecisionTreeRegressor(min_samples_split=int(min_samples_split), random_state=45)
DecisionTreeRegressor(max_depth=max_depth, random_state=45)
DecisionTreeRegressor(min_samples_split=int(min_samples_split), random_state=45)
DecisionTreeRegressor(random_state=45)
regressor = DecisionTreeRegressor(max_depth=max_depth, random_state=45)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# r2_score as r2_score
#----------------------------------------------------------------------------------------------------
r2 = r2_score(y_true, y_predict)
r2_score(y_true, y_predict)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# euclidean as euclidean
#----------------------------------------------------------------------------------------------------
euclidean(x, val)
dist_to_i = euclidean(x, val)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# KNeighborsClassifier as KNeighborsClassifier
#----------------------------------------------------------------------------------------------------
knn = KNeighborsClassifier(n_neighbors=k)
KNeighborsClassifier(n_neighbors=k)
KNeighborsClassifier()
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# stats as stats
#----------------------------------------------------------------------------------------------------
cdf_max = stats.norm.cdf(interval_max, loc=mu, scale=std)
stats.norm.cdf(xi_lower, loc=aggs['mean'], scale=aggs['std'])
plt.plot((145, 145), (0, stats.norm.pdf(145, loc=aggs['mean'], scale=aggs['std'])), linestyle='dotted')
stats.norm.pdf(145, loc=aggs['mean'], scale=aggs['std'])
stats.norm.pdf(x, loc=aggs['mean'], scale=aggs['std'])
cdf_min = stats.norm.cdf(interval_min, loc=mu, scale=std)
stats.norm.pdf(ix, loc=aggs['mean'], scale=aggs['std'])
stats.norm.pdf(obs, loc=mu, scale=std)
stats.norm.cdf(xi_upper, loc=aggs['mean'], scale=aggs['std'])
p_x_given_y = stats.norm.pdf(obs, loc=mu, scale=std)
stats.norm.cdf(interval_min, loc=mu, scale=std)
stats.norm.cdf(interval_max, loc=mu, scale=std)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# Polygon as Polygon
#----------------------------------------------------------------------------------------------------
Polygon(verts, facecolor='0.9', edgecolor='0.5')
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# scipy.stats as stats
#----------------------------------------------------------------------------------------------------
cdf_max = stats.norm.cdf(interval_max, loc=mu, scale=std)
stats.norm.cdf(xi_lower, loc=aggs['mean'], scale=aggs['std'])
plt.plot((145, 145), (0, stats.norm.pdf(145, loc=aggs['mean'], scale=aggs['std'])), linestyle='dotted')
stats.norm.pdf(145, loc=aggs['mean'], scale=aggs['std'])
stats.norm.pdf(x, loc=aggs['mean'], scale=aggs['std'])
cdf_min = stats.norm.cdf(interval_min, loc=mu, scale=std)
stats.norm.pdf(ix, loc=aggs['mean'], scale=aggs['std'])
stats.norm.pdf(obs, loc=mu, scale=std)
stats.norm.cdf(xi_upper, loc=aggs['mean'], scale=aggs['std'])
p_x_given_y = stats.norm.pdf(obs, loc=mu, scale=std)
stats.norm.cdf(interval_min, loc=mu, scale=std)
stats.norm.cdf(interval_max, loc=mu, scale=std)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# classification_report as classification_report
#----------------------------------------------------------------------------------------------------
classification_report(y_test, gbt_clf_test_preds)
print(classification_report(target_test, pred))
classification_report(y_test, adaboost_test_preds)
classification_report(target_test, pred)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# BaggingClassifier as BaggingClassifier
#----------------------------------------------------------------------------------------------------
BaggingClassifier(DecisionTreeClassifier(criterion='gini', max_depth=5), n_estimators=20)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# RandomForestClassifier as RandomForestClassifier
#----------------------------------------------------------------------------------------------------
RandomForestClassifier(n_estimators=100, max_depth=5)
RandomForestClassifier()
RandomForestClassifier(n_estimators=5, max_features=10, max_depth=2)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# GridSearchCV as GridSearchCV
#----------------------------------------------------------------------------------------------------
GridSearchCV(dt_clf, dt_param_grid, cv=3, return_train_score=True)
GridSearchCV(rf_clf, rf_param_grid, cv=3)
GridSearchCV(clf, param_grid, scoring='accuracy', cv=None, n_jobs=1)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# AdaBoostClassifier as AdaBoostClassifier
#----------------------------------------------------------------------------------------------------
AdaBoostClassifier(random_state=42)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# GradientBoostingClassifier as GradientBoostingClassifier
#----------------------------------------------------------------------------------------------------
GradientBoostingClassifier(random_state=42)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# XGBClassifier as XGBClassifier
#----------------------------------------------------------------------------------------------------
XGBClassifier()
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# LabelEncoder as LabelEncoder
#----------------------------------------------------------------------------------------------------
LabelEncoder()
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# School as School
#----------------------------------------------------------------------------------------------------
School('Middletown High School')
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# ShoppingCart as ShoppingCart
#----------------------------------------------------------------------------------------------------
ShoppingCart(20)
ShoppingCart()
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# tqdm as tqdm
#----------------------------------------------------------------------------------------------------
tqdm(range(self.total_time_steps))
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# random as random
#----------------------------------------------------------------------------------------------------
random.randrange(1, 10)
matrix[x][y] = random.randrange(1, 10)
M[x][y] = random.randrange(1, 10)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# timeit as timeit
#----------------------------------------------------------------------------------------------------
timeit.default_timer()
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# make_blobs as make_blobs
#----------------------------------------------------------------------------------------------------
make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=3, random_state=123)
make_blobs(n_features=2, centers=2, cluster_std=3, random_state=123)
make_blobs(n_samples=100, n_features=2, centers=4, cluster_std=1.6, random_state=123)
make_blobs(n_features=2, centers=2, cluster_std=1.25, random_state=123)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# cvxpy as cp
#----------------------------------------------------------------------------------------------------
cp.Minimize(cp.norm(w, 2))
cp.norm(w, 2)
cp.Variable()
cp.Minimize(cp.norm(w, 2) + C * (sum(ksi_1) + sum(ksi_2)))
cp.Variable(m)
cp.Variable(n)
cp.Variable(d)
cp.Problem(obj, constraints)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# make_moons as make_moons
#----------------------------------------------------------------------------------------------------
make_moons(n_samples=100, shuffle=False, noise=0.3, random_state=123)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# svm as svm
#----------------------------------------------------------------------------------------------------
svm.SVC(kernel='linear', C=1)
svm.SVC(kernel='poly', coef0=r, gamma=gamma, degree=d)
svm.SVC(kernel='linear', C=0.1)
clf = svm.SVC(kernel='poly', coef0=r, gamma=gamma, degree=d)
clf = svm.SVC(C=C, gamma=gamma)
svm.NuSVC(kernel='linear', nu=0.7)
svm.SVC(kernel='sigmoid', coef0=r, gamma=gamma)
svm.LinearSVC()
clf = svm.SVC(kernel='sigmoid', coef0=r, gamma=gamma)
svm.SVC(C=C, gamma=gamma)
svm.SVC(probability=True)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# SVC as SVC
#----------------------------------------------------------------------------------------------------
SVC(kernel='linear')
SVC(kernel='linear', C=5000000)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# statsmodels as sm
#----------------------------------------------------------------------------------------------------
sm.tools.add_constant(X)
sm.Logit(y, X)
#----------------------------------------------------------------------------------------------------
