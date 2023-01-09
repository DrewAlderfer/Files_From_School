from sklearn.linear_model import (
	Ridge,
	LogisticRegression,
	Lasso,
	LinearRegression,
)
import pandas as pd
from sklearn.preprocessing import (
	PolynomialFeatures,
	MinMaxScaler,
	StandardScaler,
	OneHotEncoder,
	LabelEncoder,
)
import numpy as np
import ride as Ride
import scipy.stats as stats
import statsmodels as sm
from sklearn.metrics import (
	classification_report,
	_mse,
	accuracy_score,
	mean_squared_error,
	plot_confusion_matrix,
	auc,
	f1_score,
	confusion_matrix,
)
import cvxpy as cp # pyright: ignore
import matplotlib.pyplot as plt
import itertools as combinations
import timeit
import csv
import sys
import random
import statsmodels.api as sm
import matplotlib as mpl
import seaborn as sns
from tabulate import tabulate # pyright: ignore
import warnings
# pyright: off, reportGeneralTypeIssues= false, reportUndefinedVariable= false
#----------------------------------------------------------------------------------------------------
# * as *
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# accuracy_score as accuracy_score
#----------------------------------------------------------------------------------------------------
'Accuracy Score: {}'.format(accuracy_score(labels, preds))
'Testing Accuracy for Decision Tree Classifier: {:.4}%'.format(accuracy_score(target_test, pred) * 100)
'Testing Accuracy: {}'.format(accuracy_score(y_test, preds))
acc = accuracy_score(true, preds)
accuracy_score(labels, preds)
accuracy_score(target_test, pred)
accuracy_score(true, preds)
accuracy_score(y_test, preds)
accuracy_score(y_test, test_preds)
accuracy_score(y_test, y_hat_test)
accuracy_score(y_test, y_pred)
accuracy_score(y_test, y_preds)
accuracy_score(y_train, training_preds)
accuracy_score(y_train, y_hat_train)
print('Accuracy Score: {}'.format(accuracy_score(labels, preds)))
print('Testing Accuracy for Decision Tree Classifier: {:.4}%'.format(accuracy_score(target_test, pred) * 100))
print('Testing Accuracy: ', accuracy_score(y_test, y_hat_test))
print('Testing Accuracy: {}'.format(accuracy_score(y_test, preds)))
print('Training Accuracy: ', accuracy_score(y_train, y_hat_train))
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# AdaBoostClassifier as AdaBoostClassifier
#----------------------------------------------------------------------------------------------------
AdaBoostClassifier(random_state=42)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# ADASYN as ADASYN
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# auc as auc
#----------------------------------------------------------------------------------------------------
'AUC for {}: {}'.format(names[n], auc(fpr, tpr))
'AUC: {}'.format(auc(fpr, tpr))
'AUC: {}'.format(auc(test_fpr, test_tpr))
'AUC: {}'.format(auc(train_fpr, train_tpr))
'Custom Model Test AUC: {}'.format(auc(test_fpr, test_tpr))
'Custome Model Train AUC: {}'.format(auc(train_fpr, train_tpr))
'Scikit-learn Model 1 Test AUC: {}'.format(auc(test_fpr, test_tpr))
'Scikit-learn Model 1 Train AUC: {}'.format(auc(train_fpr, train_tpr))
'Scikit-learn Model 2 with intercept Test AUC: {}'.format(auc(test_fpr, test_tpr))
'Scikit-learn Model 2 with intercept Train AUC: {}'.format(auc(train_fpr, train_tpr))
'Test AUC: {}'.format(auc(test_fpr, test_tpr))
'Train AUC: {}'.format(auc(train_fpr, train_tpr))
'Training AUC: {}'.format(auc(train_fpr, train_tpr))
auc(false_positive_rate, true_positive_rate)
auc(fpr, tpr)
auc(test_fpr, test_tpr)
auc(train_fpr, train_tpr)
print('AUC for {}: {}'.format(names[n], auc(fpr, tpr)))
print('AUC: {}'.format(auc(fpr, tpr)))
print('AUC: {}'.format(auc(test_fpr, test_tpr)))
print('AUC: {}'.format(auc(train_fpr, train_tpr)))
print('Custom Model Test AUC: {}'.format(auc(test_fpr, test_tpr)))
print('Custome Model Train AUC: {}'.format(auc(train_fpr, train_tpr)))
print('Scikit-learn Model 1 Test AUC: {}'.format(auc(test_fpr, test_tpr)))
print('Scikit-learn Model 1 Train AUC: {}'.format(auc(train_fpr, train_tpr)))
print('Scikit-learn Model 2 with intercept Test AUC: {}'.format(auc(test_fpr, test_tpr)))
print('Scikit-learn Model 2 with intercept Train AUC: {}'.format(auc(train_fpr, train_tpr)))
print('Test AUC: {}'.format(auc(test_fpr, test_tpr)))
print('Train AUC: {}'.format(auc(train_fpr, train_tpr)))
print('Training AUC: {}'.format(auc(train_fpr, train_tpr)))
roc_auc = auc(false_positive_rate, true_positive_rate)
test_auc = auc(test_fpr, test_tpr)
train_auc = auc(train_fpr, train_tpr)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# BaggingClassifier as BaggingClassifier
#----------------------------------------------------------------------------------------------------
BaggingClassifier(DecisionTreeClassifier(criterion='gini', max_depth=5), n_estimators=20)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# classification_report as classification_report
#----------------------------------------------------------------------------------------------------
classification_report(target_test, pred)
classification_report(y_test, adaboost_test_preds)
classification_report(y_test, gbt_clf_test_preds)
print(classification_report(target_test, pred))
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# comb as comb
#----------------------------------------------------------------------------------------------------
comb(a + b, a)
y.append(comb(a + b, a))
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# combinations as combinations
#----------------------------------------------------------------------------------------------------
combinations(X_train.columns, 2)
list(combinations(X_train.columns, 2))
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# confusion_matrix as confusion_matrix
#----------------------------------------------------------------------------------------------------
confusion_matrix(target_test, pred)
confusion_matrix(test_predictions, y_test)
confusion_matrix(y_test, adaboost_test_preds)
confusion_matrix(y_test, gbt_clf_test_preds)
confusion_matrix(y_test, y_hat_test)
print(confusion_matrix(target_test, pred))
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# cvxpy as cp
#----------------------------------------------------------------------------------------------------
cp.Minimize(cp.norm(w, 2) + C * (sum(ksi_1) + sum(ksi_2)))
cp.Minimize(cp.norm(w, 2))
cp.norm(w, 2)
cp.Problem(obj, constraints)
cp.Variable()
cp.Variable(d)
cp.Variable(m)
cp.Variable(n)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# cross_val_score as cross_val_score
#----------------------------------------------------------------------------------------------------
cross_val_score(adaboost_clf, df, target, cv=5)
cross_val_score(adaboost_clf, df, target, cv=5).mean()
cross_val_score(dt_clf, X_train, y_train, cv=3)
cross_val_score(gbt_clf, df, target, cv=5)
cross_val_score(gbt_clf, df, target, cv=5).mean()
cross_val_score(linreg, X, y, cv=5, scoring='neg_mean_squared_error')
cross_val_score(rf_clf, X_train, y_train, cv=3)
np.mean(cross_val_score(rf_clf, X_train, y_train, cv=3))
print(cross_val_score(adaboost_clf, df, target, cv=5).mean())
print(cross_val_score(gbt_clf, df, target, cv=5).mean())
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# csv as csv
#----------------------------------------------------------------------------------------------------
csv.reader(f)
raw = csv.reader(f)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# DecisionTreeClassifier as DecisionTreeClassifier
#----------------------------------------------------------------------------------------------------
BaggingClassifier(DecisionTreeClassifier(criterion='gini', max_depth=5), n_estimators=20)
DecisionTreeClassifier()
DecisionTreeClassifier(criterion='entropy')
DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, random_state=SEED)
DecisionTreeClassifier(criterion='entropy', max_features=6, max_depth=3, min_samples_split=0.7, min_samples_leaf=0.25, random_state=SEED)
DecisionTreeClassifier(criterion='entropy', max_features=max_feature, random_state=SEED)
DecisionTreeClassifier(criterion='entropy', min_samples_leaf=min_samples_leaf, random_state=SEED)
DecisionTreeClassifier(criterion='entropy', min_samples_split=min_samples_split, random_state=SEED)
DecisionTreeClassifier(criterion='entropy', random_state=SEED)
DecisionTreeClassifier(criterion='gini', max_depth=5)
DecisionTreeClassifier(random_state=10)
DecisionTreeClassifier(random_state=10, criterion='entropy')
dt = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, random_state=SEED)
dt = DecisionTreeClassifier(criterion='entropy', max_features=max_feature, random_state=SEED)
dt = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=min_samples_leaf, random_state=SEED)
dt = DecisionTreeClassifier(criterion='entropy', min_samples_split=min_samples_split, random_state=SEED)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# DecisionTreeRegressor as DecisionTreeRegressor
#----------------------------------------------------------------------------------------------------
DecisionTreeRegressor(max_depth=max_depth, random_state=45)
DecisionTreeRegressor(min_samples_split=5, max_depth=7, random_state=45)
DecisionTreeRegressor(min_samples_split=int(min_samples_split), random_state=45)
DecisionTreeRegressor(random_state=42)
DecisionTreeRegressor(random_state=45)
regressor = DecisionTreeRegressor(max_depth=max_depth, random_state=45)
regressor = DecisionTreeRegressor(min_samples_split=int(min_samples_split), random_state=45)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# Driver as Driver
#----------------------------------------------------------------------------------------------------
Driver()
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# euclidean as euclidean
#----------------------------------------------------------------------------------------------------
dist_to_i = euclidean(x, val)
euclidean(x, val)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# f1_score as f1_score
#----------------------------------------------------------------------------------------------------
'F1 Score: {}'.format(f1_score(labels, preds))
f1 = f1_score(true, preds)
f1 = f1_score(y_test, preds)
f1_score(labels, preds)
f1_score(true, preds)
f1_score(y_test, preds)
f1_score(y_test, y_hat_test)
f1_score(y_train, y_hat_train)
print('F1 Score: {}'.format(f1_score(labels, preds)))
print('Testing F1-Score: ', f1_score(y_test, y_hat_test))
print('Training F1-Score: ', f1_score(y_train, y_hat_train))
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# FunctionTransformer as FunctionTransformer
#----------------------------------------------------------------------------------------------------
FunctionTransformer(np.log, validate=True)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# GradientBoostingClassifier as GradientBoostingClassifier
#----------------------------------------------------------------------------------------------------
GradientBoostingClassifier(random_state=42)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# GridSearchCV as GridSearchCV
#----------------------------------------------------------------------------------------------------
GridSearchCV(clf, param_grid, scoring='accuracy', cv=None, n_jobs=1)
GridSearchCV(dt_clf, dt_param_grid, cv=3, return_train_score=True)
GridSearchCV(rf_clf, rf_param_grid, cv=3)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# itertools as itertools
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# KNeighborsClassifier as KNeighborsClassifier
#----------------------------------------------------------------------------------------------------
KNeighborsClassifier()
KNeighborsClassifier(n_neighbors=k)
knn = KNeighborsClassifier(n_neighbors=k)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# LabelEncoder as LabelEncoder
#----------------------------------------------------------------------------------------------------
LabelEncoder()
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# Lasso as Lasso
#----------------------------------------------------------------------------------------------------
lasso = Lasso(alpha=alpha)
Lasso()
Lasso(alpha=1)
Lasso(alpha=10)
Lasso(alpha=10000)
Lasso(alpha=alpha)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# LinearRegression as LinearRegression
#----------------------------------------------------------------------------------------------------
LinearRegression()
LinearRegression().fit(features_train, y_train)
LinearRegression().fit(features_train, y_train).score(features_test, y_test)
lr = LinearRegression()
rfe = RFE(LinearRegression(), n_features_to_select=n)
RFE(LinearRegression(), n_features_to_select=n)
score = LinearRegression().fit(features_train, y_train).score(features_test, y_test)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# load_iris as load_iris
#----------------------------------------------------------------------------------------------------
load_iris()
load_iris(return_X_y=True, as_frame=True)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# log as log
#----------------------------------------------------------------------------------------------------
log(p, 2)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# LogisticRegression as LogisticRegression
#----------------------------------------------------------------------------------------------------
LogisticRegression(C=1000000000000.0, fit_intercept=False, solver='liblinear')
LogisticRegression(fit_intercept=False, C=1000000000000.0, solver='liblinear')
LogisticRegression(fit_intercept=False, C=1e+16, solver='liblinear')
LogisticRegression(fit_intercept=False, C=1e+20, solver='liblinear')
LogisticRegression(fit_intercept=False, C=1e+25, solver='liblinear')
LogisticRegression(fit_intercept=False, C=c, solver='liblinear')
LogisticRegression(fit_intercept=False, solver='liblinear')
LogisticRegression(fit_intercept=True, C=1.5 ** n, solver='liblinear')
LogisticRegression(fit_intercept=True, C=1e+16, solver='liblinear')
logreg = LogisticRegression(fit_intercept=False, C=1e+20, solver='liblinear')
logreg = LogisticRegression(fit_intercept=False, C=1e+25, solver='liblinear')
logreg = LogisticRegression(fit_intercept=False, C=c, solver='liblinear')
logreg = LogisticRegression(fit_intercept=True, C=1.5 ** n, solver='liblinear')
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# make_blobs as make_blobs
#----------------------------------------------------------------------------------------------------
make_blobs(n_features=2, centers=2, cluster_std=1.25, random_state=123)
make_blobs(n_features=2, centers=2, cluster_std=3, random_state=123)
make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=3, random_state=123)
make_blobs(n_samples=100, n_features=2, centers=4, cluster_std=1.6, random_state=123)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# make_moons as make_moons
#----------------------------------------------------------------------------------------------------
make_moons(n_samples=100, shuffle=False, noise=0.3, random_state=123)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# mean_absolute_error as mean_absolute_error
#----------------------------------------------------------------------------------------------------
mean_absolute_error(y_test, y_pred)
print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# mean_squared_error as mean_squared_error
#----------------------------------------------------------------------------------------------------
inner_test_mses.append(mean_squared_error(y_test, y_hat_test))
inner_train_mses.append(mean_squared_error(y_train, y_hat_train))
mean_squared_error(y_test, lasso.predict(X_test_preprocessed))
mean_squared_error(y_test, linreg.predict(X_test_preprocessed))
mean_squared_error(y_test, ridge.predict(X_test_preprocessed))
mean_squared_error(y_test, test_preds)
mean_squared_error(y_true, y_predict, squared=False)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# MinMaxScaler as MinMaxScaler
#----------------------------------------------------------------------------------------------------
MinMaxScaler()
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# matplotlib as mpl
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# numpy as np
#----------------------------------------------------------------------------------------------------
diff_mu_ai_bi = np.mean(ai) - np.mean(bi)
FunctionTransformer(np.log, validate=True)
np.zeros(len(m_current))
np.abs(y_train - y_hat_train)
np.arange(0, 14.5, step=0.5)
np.arange(2, 11)
np.argmax(c_probs)
np.array(x)
np.bincount(labels)
np.ceil(np.max([x[:, 1], y[:, 1]]))
np.concatenate([X_test_cont, X_test_cat.todense()], axis=1)
np.dot(X, weights)
np.floor(np.min([x[:, 1], y[:, 1]]))
np.gradient(rss_survey_region)
np.isin(all_idx, training_idx)
np.linalg.inv(A)
np.linalg.solve(A, B.T)
np.linspace(-10, 10, 100)
np.linspace(start=-3, stop=5, num=10 ** 3)
np.log(ames_cont)
np.matrix([[1, 1, 1], [0.1, 0.2, 0.3], [2, 0, -1]])
np.matrix([[1, 1, 1], [0.5, 0.75, 1.25], [-2, 1, 0]])
np.max([x[:, 0], y[:, 0]])
np.mean([yi ** 2 for yi in y_hat])
np.meshgrid(x12_coord, x11_coord)
np.meshgrid(x22_coord, x21_coord)
np.meshgrid(x2_coord, x1_coord)
np.min([x[:, 0], y[:, 0]])
np.min([x[:, 1], y[:, 1]])
np.ones((X.shape[1], 1))
np.ones((X.shape[1], 1)).flatten()
np.power(np.abs(np.array(a) - np.array(b)), c)
np.random.choice(all_idx, size=round(546 * 0.8), replace=False)
np.random.choice(self.population)
np.random.choice(union, size=len(a), replace=False)
np.random.seed(42)
np.round(y_hat_test, 2)
np.savetxt(sys.stdout, bval_rss, '%16.2f')
np.set_printoptions(formatter={'float_kind': '{:f}'.format})
np.shape(X_train_scaled)
np.sqrt(mean_sq_err)
np.sqrt(mean_squared_error(y_test, y_pred))
np.sum(np.power(np.abs(np.array(a) - np.array(b)), c))
np.transpose(A.dot(B))
np.transpose(B)
np.unique(y)
np.unique(y).astype('str')
np.zeros((3, 3))
sigmoid(np.dot(X, weights))
sigmoid(np.dot(X_test, weights))
sigmoid(np.dot(X_train, weights))
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# OneHotEncoder as OneHotEncoder
#----------------------------------------------------------------------------------------------------
OneHotEncoder(drop='first', sparse=False)
OneHotEncoder(handle_unknown='ignore')
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# pandas as pd
#----------------------------------------------------------------------------------------------------
col_transformed_test = pd.DataFrame(poly.transform(features_test[[col]]))
col_transformed_train = pd.DataFrame(poly.fit_transform(features_train[[col]]))
col_transformed_train = pd.DataFrame(poly.fit_transform(X_train[[col]]), columns=poly.get_feature_names([col]))
col_transformed_val = pd.DataFrame(poly.fit_transform(X_val[[col]]), columns=poly.get_feature_names([col]))

features_test = pd.concat([features_test.drop(col, axis=1), col_transformed_test], axis=1)
features_train = pd.concat([features_train.drop(col, axis=1), col_transformed_train], axis=1)
final_model.fit(pd.concat([X_train, X_test]), pd.concat([y_train, y_test]))

pd.concat([ames_log_norm, ames_ohe], axis=1)
pd.concat([fold for (i, fold) in enumerate(X_folds) if i != n])
pd.concat([pd.DataFrame(log_transformer.transform(X_test[continuous]), index=X_test.index), pd.DataFrame(ohe.transform(X_test[categoricals]), index=X_test.index)], axis=1)
pd.concat([X_test, y_test], axis=1)
pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)

pd.DataFrame()
pd.DataFrame(initial_weights)
pd.DataFrame(log_transformer.transform(X_test[continuous]), index=X_test.index)
pd.DataFrame(ohe.transform(X_test[categoricals]), index=X_test.index)
pd.DataFrame(poly.fit_transform(features_train[[col]]))
pd.DataFrame(poly.transform(X_test[[col]]), columns=poly.get_feature_names([col]))
pd.DataFrame({'color': ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet']})

pd.get_dummies(ames[categoricals], prefix=categoricals, drop_first=True)
pd.get_dummies(df)
pd.get_dummies(df['class'], drop_first=True)
pd.get_dummies(df[relevant_columns], drop_first=True, dtype=float)
pd.get_dummies(salaries)
pd.get_dummies(salaries['Target'], drop_first=True)
pd.get_dummies(salaries[xcols], drop_first=True)
pd.get_dummies(X, drop_first=True)
pd.plotting.scatter_matrix(df, figsize=(10, 10))
pd.read_csv('./data_banknote_authentication.csv', header=None, names=['Variance', 'Skewness', 'Kurtosis', 'Entropy'])
pd.read_csv('./mushrooms.csv')
pd.read_csv('ames.csv')
pd.read_csv('ames.csv', index_col=0)
pd.read_csv('creditcard.csv.gz', compression='gzip')
pd.read_csv('data_banknote_authentication.csv', header=None)
pd.read_csv('heart.csv')
pd.read_csv('housing_prices.csv', index_col=0)
pd.read_csv('mushrooms.csv')
pd.read_csv('petrol_consumption.csv')
pd.read_csv('pima-indians-diabetes.csv')
pd.read_csv('salaries_final.csv', index_col=0)
pd.read_csv('simulation.csv')
pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'text'])
pd.read_csv('titanic.csv')
pd.read_csv('winequality-red.csv')
pd.read_excel('movie_data.xlsx')
pd.read_excel('movie_data_detailed_with_ols.xlsx')
pd.Series(encoder.fit_transform(y_train))
pd.Series(encoder.transform(y_test))
pd.Series(residuals)
pd.Series(residuals).value_counts()
pd.Series(residuals).value_counts(normalize=True)
pd.Series(y_resampled)
pd.Series(y_resampled).value_counts()
pd.Series(y_train_resampled)
pd.Series(y_train_resampled).value_counts()
print(pd.Series(residuals).value_counts())
print(pd.Series(residuals).value_counts(normalize=True))
print(pd.Series(y_resampled).value_counts())
print(pd.Series(y_train_resampled).value_counts())
self.time_step_statistics_df = pd.DataFrame()
weights_col = pd.concat([weights_col, pd.DataFrame(weights)], axis=1)
weights_col = pd.DataFrame(initial_weights)
X_test = pd.concat([pd.DataFrame(log_transformer.transform(X_test[continuous]), index=X_test.index), pd.DataFrame(ohe.transform(X_test[categoricals]), index=X_test.index)], axis=1)
X_test = pd.concat([X_test.drop(col, axis=1), col_transformed_test], axis=1)
X_train = pd.concat([fold for (i, fold) in enumerate(X_folds) if i != n])
X_train = pd.concat([pd.DataFrame(log_transformer.transform(X_train[continuous]), index=X_train.index), pd.DataFrame(ohe.transform(X_train[categoricals]), index=X_train.index)], axis=1)
X_train = pd.concat([X_train.drop(col, axis=1), col_transformed_train], axis=1)
X_val = pd.concat([X_val.drop(col, axis=1), col_transformed_val], axis=1)
y_train = pd.concat([fold for (i, fold) in enumerate(y_folds) if i != n])
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# plot_confusion_matrix as plot_confusion_matrix
#----------------------------------------------------------------------------------------------------
plot_confusion_matrix(classifier, X, y, values_format='.3g')
plot_confusion_matrix(clf, X, y, values_format='.3g')
plot_confusion_matrix(logreg, X_test, y_test, cmap=plt.cm.Blues)
plot_confusion_matrix(logreg, X_test, y_test, display_labels=['not fraud', 'fraud'], values_format='.5g')
plot_confusion_matrix(model_log, X_test, y_test)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# matplotlib.pyplot as plt
#----------------------------------------------------------------------------------------------------
(fig, ax) = plt.subplots(figsize=(10, 7))
plot_confusion_matrix(logreg, X_test, y_test, cmap=plt.cm.Blues)
plt.axhline(y=0, color='lightgrey')
plt.axis('tight')
plt.axvline(x=0, color='lightgrey')
plt.barh(range(n_features), model.feature_importances_, align='center')
plt.boxplot([df[col] for col in df.columns])
plt.contourf(X1_C, X2_C, Z, alpha=1)
plt.figure(figsize=(10, 10))
plt.figure(figsize=(10, 4))
plt.figure(figsize=(10, 8))
plt.figure(figsize=(11, 11))
plt.figure(figsize=(12, 12))
plt.figure(figsize=(12, 12), dpi=500)
plt.figure(figsize=(12, 14))
plt.figure(figsize=(12, 6))
plt.figure(figsize=(20, 10))
plt.figure(figsize=(20, 5))
plt.figure(figsize=(5, 5))
plt.figure(figsize=(7, 6))
plt.figure(figsize=(8, 5))
plt.figure(figsize=(8, 8))
plt.gca()
plt.grid(False)
plt.hlines(y=y_val, xmin=x_value, xmax=x_value + delta_x, color='lightgreen', label=hline_lab)
plt.legend()
plt.legend(bbox_to_anchor=(1, 1))
plt.legend(loc='lower right')
plt.legend(loc='upper left')
plt.legend(loc='upper left', bbox_to_anchor=[0, 1], ncol=2, fancybox=True)
plt.legend(loc=(1.01, 0.85))
plt.plot((145, 145), (0, stats.norm.pdf(145, loc=aggs['mean'], scale=aggs['std'])), linestyle='dotted')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.plot([d1_min, d1_max], [d2_at_mind1, d2_at_maxd1], color='black')
plt.plot([d1_min, d1_max], [sup_dn_at_mind1, sup_dn_at_maxd1], '-.', color='blue')
plt.plot([d1_min, d1_max], [sup_up_at_mind1, sup_up_at_maxd1], '-.', color='blue')
plt.plot(data[col], target, 'o')
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve')
plt.plot(fpr, tpr, color=colors[n], lw=lw, label='ROC curve Normalization Weight: {}'.format(names[n]))
plt.plot(fpr, tpr, color=colors[n], lw=lw, label='ROC curve Regularization Weight: {}'.format(names[n]))
plt.plot(max_depths, mse_results, 'r', label='RMSE')
plt.plot(max_depths, r2_results, 'b', label='R2')
plt.plot(max_depths, test_results, 'r', label='Test AUC')
plt.plot(max_depths, train_results, 'b', label='Train AUC')
plt.plot(max_features, test_results, 'r', label='Test AUC')
plt.plot(max_features, train_results, 'b', label='Train AUC')
plt.plot(min_samples_leafs, test_results, 'r', label='Test AUC')
plt.plot(min_samples_leafs, train_results, 'b', label='Train AUC')
plt.plot(min_samples_splits, mse_results, 'r', label='RMSE')
plt.plot(min_samples_splits, r2_results, 'b', label='R2')
plt.plot(min_samples_splits, test_results, 'r', label='Test AUC')
plt.plot(min_samples_splits, train_results, 'b', label='Train AUC')
plt.plot(range_stds, test_accs, label='Test Accuracy')
plt.plot(range_stds, train_accs, label='Train Accuracy')
plt.plot(tan_line['x_dev'], tan_line['tan'], color='yellow', label=tan_line['lab'])
plt.plot(test_fpr, test_tpr, color='darkorange', lw=lw, label='Custom Model Test ROC curve')
plt.plot(test_fpr, test_tpr, color='darkorange', lw=lw, label='ROC curve')
plt.plot(test_fpr, test_tpr, color='darkorange', lw=lw, label='Test ROC curve')
plt.plot(test_fpr, test_tpr, color='purple', lw=lw, label='Scikit learn Model 2 with intercept Test ROC curve')
plt.plot(test_fpr, test_tpr, color='yellow', lw=lw, label='Scikit learn Model 1 Test ROC curve')
plt.plot(train_fpr, train_tpr, color='blue', lw=lw, label='Custom Model Train ROC curve')
plt.plot(train_fpr, train_tpr, color='blue', lw=lw, label='Train ROC curve')
plt.plot(train_fpr, train_tpr, color='darkorange', lw=lw, label='ROC curve')
plt.plot(train_fpr, train_tpr, color='gold', lw=lw, label='Scikit learn Model 1 Train ROC curve')
plt.plot(train_fpr, train_tpr, color='red', lw=lw, label='Scikit learn Model 2 with intercept Train ROC curve')
plt.plot(x, pdf)
plt.plot(x, y)
plt.plot(x, y, '.b')
plt.plot(x_values, derivative_values, color='darkorange', label="f '(x) = 6x")
plt.plot(x_values, derivative_values, color='darkorange', label="f '(x)")
plt.plot(x_values, function_values, label='f (x) = 3x^2\N{MINUS SIGN}11 ')
plt.plot(x_values, function_values, label='f (x)')
plt.plot(x_values, y_values, label='3x^2 + 11')
plt.plot(x_values, y_values, label='3x^2 - 11')
plt.plot(x_values, y_values, label='4x + 15')
plt.plot(y_pred, linestyle='-', marker='o', label='predictions')
plt.plot(y_test, linestyle='-', marker='o', label='actual values')
plt.plot(y_test, y_test, label='Actual data')
plt.plot(y_train, y_train, label='Actual data')
plt.savefig('./decision_tree.png')
plt.scatter(1.1124498053361267, rss(1.1124498053361267), c='red')
plt.scatter(df['budget'], df['domgross'], label='Actual Data Points')
plt.scatter(list(range(10, 95)), testing_accuracy, label='testing_accuracy')
plt.scatter(list(range(10, 95)), testing_f1, label='testing_f1')
plt.scatter(list(range(10, 95)), testing_precision, label='testing_precision')
plt.scatter(list(range(10, 95)), testing_recall, label='testing_recall')
plt.scatter(list(range(10, 95)), training_accuracy, label='training_accuracy')
plt.scatter(list(range(10, 95)), training_f1, label='training_f1')
plt.scatter(list(range(10, 95)), training_precision, label='training_precision')
plt.scatter(list(range(10, 95)), training_recall, label='training_recall')
plt.scatter(x, 1.331 * x, label='Median Ratio Model')
plt.scatter(x, 1.575 * x, label='Mean Ratio Model')
plt.scatter(X1, X2, c=y, edgecolors='k')
plt.scatter(X1, X2, c=y_test, edgecolors='gray')
plt.scatter(X1, X2, c=y_train, edgecolors='gray')
plt.scatter(X[:, 0], X[:, 1], c=labels, s=25)
plt.scatter(X[:, 0], X[:, 1], c=y, s=25)
plt.scatter(x[:, 0], x[:, 1], color='purple')
plt.scatter(X_11, X_12, c=y_1)
plt.scatter(X_1[:, 0], X_1[:, 1], c=y_1, s=25)
plt.scatter(X_21, X_22, c=y_2)
plt.scatter(X_2[:, 0], X_2[:, 1], c=y_2, s=25)
plt.scatter(X_3[:, 0], X_3[:, 1], c=y_3, edgecolors='gray')
plt.scatter(X_3[:, 0], X_3[:, 1], c=y_3, s=25)
plt.scatter(X_4[:, 0], X_4[:, 1], c=y_4, edgecolors='gray')
plt.scatter(X_4[:, 0], X_4[:, 1], c=y_4, s=25)
plt.scatter(y[:, 0], y[:, 1], color='yellow')
plt.scatter(y_test, lm_test_predictions, label='Model')
plt.scatter(y_test, poly_test_predictions, label='Model')
plt.scatter(y_train, lm_train_predictions, label='Model')
plt.scatter(y_train, poly_train_predictions, label='Model')
plt.show()
plt.style.use('ggplot')
plt.style.use('seaborn')
plt.subplot(1, 3, i + 1)
plt.subplot(121)
plt.subplot(122)
plt.subplot(221)
plt.subplot(222)
plt.subplot(223)
plt.subplot(224)
plt.subplot(3, 3, k + 1)
plt.subplot(4, 2, k + 1)
plt.subplots()
plt.subplots(1, 2, figsize=(10, 5), sharey=True)
plt.subplots(4, 2, figsize=(15, 15))
plt.subplots(figsize=(10, 4))
plt.subplots(figsize=(10, 6))
plt.subplots(figsize=(10, 7))
plt.subplots(figsize=(12, 5))
plt.subplots(figsize=(12, 6))
plt.subplots(nrows=1, ncols=1, figsize=(12, 12), dpi=300, tight_layout=True)
plt.tight_layout()
plt.title(' gam= %r, r = %r , score = %r' % (gamma, r, round(clf.score(X_3, y_3), 2)))
plt.title('Actual vs. predicted values')
plt.title('Box plot of all columns in dataset')
plt.title('Combination sample space of a 25 observation sample compared to various second sample sizes')
plt.title('Conditional Probability of Resting Blood Pressure ~145 for Those With Heart Disease')
plt.title('d= %r, gam= %r, r = %r , score = %r' % (d, gamma, r, round(clf.score(X_3, y_3), 2)))
plt.title('d= %r, gam= %r, r = %r , score = %r' % (d, gamma, r, round(clf.score(X_test, y_test), 2)))
plt.title('d= %r, gam= %r, r = %r , score = %r' % (d, gamma, r, round(clf.score(X_train, y_train), 2)))
plt.title('Four blobs with Varying Separability')
plt.title('Four Blobs with Varying Separability')
plt.title('Four Blobs')
plt.title('Four blobs')
plt.title('gam= %r, C= %r, score = %r' % (gamma, C, round(clf.score(X_4, y_4), 2)))
plt.title('Gross Domestic Sales vs. Budget', fontsize=18)
plt.title('LinearSVC')
plt.title('Model vs data for test set')
plt.title('Model vs data for training set')
plt.title('NuSVC, nu=0.5')
plt.title('Receiver operating characteristic (ROC) Curve for Test Set')
plt.title('Receiver operating characteristic (ROC) Curve for Training Set')
plt.title('Receiver operating characteristic (ROC) Curve')
plt.title('RSS Loss Function for Various Values of m')
plt.title('RSS Loss Function for Various Values of m, with minimum marked')
plt.title('SVC, C=0.1')
plt.title('SVC, C=1')
plt.title('Train and Test Accruaccy Versus Various Standard Deviation Bin Ranges for GNB')
plt.title('Two Blobs with Mild Overlap')
plt.title('Two blobs')
plt.title('Two interleaving half circles')
plt.title('Two Moons with Substantial Overlap')
plt.title('Two Seperable Blobs')
plt.title(col)
plt.vlines(x=x_value + delta_x, ymin=y_val, ymax=y_val_max, color='darkorange', label=vline_lab)
plt.xlabel('Budget', fontsize=16)
plt.xlabel('False Positive Rate')
plt.xlabel('Feature importance')
plt.xlabel('max features')
plt.xlabel('Min. Sample Leafs')
plt.xlabel('Min. Sample splits')
plt.xlabel('Resting Blood Pressure')
plt.xlabel('Size of second sample')
plt.xlabel('Standard Deviations Used for Integral Band Width')
plt.xlabel('Tree Depth')
plt.xlabel('Tree depth')
plt.xlabel('x', fontsize=14)
plt.xlabel(col)
plt.xlim([0.0, 1.0])
plt.xticks([i / 20.0 for i in range(21)])
plt.xticks(range(len(df.columns.values)), df.columns.values)
plt.ylabel('AUC score')
plt.ylabel('Classifier Accuracy')
plt.ylabel('Feature')
plt.ylabel('Gross Domestic Sales', fontsize=16)
plt.ylabel('Number of combinations for permutation test')
plt.ylabel('Prices')
plt.ylabel('Probability Density')
plt.ylabel('R-squared')
plt.ylabel('RMSE')
plt.ylabel('True Positive Rate')
plt.ylabel('y', fontsize=14)
plt.ylim([0.0, 1.05])
plt.ylim([np.floor(np.min([x[:, 1], y[:, 1]])), np.ceil(np.max([x[:, 1], y[:, 1]]))])
plt.yticks([i / 20.0 for i in range(21)])
plt.yticks(np.arange(n_features), data_train.columns.values)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# Polygon as Polygon
#----------------------------------------------------------------------------------------------------
Polygon(verts, facecolor='0.9', edgecolor='0.5')
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# PolynomialFeatures as PolynomialFeatures
#----------------------------------------------------------------------------------------------------
poly = PolynomialFeatures(degree, include_bias=False)
PolynomialFeatures(3)
PolynomialFeatures(degree, include_bias=False)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# precision_score as precision_score
#----------------------------------------------------------------------------------------------------
'Precision Score: {}'.format(precision_score(labels, preds))
precision_score(labels, preds)
precision_score(y_test, y_hat_test)
precision_score(y_train, y_hat_train)
print('Precision Score: {}'.format(precision_score(labels, preds)))
print('Testing Precision: ', precision_score(y_test, y_hat_test))
print('Training Precision: ', precision_score(y_train, y_hat_train))
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# r2_score as r2_score
#----------------------------------------------------------------------------------------------------
r2 = r2_score(y_true, y_predict)
r2_score(y_true, y_predict)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# random as random
#----------------------------------------------------------------------------------------------------
M[x][y] = random.randrange(1, 10)
matrix[x][y] = random.randrange(1, 10)
random.randrange(1, 10)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# RandomForestClassifier as RandomForestClassifier
#----------------------------------------------------------------------------------------------------
RandomForestClassifier()
RandomForestClassifier(n_estimators=100, max_depth=5)
RandomForestClassifier(n_estimators=5, max_features=10, max_depth=2)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# rcParams as rcParams
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# recall_score as recall_score
#----------------------------------------------------------------------------------------------------
'Recall Score: {}'.format(recall_score(labels, preds))
print('Recall Score: {}'.format(recall_score(labels, preds)))
print('Testing Recall: ', recall_score(y_test, y_hat_test))
print('Training Recall: ', recall_score(y_train, y_hat_train))
recall_score(labels, preds)
recall_score(y_test, y_hat_test)
recall_score(y_train, y_hat_train)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# RFE as RFE
#----------------------------------------------------------------------------------------------------
rfe = RFE(LinearRegression(), n_features_to_select=n)
RFE(LinearRegression(), n_features_to_select=n)
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
# Ridge as Ridge
#----------------------------------------------------------------------------------------------------
Ridge()
Ridge(alpha=10)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# roc_curve as roc_curve
#----------------------------------------------------------------------------------------------------
(false_positive_rate, true_positive_rate, thresholds) = roc_curve(y_test, y_pred)
(false_positive_rate, true_positive_rate, thresholds) = roc_curve(y_train, train_pred)
(fpr, tpr, thresholds) = roc_curve(y_test, y_score)
(test_fpr, test_tpr, test_thresholds) = roc_curve(y_test, y_test_score)
(train_fpr, train_tpr, train_thresholds) = roc_curve(y_train, y_train_score)
roc_curve(y_test, y_hat_test)
roc_curve(y_test, y_pred)
roc_curve(y_test, y_preds)
roc_curve(y_test, y_score)
roc_curve(y_test, y_test_score)
roc_curve(y_train, train_pred)
roc_curve(y_train, y_hat_train)
roc_curve(y_train, y_train_score)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# School as School
#----------------------------------------------------------------------------------------------------
School('Middletown High School')
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# ShoppingCart as ShoppingCart
#----------------------------------------------------------------------------------------------------
ShoppingCart()
ShoppingCart(20)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# SimpleImputer as SimpleImputer
#----------------------------------------------------------------------------------------------------
SimpleImputer(strategy='constant', fill_value='missing')
SimpleImputer(strategy='median')
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# statsmodels.api as sm
#----------------------------------------------------------------------------------------------------
sm.Logit(y, X)
sm.tools.add_constant(X)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# statsmodels as sm
#----------------------------------------------------------------------------------------------------
sm.Logit(y, X)
sm.tools.add_constant(X)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# SMOTE as SMOTE
#----------------------------------------------------------------------------------------------------
SMOTE()
SMOTE().fit_resample(X, y)
SMOTE().fit_resample(X_train, y_train)
SMOTE().fit_sample(X, y)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# seaborn as sns
#----------------------------------------------------------------------------------------------------
sns.color_palette('Set2')
sns.color_palette('Set2', n_colors=len(names))
sns.set_style('darkgrid', {'axes.facecolor': '0.9'})
sns.set_style('white')
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# StandardScaler as StandardScaler
#----------------------------------------------------------------------------------------------------
StandardScaler()
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# stats as stats
#----------------------------------------------------------------------------------------------------
cdf_max = stats.norm.cdf(interval_max, loc=mu, scale=std)
cdf_min = stats.norm.cdf(interval_min, loc=mu, scale=std)
p_x_given_y = stats.norm.pdf(obs, loc=mu, scale=std)
plt.plot((145, 145), (0, stats.norm.pdf(145, loc=aggs['mean'], scale=aggs['std'])), linestyle='dotted')
stats.norm.cdf(interval_max, loc=mu, scale=std)
stats.norm.cdf(interval_min, loc=mu, scale=std)
stats.norm.cdf(xi_lower, loc=aggs['mean'], scale=aggs['std'])
stats.norm.cdf(xi_upper, loc=aggs['mean'], scale=aggs['std'])
stats.norm.pdf(145, loc=aggs['mean'], scale=aggs['std'])
stats.norm.pdf(ix, loc=aggs['mean'], scale=aggs['std'])
stats.norm.pdf(obs, loc=mu, scale=std)
stats.norm.pdf(x, loc=aggs['mean'], scale=aggs['std'])
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# scipy.stats as stats
#----------------------------------------------------------------------------------------------------
cdf_max = stats.norm.cdf(interval_max, loc=mu, scale=std)
cdf_min = stats.norm.cdf(interval_min, loc=mu, scale=std)
p_x_given_y = stats.norm.pdf(obs, loc=mu, scale=std)
plt.plot((145, 145), (0, stats.norm.pdf(145, loc=aggs['mean'], scale=aggs['std'])), linestyle='dotted')
stats.norm.cdf(interval_max, loc=mu, scale=std)
stats.norm.cdf(interval_min, loc=mu, scale=std)
stats.norm.cdf(xi_lower, loc=aggs['mean'], scale=aggs['std'])
stats.norm.cdf(xi_upper, loc=aggs['mean'], scale=aggs['std'])
stats.norm.pdf(145, loc=aggs['mean'], scale=aggs['std'])
stats.norm.pdf(ix, loc=aggs['mean'], scale=aggs['std'])
stats.norm.pdf(obs, loc=mu, scale=std)
stats.norm.pdf(x, loc=aggs['mean'], scale=aggs['std'])
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# SVC as SVC
#----------------------------------------------------------------------------------------------------
SVC(kernel='linear')
SVC(kernel='linear', C=5000000)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# svm as svm
#----------------------------------------------------------------------------------------------------
clf = svm.SVC(C=C, gamma=gamma)
clf = svm.SVC(kernel='poly', coef0=r, gamma=gamma, degree=d)
clf = svm.SVC(kernel='sigmoid', coef0=r, gamma=gamma)
svm.LinearSVC()
svm.NuSVC(kernel='linear', nu=0.7)
svm.SVC(C=C, gamma=gamma)
svm.SVC(kernel='linear', C=0.1)
svm.SVC(kernel='linear', C=1)
svm.SVC(kernel='poly', coef0=r, gamma=gamma, degree=d)
svm.SVC(kernel='sigmoid', coef0=r, gamma=gamma)
svm.SVC(probability=True)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# sys as sys
#----------------------------------------------------------------------------------------------------
np.savetxt(sys.stdout, bval_rss, '%16.2f')
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# tabulate as tabulate
#----------------------------------------------------------------------------------------------------
print(tabulate([['Entropy']], tablefmt='fancy_grid'))
print(tabulate(output, tablefmt='fancy_grid'))
tabulate([['Entropy']], tablefmt='fancy_grid')
tabulate(output, tablefmt='fancy_grid')
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# timeit as timeit
#----------------------------------------------------------------------------------------------------
timeit.default_timer()
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# tqdm as tqdm
#----------------------------------------------------------------------------------------------------
tqdm(range(self.total_time_steps))
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# train_test_split as train_test_split
#----------------------------------------------------------------------------------------------------
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=i / 100.0)
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=None)
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=t_size, random_state=42)
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=t_size, random_state=i)
train_test_split(data, target, test_size=0.25, random_state=0)
train_test_split(data, target, test_size=0.25, random_state=123)
train_test_split(df, target, test_size=0.25, random_state=42)
train_test_split(features, target, test_size=0.2, random_state=42)
train_test_split(one_hot_df, labels, test_size=0.25, random_state=42)
train_test_split(X, y, random_state=0)
train_test_split(X, y, random_state=1)
train_test_split(X, y, random_state=10)
train_test_split(X, y, random_state=17)
train_test_split(X, y, random_state=42)
train_test_split(X, y, test_size=0.2, random_state=10)
train_test_split(X, y, test_size=0.2, random_state=4)
train_test_split(X, y, test_size=0.2, random_state=42)
train_test_split(X, y, test_size=0.25, random_state=22)
train_test_split(X, y, test_size=0.25, random_state=42)
train_test_split(X, y, test_size=0.3, random_state=SEED)
train_test_split(X, y, test_size=i / 100.0)
train_test_split(X, y, test_size=None)
train_test_split(X, y, test_size=t_size, random_state=42)
train_test_split(X, y, test_size=t_size, random_state=i)
train_test_split(X_3, y_3, test_size=0.33, random_state=123)
train_test_split(X_resampled, y_resampled, random_state=0)
train_test_split(X_train, y_train, random_state=0)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# tree as tree
#----------------------------------------------------------------------------------------------------
tree.plot_tree(classifier_2, feature_names=X.columns, class_names=np.unique(y).astype('str'), filled=True, rounded=True)
tree.plot_tree(clf, feature_names=df.columns, class_names=np.unique(y).astype('str'), filled=True)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# warnings as warnings
#----------------------------------------------------------------------------------------------------
warnings.filterwarnings('ignore')
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# XGBClassifier as XGBClassifier
#----------------------------------------------------------------------------------------------------
XGBClassifier()
#----------------------------------------------------------------------------------------------------
