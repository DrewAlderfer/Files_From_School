# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (learn-env)
#     language: python
#     name: python3
# ---

# + [markdown] nbgrader={"grade": false, "grade_id": "cell-6899ad425e03acd4", "locked": true, "schema_version": 3, "solution": false, "task": false}
# # Phase 3 Code Challenge
#
# This assessment is designed to test your understanding of Module 3 material. It covers:
#
# * Gradient Descent
# * Logistic Regression
# * Classification Metrics
# * Decision Trees
#
# _Read the instructions carefully_. You will be asked both to write code and to answer short answer questions.
#
# ## Code Tests
#
# We have provided some code tests for you to run to check that your work meets the item specifications. Passing these tests does not necessarily mean that you have gotten the item correct - there are additional hidden tests. However, if any of the tests do not pass, this tells you that your code is incorrect and needs changes to meet the specification. To determine what the issue is, read the comments in the code test cells, the error message you receive, and the item instructions.
#
# ## Short Answer Questions 
#
# For the short answer questions...
#
# * _Use your own words_. It is OK to refer to outside resources when crafting your response, but _do not copy text from another source_.
#
# * _Communicate clearly_. We are not grading your writing skills, but you can only receive full credit if your teacher is able to fully understand your response. 
#
# * _Be concise_. You should be able to answer most short answer questions in a sentence or two. Writing unnecessarily long answers increases the risk of you being unclear or saying something incorrect.

# + nbgrader={"grade": false, "grade_id": "cell-c2a2bae912a0e147", "locked": true, "schema_version": 3, "solution": false, "task": false}
# Run this cell without changes to import the necessary libraries

from numbers import Number

# %matplotlib inline

# + [markdown] nbgrader={"grade": false, "grade_id": "cell-962cbb6c01caf427", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ---
# ## Part 1: Gradient Descent [Suggested Time: 20 min]
# ---
# In this part, you will describe how gradient descent works to calculate a parameter estimate. Below is an image of a best fit line from a linear regression model using TV advertising spending to predict product sales.
#
# ![best fit line](https://raw.githubusercontent.com/learn-co-curriculum/dsc-cc-images/main/phase_3/best_fit_line.png)
#
# This best fit line can be described by the equation $y = mx + b$. Below is the RSS cost curve associated with the slope parameter $m$:
#
# ![cost curve](https://raw.githubusercontent.com/learn-co-curriculum/dsc-cc-images/main/phase_3/cost_curve.png)
#
# where RSS is the residual sum of squares: $RSS = \sum_{i=1}^n(y_i - (mx_i + b))^2$ 

# + [markdown] nbgrader={"grade": false, "grade_id": "cell-f5be777299f6d5be", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ### 1.1) Short Answer: Explain how the RSS curve above could be used to find an optimal value for the slope parameter $m$. 
#
# Your answer should provide a one sentence summary, not every step of the process.
# -

# ### Answer:
# The RSS curve describes the sum of square errors for a given slope of the regressions line. Using this knowledge and holding the y-intercept constant we
# can find the optimal slope by calculating the function minimum which will be the point at which the derivative of our RSS curve approaches zero.
#

# + [markdown] nbgrader={"grade": false, "grade_id": "cell-04569212f96246b5", "locked": true, "schema_version": 3, "solution": false, "task": false}
# Below is a visualization showing the iterations of a gradient descent algorithm applied the RSS curve. Each yellow marker represents an estimate, and the lines between markers represent the steps taken between estimates in each iteration. Numeric labels identify the iteration numbers.
#
# ![gradient descent](https://raw.githubusercontent.com/learn-co-curriculum/dsc-cc-images/main/phase_3/gd.png)

# + [markdown] nbgrader={"grade": false, "grade_id": "cell-8f8743b8bb5caf43", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ### 1.2) Short Answer: Explain why the distances between markers get smaller over successive iterations.
# -

# ### Answer:
# The steps get smaller because the learning rate adjust the step-size proportionally to calculated slope at each step.

# + [markdown] nbgrader={"grade": false, "grade_id": "cell-f38904edac3e34ba", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ### 1.3) Short Answer: What would be the effect of decreasing the learning rate for this application of gradient descent?
# -

# ### Answer:
# We would decrease the step-size and therefore increase the number of steps necessary to reach the local minimum.
#
#

# + [markdown] nbgrader={"grade": false, "grade_id": "cell-58cbc9e518eda9a5", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ---
# ## Part 2: Logistic Regression [Suggested Time: 15 min]
# ---
# In this part, you will answer general questions about logistic regression.

# + [markdown] nbgrader={"grade": false, "grade_id": "cell-a5eed21ce4450ee7", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ### 2.1) Short Answer: Provide one reason why logistic regression is better than linear regression for modeling a binary target/outcome.
# -

# ### Answer:
# Predictions made by a linear model will be continuous and therefore are bad representations for a binary target variable. They are also not bound to
# a proportional domain (0 to 1), unlike logistic regressions which can be represented predicted values as a probability.

# + [markdown] nbgrader={"grade": false, "grade_id": "cell-fc85e3d7f84c78d9", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ### 2.2) Short Answer: Compare logistic regression to another classification model of your choice (e.g. KNN, Decision Tree, etc.). What is one advantage and one disadvantage logistic regression has when compared with the other model?
# -

# ### Answer:
# * One advantage of a logistic regression is its speed and relative simplicity. It does very well on data were the target variable is well balanced between
# binary classifications.
# * One Advantage that KNN models have over logistic regression models is their accuracy in predicting data with more than two classes and data that is imbalanced
# or that more complex over lapping among the predictor variables.
#
#

# + [markdown] nbgrader={"grade": false, "grade_id": "cell-0d9d765be95e6cc0", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ---
# ## Part 3: Classification Metrics [Suggested Time: 20 min]
# ---
# In this part, you will make sense of classification metrics produced by various classifiers.

# + [markdown] nbgrader={"grade": false, "grade_id": "cell-d2ad4f31491e50b6", "locked": true, "schema_version": 3, "solution": false, "task": false}
# The confusion matrix below represents the predictions generated by a classisification model on a small testing dataset.
#
# ![cnf matrix](https://raw.githubusercontent.com/learn-co-curriculum/dsc-cc-images/main/phase_3/cnf_matrix.png)

# + [markdown] nbgrader={"grade": false, "grade_id": "cell-e4b5c09376d185ce", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ### 3.1) Create a numeric variable `precision` containing the precision of the classifier.

# +
# CodeGrade step3.1
# Replace None with appropriate code

precision = (30)/34
print(precision)

# + [markdown] nbgrader={"grade": false, "grade_id": "cell-09e2fa2bf91d1c95", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ### 3.2) Create a numeric variable `f1score` containing the F-1 score of the classifier.

# + nbgrader={"grade": false, "grade_id": "cell-6bce80c352c6ad99", "locked": true, "schema_version": 3, "solution": false, "task": false}
# CodeGrade step3.2
# Replace None with appropriate code
recall = 30/(30+12)

f1score = (precision * recall)/(precision + recall)
print(f1score)

# + [markdown] nbgrader={"grade": false, "grade_id": "cell-8c9611b7378f9cd8", "locked": true, "schema_version": 3, "solution": false, "task": false}
# The ROC curves below were calculated for three different models applied to one dataset.
#
# 1. Only Age was used as a feature in the model
# 2. Only Estimated Salary was used as a feature in the model
# 3. All features were used in the model
#
# ![roc](https://raw.githubusercontent.com/learn-co-curriculum/dsc-cc-images/main/phase_3/many_roc.png)

# + [markdown] nbgrader={"grade": false, "grade_id": "cell-6b2fccd135d7bd12", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ### 3.3) Short Answer: Identify the best ROC curve in the above graph and explain why it is the best. 
# -

# ### Answer:
# The model that uses all features is the best because its AUC (area under the curve) score is highest. This means that it is making the fewest
# misclassification out of the three models.

# + [markdown] nbgrader={"grade": false, "grade_id": "cell-9a2e4b682abfc6ec", "locked": true, "schema_version": 3, "solution": false, "task": false}
# Run the following cells to load a sample dataset, run a classification model on it, and perform some EDA.

# + nbgrader={"grade": false, "grade_id": "cell-9e7642482fd78eb5", "locked": true, "schema_version": 3, "solution": false, "task": false}
# Run this cell without changes

# Include relevant imports
import pickle, sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_curve, roc_auc_score

network_df = pickle.load(open('sample_network_data.pkl', 'rb'))

# partion features and target 
X = network_df.drop('Purchased', axis=1)
y = network_df['Purchased']

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2019)

# scale features
scale = StandardScaler()
scale.fit(X_train)
X_train = scale.transform(X_train)
X_test = scale.transform(X_test)

# build classifier
model = LogisticRegression(C=1e5, solver='lbfgs')
model.fit(X_train, y_train)
y_test_pred = model.predict(X_test)

# get the accuracy score
print(f'The classifier has an accuracy score of {round(accuracy_score(y_test, y_test_pred), 3)}.')

# + nbgrader={"grade": false, "grade_id": "cell-e21cfbd2172b791a", "locked": true, "schema_version": 3, "solution": false, "task": false}
# Run this cell without changes

y.value_counts(normalize=True)

# + [markdown] nbgrader={"grade": false, "grade_id": "cell-b3dee6c580108f26", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ### 3.4) Short Answer: Explain how the distribution of `y` shown above could explain the high accuracy score of the classification model.
# -

# ### Answer:
# This accuracy score of the model is barely higher than the probability of predicting the values with "model-less" predictor. AKA, if we simply
# guess category 0 for every value we would end up with a ~95% accuracy score.
#
#

# + [markdown] nbgrader={"grade": false, "grade_id": "cell-15288334b184b850", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ### 3.5) Short Answer: What is one method you could use to improve your model to address the issue discovered in Question 3.4?
# -

# ### Answer:
# We could use SMOTE to oversample our data. Often it is also possible to use tools like `stratify` built-in to scikit-learn. Stratify attempts to
# correct for imbalances in data by splitting the classification values proporationally within a Train/Test split or Cross-Validation process.
#
# However, given the extreme imbalance in the dataset we would either need to select a model that excels at this type of situation or oversample.

# + [markdown] nbgrader={"grade": false, "grade_id": "cell-6bdb41dda25eb6b0", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ---
# ## Part 4: Decision Trees [Suggested Time: 20 min]
# ---
# In this part, you will use decision trees to fit a classification model to a wine dataset. The data contain the results of a chemical analysis of wines grown in one region in Italy using three different cultivars (grape types). There are thirteen features from the measurements taken, and the wines are classified by cultivar in the `target` variable.

# + nbgrader={"grade": false, "grade_id": "cell-15de0bc4280a2aac", "locked": true, "schema_version": 3, "solution": false, "task": false}
# Run this cell without changes

# Relevant imports 
import pandas as pd 
import numpy as np 
from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier

# Load the data 
wine = load_wine()
X, y = load_wine(return_X_y=True)
X = pd.DataFrame(X, columns=wine.feature_names)
y = pd.Series(y)
y.name = 'target'

# + [markdown] nbgrader={"grade": false, "grade_id": "cell-561128e9ee6b0299", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ### 4.1) Use `train_test_split()` to evenly split `X` and `y` data between training sets (`X_train` and `y_train`) and test sets (`X_test` and `y_test`), with `random_state=1`.
#
# Do not alter `X` or `y` before performing the split.

# + nbgrader={"grade": false, "grade_id": "cell-0be055a675c0a674", "locked": true, "schema_version": 3, "solution": false, "task": false}
# CodeGrade step4.1
# Replace None with appropriate code

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=1)

# + [markdown] nbgrader={"grade": false, "grade_id": "cell-eac2fc7be9725bf0", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ### 4.2) Create an untuned decision tree classifier `wine_dt` and fit it using `X_train` and `y_train`, with `random_state=1`. 
#
# Use parameter defaults for your classifier. You must use the Scikit-learn DecisionTreeClassifier (docs [here](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html))

# + nbgrader={"grade": false, "grade_id": "cell-28bca1a3b0de0dd8", "locked": true, "schema_version": 3, "solution": false, "task": false}
# CodeGrade step4.2
# Replace None with appropriate code

wine_dt = DecisionTreeClassifier(criterion='entropy', random_state=1)
wine_dt.fit(X_train, y_train)

# + [markdown] nbgrader={"grade": false, "grade_id": "cell-55b417dc67abb7c6", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ### 4.3) Create an array `y_pred` generated by using `wine_dt` to make predictions for the test data.

# +
# CodeGrade step4.3
# Replace None with appropriate code

y_pred = wine_dt.predict(X_test)

# + [markdown] nbgrader={"grade": false, "grade_id": "cell-536526728a8066e2", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ### 4.4) Create a numeric variable `wine_dt_acc` containing the accuracy score for your predictions. 
#
# Hint: You can use the `sklearn.metrics` module.

# + nbgrader={"grade": false, "grade_id": "cell-67272706fb08c3bf", "locked": true, "schema_version": 3, "solution": false, "task": false}
# CodeGrade step4.4
# Replace None with appropriate code
from sklearn.metrics import classification_report
print(f"target baseline:\n{y_test.value_counts(normalize=True)}")
wine_dt_acc = accuracy_score(y_test, y_pred)
print(wine_dt_acc)
print(classification_report(y_test, y_pred))

# + [markdown] nbgrader={"grade": false, "grade_id": "cell-266fbd755dbbb4c2", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ### 4.5) Short Answer: Based on the accuracy score, does the model seem to be performing well or to have substantial performance issues? Explain your answer.
# -

# ### Answer:
# The model appears to perform well. The values are fairly evenly distributed within the target variable, so the score is well above what we would expect given 'random chance'.
# I would be interested in seeing how well it would perform with tuning. Given that this accuracy score is on the test
# data and that other metrics are all in line with our accruracy score I would say that %92 accuracy is a good starting place.
