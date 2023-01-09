# Regression Model Validation - Lab

## Introduction

In this lab, you'll be able to validate your Ames Housing data model using a train-test split.

## Objectives

You will be able to:

* Perform a train-test split
* Prepare training and testing data for modeling
* Compare training and testing errors to determine if model is over or underfitting

## Let's Use Our Ames Housing Data Again!

We included the code to load the data below.

```python
import pandas as pd
import numpy as np
ames = pd.read_csv('ames.csv', index_col=0)
ames
```
## Perform a Train-Test Split

Use `train_test_split` ([documentation here](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)) with the default split size. At the end you should have `X_train`, `X_test`, `y_train`, and `y_test` variables, where `y` represents `SalePrice` and `X` represents all other columns.

```python
from sklearn.model_selection import train_test_split
```
```python
X = ames.drop("SalePrice", axis=1)
y = ames["SalePrice"]
```
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
```
## Prepare Both Sets for Modeling

This code is completed for you and should work as long as the correct variables were created.

```python
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

continuous = ['LotArea', '1stFlrSF', 'GrLivArea']
categoricals = ['BldgType', 'KitchenQual', 'Street']

# Instantiate transformers
log_transformer = FunctionTransformer(np.log, validate=True)
ohe = OneHotEncoder(drop='first', sparse=False)

# Fit transformers
log_transformer.fit(X_train[continuous])
ohe.fit(X_train[categoricals])

# Transform training data
X_train = pd.concat([
    pd.DataFrame(log_transformer.transform(X_train[continuous]), index=X_train.index),
    pd.DataFrame(ohe.transform(X_train[categoricals]), index=X_train.index)
], axis=1)

# Transform test data
X_test = pd.concat([
    pd.DataFrame(log_transformer.transform(X_test[continuous]), index=X_test.index),
    pd.DataFrame(ohe.transform(X_test[categoricals]), index=X_test.index)
], axis=1)
```
## Fit a Linear Regression on the Training Data

```python
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
```
```python
linreg.fit(X_train, y_train)
```
## Evaluate and Validate Model

### Generate Predictions on Training and Test Sets

```python
y_hat_train = linreg.predict(X_train)
y_hat_test = linreg.predict(X_test)
```
### Calculate the Mean Squared Error (MSE)

You can use `mean_squared_error` from scikit-learn ([documentation here](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html)).

```python
from sklearn.metrics import mean_squared_error
```
```python
train_mse = mean_squared_error(y_train, y_hat_train)
test_mse = mean_squared_error(y_test, y_hat_test)
print('Train Mean Squared Error:', train_mse)
print('Test Mean Squared Error: ', test_mse)
```
If your test error is substantially worse than the train error, this is a sign that the model doesn't generalize well to future cases.

One simple way to demonstrate overfitting and underfitting is to alter the size of our train-test split. By default, scikit-learn allocates 25% of the data to the test set and 75% to the training set. Fitting a model on only 10% of the data is apt to lead to underfitting, while training a model on 99% of the data is apt to lead to overfitting.

## Level Up: Evaluate the Effect of Train-Test Split Size

Iterate over a range of train-test split sizes from .5 to .9. For each of these, generate a new train/test split sample. Preprocess both sets of data. Fit a model to the training sample and calculate both the training error and the test error (MSE) for each of these splits. Plot these two curves (train error vs. training size and test error vs. training size) on a graph.

```python
import matplotlib.pyplot as plt

train_mses = []
test_mses = []

t_sizes = np.linspace(0.5, 0.9, 10)
for t_size in t_sizes:

    # Create new split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=t_size, random_state=42)

    # Fit transformers on new train and test
    log_transformer.fit(X_train[continuous])
    ohe.fit(X_train[categoricals])

    # Transform training data
    X_train = pd.concat([
        pd.DataFrame(log_transformer.transform(X_train[continuous]), index=X_train.index),
        pd.DataFrame(ohe.transform(X_train[categoricals]), index=X_train.index)
    ], axis=1)

    # Transform test data
    X_test = pd.concat([
        pd.DataFrame(log_transformer.transform(X_test[continuous]), index=X_test.index),
        pd.DataFrame(ohe.transform(X_test[categoricals]), index=X_test.index)
    ], axis=1)

    # Fit model
    linreg.fit(X_train, y_train)

    # Append metrics to their respective lists
    y_hat_train = linreg.predict(X_train)
    y_hat_test = linreg.predict(X_test)
    train_mses.append(mean_squared_error(y_train, y_hat_train))
    test_mses.append(mean_squared_error(y_test, y_hat_test))

fig, ax = plt.subplots()
ax.scatter(t_sizes, train_mses, label='Training Error')
ax.scatter(t_sizes, test_mses, label='Testing Error')
ax.legend();
```
### Extension

Repeat the previous example, but for each train-test split size, generate 10 iterations of models/errors and save the average train/test error. This will help account for any particularly good/bad models that might have resulted from poor/good splits in the data.

```python

train_mses = []
test_mses = []

t_sizes = np.linspace(0.5, 0.9, 10)
for t_size in t_sizes:

    inner_train_mses = []
    inner_test_mses = []
    for i in range(10):
        # Create new split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=t_size, random_state=i)

        # Skipping fitting the transformers; data quality issues cause too many OHE problems when
        # fitting this number of different models, but if you don't use drop='first' the
        # multicollinearity issues get pretty bad

        # Transform training data
        X_train = pd.concat([
            pd.DataFrame(log_transformer.transform(X_train[continuous]), index=X_train.index),
            pd.DataFrame(ohe.transform(X_train[categoricals]), index=X_train.index)
        ], axis=1)

        # Transform test data
        X_test = pd.concat([
            pd.DataFrame(log_transformer.transform(X_test[continuous]), index=X_test.index),
            pd.DataFrame(ohe.transform(X_test[categoricals]), index=X_test.index)
        ], axis=1)

        # Fit model
        linreg.fit(X_train, y_train)

        # Append metrics to their respective lists
        y_hat_train = linreg.predict(X_train)
        y_hat_test = linreg.predict(X_test)
        inner_train_mses.append(mean_squared_error(y_train, y_hat_train))
        inner_test_mses.append(mean_squared_error(y_test, y_hat_test))

    train_mses.append(np.mean(inner_train_mses))
    test_mses.append(np.mean(inner_test_mses))

fig, ax = plt.subplots()
ax.scatter(t_sizes, train_mses, label='Average Training Error')
ax.scatter(t_sizes, test_mses, label='Average Testing Error')
ax.legend();
```
What's happening here? Evaluate your result!

##  Summary

Congratulations! You now practiced your knowledge of MSE and used your train-test split skills to validate your model.


-----File-Boundary-----
# Introduction to Cross-Validation - Lab

## Introduction

In this lab, you'll be able to practice your cross-validation skills!


## Objectives

You will be able to:

- Perform cross validation on a model
- Compare and contrast model validation strategies

## Let's Get Started

We included the code to pre-process the Ames Housing dataset below. This is done for the sake of expediency, although it may result in data leakage and therefore overly optimistic model metrics.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

ames = pd.read_csv('ames.csv')

continuous = ['LotArea', '1stFlrSF', 'GrLivArea', 'SalePrice']
categoricals = ['BldgType', 'KitchenQual', 'SaleType', 'MSZoning', 'Street', 'Neighborhood']

ames_cont = ames[continuous]

# log features
log_names = [f'{column}_log' for column in ames_cont.columns]

ames_log = np.log(ames_cont)
ames_log.columns = log_names

# normalize (subract mean and divide by std)

def normalize(feature):
    return (feature - feature.mean()) / feature.std()

ames_log_norm = ames_log.apply(normalize)

# one hot encode categoricals
ames_ohe = pd.get_dummies(ames[categoricals], prefix=categoricals, drop_first=True)

preprocessed = pd.concat([ames_log_norm, ames_ohe], axis=1)

X = preprocessed.drop('SalePrice_log', axis=1)
y = preprocessed['SalePrice_log']
```
## Train-Test Split

Perform a train-test split with a test set of 20% and a random state of 4.

```python
from sklearn.model_selection import train_test_split
```
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
```
### Fit a Model

Fit a linear regression model on the training set

```python
from sklearn.linear_model import LinearRegression
```
```python
linreg = LinearRegression()
linreg.fit(X_train, y_train)
```
### Calculate MSE

Calculate the mean squared error on the test set

```python
from sklearn.metrics import mean_squared_error
```
```python

y_hat_test = linreg.predict(X_test)
test_mse = mean_squared_error(y_test, y_hat_test)
test_mse
```
## Cross-Validation using Scikit-Learn

Now let's compare that single test MSE to a cross-validated test MSE.

```python
from sklearn.model_selection import cross_val_score
```
```python
cv_5_results = -cross_val_score(linreg, X, y, cv=5, scoring="neg_mean_squared_error")
cv_5_results
```
```python
cv_5_results.mean()
```
Compare and contrast the results. What is the difference between the train-test split and cross-validation results? Do you "trust" one more than the other?

```python
"""
Rounded to 1 decimal place, both the train-test split result and the cross-validation
result would be 0.2

However with more decimal places, the differences become apparent. The train-test split
result is about 0.152 whereas the average cross-validation result is about 0.177. Because
this is an error-based metric, a higher value is worse, so this means that the train-test
split result is "better" (more optimistic)

Another way to look at the results would be to compare the train-test split result to each
of the 5 split results. Only 1 out of 5 splits has a better MSE than the train-test split,
meaning that the average is not being skewed by just 1 or 2 values that are significantly
worse than the train-test split result

Taking the average as well as the spread into account, I would be more inclined to "trust"
the cross-validated (less optimistic) score and assume that the performance on unseen data
would be closer to 0.177 than 0.152
"""
```
## Level Up: Let's Build It from Scratch!

### Create a Cross-Validation Function

Write a function `kfolds(data, k)` that splits a dataset into `k` evenly sized pieces. If the full dataset is not divisible by `k`, make the first few folds one larger then later ones.

For example, if you had this dataset:

```python
example_data = pd.DataFrame({
    "color": ["red", "orange", "yellow", "green", "blue", "indigo", "violet"]
})
example_data
```
`kfolds(example_data, 3)` should return:

* a dataframe with `red`, `orange`, `yellow`
* a dataframe with `green`, `blue`
* a dataframe with `indigo`, `violet`

Because the example dataframe has 7 records, which is not evenly divisible by 3, so the "leftover" 1 record extends the length of the first dataframe.

```python
def kfolds(data, k):
    folds = []

    # Calculate the fold size plus the remainder
    num_observations = len(data)
    small_fold_size = num_observations // k
    large_fold_size = small_fold_size + 1
    leftovers = num_observations % k

    start_index = 0
    for fold_n in range(k):
        # Select fold size based on whether or not all of the leftovers have been used up
        if fold_n < leftovers:
            fold_size = large_fold_size
        else:
            fold_size = small_fold_size

        # Get fold from dataframe and add it to the list
        fold = data.iloc[start_index:start_index + fold_size]
        folds.append(fold)

        # Move the start index to the start of the next fold
        start_index += fold_size

    return folds

results = kfolds(example_data, 3)
for result in results:
    print(result, "\n")
```
### Apply Your Function to the Ames Housing Data

Get folds for both `X` and `y`.

```python

X_folds = kfolds(X, 5)
y_folds = kfolds(y, 5)
```
### Perform a Linear Regression for Each Fold and Calculate the Test Error

Remember that for each fold you will need to concatenate all but one of the folds to represent the training data, while the one remaining fold represents the test data.

```python
test_errs = []
k = 5

for n in range(k):
    # Split into train and test for the fold
    X_train = pd.concat([fold for i, fold in enumerate(X_folds) if i!=n])
    X_test = X_folds[n]
    y_train = pd.concat([fold for i, fold in enumerate(y_folds) if i!=n])
    y_test = y_folds[n]

    # Fit a linear regression model
    linreg.fit(X_train, y_train)

    # Evaluate test errors
    y_hat_test = linreg.predict(X_test)
    test_residuals = y_hat_test - y_test
    test_errs.append(np.mean(test_residuals.astype(float)**2))

print(test_errs)
```
If your code was written correctly, these should be the same errors as scikit-learn produced with `cross_val_score` (within rounding error). Test this out below:

```python

for k in range(5):
    print(f"Split {k+1}")
    print(f"My result:      {round(test_errs[k], 4)}")
    print(f"sklearn result: {round(cv_5_results[k], 4)}\n")
```
This was a bit of work! Hopefully you have a clearer understanding of the underlying logic for cross-validation if you attempted this exercise.

##  Summary

Congratulations! You are now familiar with cross-validation and know how to use `cross_val_score()`. Remember that the results obtained from cross-validation are more robust than train-test split.


-----File-Boundary-----
# Bias-Variance Tradeoff - Lab

## Introduction

In this lab, you'll practice the concepts you learned in the last lesson, bias-variance tradeoff.

## Objectives

In this lab you will:

- Demonstrate the tradeoff between bias and variance by way of fitting a machine learning model

## Let's get started!

In this lab, you'll try to predict some movie revenues based on certain factors, such as ratings and movie year. Start by running the following cell which imports all the necessary functions and the dataset:

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import *
import matplotlib.pyplot as plt
%matplotlib inline

df = pd.read_excel('movie_data_detailed_with_ols.xlsx')
df.head()
```
Subset the `df` DataFrame to only keep the `'domgross'`, `'budget'`, `'imdbRating'`, `'Metascore'`, and `'imdbVotes'` columns.

```python
# Subset the DataFrame
df = df[['domgross', 'budget', 'imdbRating', 'Metascore', 'imdbVotes']]
df.head()
```
## Split the data


- First, assign the predictors to `X` and the outcome variable, `'domgross'` to `y`
- Split the data into training and test sets. Set the seed to 42 and the `test_size` to 0.25

```python
# domgross is the outcome variable
X = df[['budget', 'imdbRating', 'Metascore', 'imdbVotes']]
y = df['domgross']

X_train , X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
```
Use the `MinMaxScaler` to scale the training set. Remember you can fit and transform in a single method using `.fit_transform()`.

```python
# Transform with MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
```
Transform the test data (`X_test`) using the same `scaler`:

```python
# Scale the test set
X_test_scaled = scaler.transform(X_test)
```
## Fit a regression model to the training data

```python
# Your code
linreg = LinearRegression()
linreg.fit(X_train_scaled, y_train)
```
Use the model to make predictions on both the training and test sets:

```python
# Training set predictions
lm_train_predictions = linreg.predict(X_train_scaled)

# Test set predictions
lm_test_predictions = linreg.predict(X_test_scaled)
```
Plot predictions for the training set against the actual data:

```python
# Run this cell - vertical distance between the points and the line denote the errors
plt.figure(figsize=(8, 5))
plt.scatter(y_train, lm_train_predictions, label='Model')
plt.plot(y_train, y_train, label='Actual data')
plt.title('Model vs data for training set')
plt.legend();
```
Plot predictions for the test set against the actual data:

```python
# Run this cell - vertical distance between the points and the line denote the errors
plt.figure(figsize=(8, 5))
plt.scatter(y_test, lm_test_predictions, label='Model')
plt.plot(y_test, y_test, label='Actual data')
plt.title('Model vs data for test set')
plt.legend();
```
## Bias

Create a function `bias()` to calculate the bias of a model's predictions given the actual data: $Bias(\hat{f}(x)) = E[\hat{f}(x)-f(x)]$
(The expected value can simply be taken as the mean or average value.)

```python
import numpy as np
def bias(y, y_hat):
    return np.mean(y_hat - y)
```
## Variance
Create a function `variance()` to calculate the variance of a model's predictions: $Var(\hat{f}(x)) = E[\hat{f}(x)^2] - \big(E[\hat{f}(x)]\big)^2$

```python
def variance(y_hat):
    return np.mean([yi**2 for yi in y_hat]) - np.mean(y_hat)**2
```
## Calculate bias and variance

```python
# Bias and variance for training set
b = bias(y_train, lm_train_predictions)
v = variance(lm_train_predictions)
print('Train bias: {} \nTrain variance: {}'.format(b, v))
```
```python
# Bias and variance for test set
b = bias(y_test, lm_test_predictions)
v = variance(lm_test_predictions)
print('Test bias: {} \nTest variance: {}'.format(b, v))
```
## Overfit a new model

Use `PolynomialFeatures` with degree 3 and transform `X_train_scaled` and `X_test_scaled`.

**Important note:** By including this, you don't only take polynomials of single variables, but you also combine variables, eg:

$ \text{Budget} * \text{MetaScore} ^ 2 $

What you're essentially doing is taking interactions and creating polynomials at the same time! Have a look at how many columns we get using `np.shape()`!

```python
poly = PolynomialFeatures(3)

X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.fit_transform(X_test_scaled)
```
```python
# Check the shape
np.shape(X_train_poly)
```
Fit a regression model to the training data:

```python
polyreg = LinearRegression()
polyreg.fit(X_train_poly, y_train)
```
Use the model to make predictions on both the training and test sets:

```python
# Training set predictions
poly_train_predictions = polyreg.predict(X_train_poly)

# Test set predictions
poly_test_predictions = polyreg.predict(X_test_poly)
```
Plot predictions for the training set against the actual data:

```python
# Run this cell - vertical distance between the points and the line denote the errors
plt.figure(figsize=(8, 5))
plt.scatter(y_train, poly_train_predictions, label='Model')
plt.plot(y_train, y_train, label='Actual data')
plt.title('Model vs data for training set')
plt.legend();
```
Plot predictions for the test set against the actual data:

```python
# Run this cell - vertical distance between the points and the line denote the errors
plt.figure(figsize=(8, 5))
plt.scatter(y_test, poly_test_predictions, label='Model')
plt.plot(y_test, y_test, label='Actual data')
plt.title('Model vs data for test set')
plt.legend();
```
Calculate the bias and variance for the training set:

```python
# Bias and variance for training set
b = bias(y_train, poly_train_predictions)
v = variance(poly_train_predictions)
print('Train bias: {} \nTrain variance: {}'.format(b, v))
```
Calculate the bias and variance for the test set:

```python
# Bias and variance for test set
b = bias(y_test, poly_test_predictions)
v = variance(poly_test_predictions)
print('Test bias: {} \nTest variance: {}'.format(b, v))
```
## Interpret the overfit model

```python
# The training predictions from the second model perfectly match the actual data points - which indicates overfitting.
# The bias and variance for the test set both increased drastically for this overfit model.
```
## Level Up (Optional)

In this lab we went from 4 predictors to 35 by adding polynomials and interactions, using `PolynomialFeatures`. That being said, where 35 leads to overfitting, there are probably ways to improve by adding just a few polynomials. Feel free to experiment and see how bias and variance improve!

## Summary

This lab gave you insight into how bias and variance change for a training and a test set by using both simple and complex models.


-----File-Boundary-----
# Ridge and Lasso Regression - Lab

## Introduction

In this lab, you'll practice your knowledge of ridge and lasso regression!

## Objectives

In this lab you will:

- Use lasso and ridge regression with scikit-learn
- Compare and contrast lasso, ridge and non-regularized regression

## Housing Prices Data

We'll use this version of the Ames Housing dataset:

```python
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('housing_prices.csv', index_col=0)
df.info()
```
More information about the features is available in the `data_description.txt` file in this repository.

## Data Preparation

The code below:

* Separates the data into `X` (predictor) and `y` (target) variables
* Splits the data into 75-25 training-test sets, with a `random_state` of 10
* Separates each of the `X` values into continuous vs. categorical features
* Fills in missing values (using different strategies for continuous vs. categorical features)
* Scales continuous features to a range of 0 to 1
* Dummy encodes categorical features
* Combines the preprocessed continuous and categorical features back together

```python
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Create X and y
y = df['SalePrice']
X = df.drop(columns=['SalePrice'])

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)

# Separate X data into continuous vs. categorical
X_train_cont = X_train.select_dtypes(include='number')
X_test_cont = X_test.select_dtypes(include='number')
X_train_cat = X_train.select_dtypes(exclude='number')
X_test_cat = X_test.select_dtypes(exclude='number')

# Impute missing values using SimpleImputer, median for continuous and
# filling in 'missing' for categorical
impute_cont = SimpleImputer(strategy='median')
X_train_cont = impute_cont.fit_transform(X_train_cont)
X_test_cont = impute_cont.transform(X_test_cont)
impute_cat = SimpleImputer(strategy='constant', fill_value='missing')
X_train_cat = impute_cat.fit_transform(X_train_cat)
X_test_cat = impute_cat.transform(X_test_cat)

# Scale continuous values using MinMaxScaler
scaler = MinMaxScaler()
X_train_cont = scaler.fit_transform(X_train_cont)
X_test_cont = scaler.transform(X_test_cont)

# Dummy encode categorical values using OneHotEncoder
ohe = OneHotEncoder(handle_unknown='ignore')
X_train_cat = ohe.fit_transform(X_train_cat)
X_test_cat = ohe.transform(X_test_cat)

# Combine everything back together
X_train_preprocessed = np.concatenate([X_train_cont, X_train_cat.todense()], axis=1)
X_test_preprocessed = np.concatenate([X_test_cont, X_test_cat.todense()], axis=1)
```
## Linear Regression Model

Let's use this data to build a first naive linear regression model. Fit the model on the training data (`X_train_preprocessed`), then compute the R-Squared and the MSE for both the training and test sets.

```python
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

# Fit the model
linreg = LinearRegression()
linreg.fit(X_train_preprocessed, y_train)

# Print R2 and MSE for training and test sets
print('Training r^2:', linreg.score(X_train_preprocessed, y_train))
print('Test r^2:    ', linreg.score(X_test_preprocessed, y_test))
print('Training MSE:', mean_squared_error(y_train, linreg.predict(X_train_preprocessed)))
print('Test MSE:    ', mean_squared_error(y_test, linreg.predict(X_test_preprocessed)))
```
Notice the severe overfitting above; our training R-Squared is very high, but the test R-Squared is negative! Similarly, the scale of the test MSE is orders of magnitude higher than that of the training MSE.

## Ridge and Lasso Regression

Use all the data (scaled features and dummy categorical variables, `X_train_preprocessed`) to build some models with regularization - two each for lasso and ridge regression. Each time, look at R-Squared and MSE.

Remember that you can use the scikit-learn documentation if you don't remember how to import or use these classes:

* [`Lasso` documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)
* [`Ridge` documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)

### Lasso

#### With default hyperparameters (`alpha` = 1)

```python
from sklearn.linear_model import Lasso

lasso = Lasso() # Lasso is also known as the L1 norm
lasso.fit(X_train_preprocessed, y_train)

print('Training r^2:', lasso.score(X_train_preprocessed, y_train))
print('Test r^2:    ', lasso.score(X_test_preprocessed, y_test))
print('Training MSE:', mean_squared_error(y_train, lasso.predict(X_train_preprocessed)))
print('Test MSE:    ', mean_squared_error(y_test, lasso.predict(X_test_preprocessed)))
```
#### With a higher regularization hyperparameter (`alpha` = 10)

```python

lasso_10 = Lasso(alpha=10)
lasso_10.fit(X_train_preprocessed, y_train)

print('Training r^2:', lasso_10.score(X_train_preprocessed, y_train))
print('Test r^2:    ', lasso_10.score(X_test_preprocessed, y_test))
print('Training MSE:', mean_squared_error(y_train, lasso_10.predict(X_train_preprocessed)))
print('Test MSE:    ', mean_squared_error(y_test, lasso_10.predict(X_test_preprocessed)))
```
## Ridge

#### With default hyperparameters (`alpha` = 1)

```python
from sklearn.linear_model import Ridge

ridge = Ridge() # Ridge is also known as the L2 norm
ridge.fit(X_train_preprocessed, y_train)

print('Training r^2:', ridge.score(X_train_preprocessed, y_train))
print('Test r^2:    ', ridge.score(X_test_preprocessed, y_test))
print('Training MSE:', mean_squared_error(y_train, ridge.predict(X_train_preprocessed)))
print('Test MSE:    ', mean_squared_error(y_test, ridge.predict(X_test_preprocessed)))
```
#### With higher regularization hyperparameter (`alpha` = 10)

```python

ridge_10 = Ridge(alpha=10)
ridge_10.fit(X_train_preprocessed, y_train)

print('Training r^2:', ridge_10.score(X_train_preprocessed, y_train))
print('Test r^2:    ', ridge_10.score(X_test_preprocessed, y_test))
print('Training MSE:', mean_squared_error(y_train, ridge_10.predict(X_train_preprocessed)))
print('Test MSE:    ', mean_squared_error(y_test, ridge_10.predict(X_test_preprocessed)))
```
## Comparing the Metrics

Which model seems best, based on the metrics?

```python
print('Test r^2')
print('Linear Regression:', linreg.score(X_test_preprocessed, y_test))
print('Lasso, alpha=1:   ', lasso.score(X_test_preprocessed, y_test))
print('Lasso, alpha=10:  ', lasso_10.score(X_test_preprocessed, y_test))
print('Ridge, alpha=1:   ', ridge.score(X_test_preprocessed, y_test))
print('Ridge, alpha=10:  ', ridge_10.score(X_test_preprocessed, y_test))
print()
print('Test MSE')
print('Linear Regression:', mean_squared_error(y_test, linreg.predict(X_test_preprocessed)))
print('Lasso, alpha=1:   ', mean_squared_error(y_test, lasso.predict(X_test_preprocessed)))
print('Lasso, alpha=10:  ', mean_squared_error(y_test, lasso_10.predict(X_test_preprocessed)))
print('Ridge, alpha=1:   ', mean_squared_error(y_test, ridge.predict(X_test_preprocessed)))
print('Ridge, alpha=10:  ', mean_squared_error(y_test, ridge_10.predict(X_test_preprocessed)))
```
<details>
    <summary style="cursor: pointer"><b>Answer (click to reveal)</b></summary>

In terms of both R-Squared and MSE, the `Lasso` model with `alpha`=10 has the best metric results.

(Remember that better R-Squared is higher, whereas better MSE is lower.)

</details>

## Comparing the Parameters

Compare the number of parameter estimates that are (very close to) 0 for the `Ridge` and `Lasso` models with `alpha`=10.

Use 10**(-10) as an estimate that is very close to 0.

```python
print('Zeroed-out ridge params:', sum(abs(ridge_10.coef_) < 10**(-10)),
     'out of', len(ridge_10.coef_))
```
```python
print('Zeroed-out lasso params:', sum(abs(lasso_10.coef_) < 10**(-10)),
     'out of', len(lasso_10.coef_))
```
<details>
    <summary style="cursor: pointer"><b>Answer (click to reveal)</b></summary>

The ridge model did not penalize any coefficients to 0, while the lasso model removed about 1/4 of the coefficients. The lasso model essentially performed variable selection for us, and got the best metrics as a result!

</details>

## Finding an Optimal Alpha

Earlier we tested two values of `alpha` to see how it affected our MSE and the value of our coefficients. We could continue to guess values of `alpha` for our ridge or lasso regression one at a time to see which values minimize our loss, or we can test a range of values and pick the alpha which minimizes our MSE. Here is an example of how we would do this:

```python
import matplotlib.pyplot as plt
%matplotlib inline

train_mse = []
test_mse = []
alphas = np.linspace(0, 200, num=50)

for alpha in alphas:
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train_preprocessed, y_train)

    train_preds = lasso.predict(X_train_preprocessed)
    train_mse.append(mean_squared_error(y_train, train_preds))

    test_preds = lasso.predict(X_test_preprocessed)
    test_mse.append(mean_squared_error(y_test, test_preds))

fig, ax = plt.subplots()
ax.plot(alphas, train_mse, label='Train')
ax.plot(alphas, test_mse, label='Test')
ax.set_xlabel('alpha')
ax.set_ylabel('MSE')

# np.argmin() returns the index of the minimum value in a list
optimal_alpha = alphas[np.argmin(test_mse)]

# Add a vertical line where the test MSE is minimized
ax.axvline(optimal_alpha, color='black', linestyle='--')
ax.legend();

print(f'Optimal Alpha Value: {int(optimal_alpha)}')
```
Take a look at this graph of our training and test MSE against `alpha`. Try to explain to yourself why the shapes of the training and test curves are this way. Make sure to think about what `alpha` represents and how it relates to overfitting vs underfitting.

---

<details>
    <summary style="cursor: pointer"><b>Answer (click to reveal)</b></summary>

For `alpha` values below 28, the model is overfitting. As `alpha` increases up to 28, the MSE for the training data increases and MSE for the test data decreases, indicating that we are reducing overfitting.

For `alpha` values above 28, the model is starting to underfit. You can tell because _both_ the train and the test MSE values are increasing.

</details>

## Summary

Well done! You now know how to build lasso and ridge regression models, use them for feature selection and find an optimal value for `alpha`.


-----File-Boundary-----
# Extensions to Linear Models - Lab

## Introduction

In this lab, you'll practice many concepts you have learned so far, from adding interactions and polynomials to your model to regularization!

## Summary

You will be able to:

- Build a linear regression model with interactions and polynomial features
- Use feature selection to obtain the optimal subset of features in a dataset

## Let's Get Started!

Below we import all the necessary packages for this lab.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from itertools import combinations

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
```
Load the data.

```python

# Load data from CSV
df = pd.read_csv("ames.csv")
# Subset columns
df = df[['LotArea', 'OverallQual', 'OverallCond', 'TotalBsmtSF',
         '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'TotRmsAbvGrd',
         'GarageArea', 'Fireplaces', 'SalePrice']]

# Split the data into X and y
y = df['SalePrice']
X = df.drop(columns='SalePrice')

# Split into train, test, and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, random_state=0)
```
## Build a Baseline Housing Data Model

Above, we imported the Ames housing data and grabbed a subset of the data to use in this analysis.

Next steps:

- Scale all the predictors using `StandardScaler`, then convert these scaled features back into DataFrame objects
- Build a baseline `LinearRegression` model using *scaled variables* as predictors and use the $R^2$ score to evaluate the model

```python

# Scale X_train and X_test using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Ensure X_train and X_test are scaled DataFrames
# (hint: you can set the columns using X.columns)
X_train = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test = pd.DataFrame(X_test_scaled, columns=X.columns)

X_train
```
```python

# Create a LinearRegression model and fit it on scaled training data
regression = LinearRegression()
regression.fit(X_train, y_train)

# Calculate a baseline r-squared score on training data
baseline = regression.score(X_train, y_train)
baseline
```
## Add Interactions

Instead of adding all possible interaction terms, let's try a custom technique. We are only going to add the interaction terms that increase the $R^2$ score as much as possible. Specifically we are going to look for the 7 interaction terms that each cause the most increase in the coefficient of determination.

### Find the Best Interactions

Look at all the possible combinations of variables for interactions by adding interactions one by one to the baseline model. Create a data structure that stores the pair of columns used as well as the $R^2$ score for each combination.

***Hint:*** We have imported the `combinations` function from `itertools` for you ([documentation here](https://docs.python.org/3/library/itertools.html#itertools.combinations)). Try applying this to the columns of `X_train` to find all of the possible pairs.

Print the 7 interactions that result in the highest $R^2$ scores.

```python

# Set up data structure
# (Here we are using a list of tuples, but you could use a dictionary,
# a list of lists, some other structure. Whatever makes sense to you.)
interactions = []

# Find combinations of columns and loop over them
column_pairs = list(combinations(X_train.columns, 2))
for (col1, col2) in column_pairs:
    # Make copies of X_train and X_test
    features_train = X_train.copy()
    features_test = X_test.copy()

    # Add interaction term to data
    features_train["interaction"] = features_train[col1] * features_train[col2]
    features_test["interaction"] = features_test[col1] * features_test[col2]

    # Find r-squared score (fit on training data, evaluate on test data)
    score = LinearRegression().fit(features_train, y_train).score(features_test, y_test)

    # Append to data structure
    interactions.append(((col1, col2), score))

# Sort and subset the data structure to find the top 7
top_7_interactions = sorted(interactions, key=lambda record: record[1], reverse=True)[:7]
print("Top 7 interactions:")
print(top_7_interactions)
```
### Add the Best Interactions

Write code to include the 7 most important interactions in `X_train` and `X_test` by adding 7 columns. Use the naming convention `"col1_col2"`, where `col1` and `col2` are the two columns in the interaction.

```python

# Loop over top 7 interactions
for record in top_7_interactions:
    # Extract column names from data structure
    col1, col2 = record[0]

    # Construct new column name
    new_col_name = col1 + "_" + col2

    # Add new column to X_train and X_test
    X_train[new_col_name] = X_train[col1] * X_train[col2]
    X_test[new_col_name] = X_test[col1] * X_test[col2]

X_train
```
## Add Polynomials

Now let's repeat that process for adding polynomial terms.

### Find the Best Polynomials

Try polynomials of degrees 2, 3, and 4 for each variable, in a similar way you did for interactions (by looking at your baseline model and seeing how $R^2$ increases). Do understand that when going for a polynomial of degree 4 with `PolynomialFeatures`, the particular column is raised to the power of 2 and 3 as well in other terms.

We only want to include "pure" polynomials, so make sure no interactions are included.

Once again you should make a data structure that contains the values you have tested. We recommend a list of tuples of the form:

`(col_name, degree, R2)`, so eg. `('OverallQual', 2, 0.781)`

```python

# Set up data structure
polynomials = []

# Loop over all columns
for col in X_train.columns:
    # Loop over degrees 2, 3, 4
    for degree in (2, 3, 4):

        # Make a copy of X_train and X_test
        features_train = X_train.copy().reset_index()
        features_test = X_test.copy().reset_index()

        # Instantiate PolynomialFeatures with relevant degree
        poly = PolynomialFeatures(degree, include_bias=False)

        # Fit polynomial to column and transform column
        # Hint: use the notation df[[column_name]] to get the right shape
        # Hint: convert the result to a DataFrame
        col_transformed_train = pd.DataFrame(poly.fit_transform(features_train[[col]]))
        col_transformed_test = pd.DataFrame(poly.transform(features_test[[col]]))

        # Add polynomial to data
        # Hint: use pd.concat since you're combining two DataFrames
        # Hint: drop the column before combining so it doesn't appear twice
        features_train = pd.concat([features_train.drop(col, axis=1), col_transformed_train], axis=1)
        features_test = pd.concat([features_test.drop(col, axis=1), col_transformed_test], axis=1)

        # Find r-squared score
        score = LinearRegression().fit(features_train, y_train).score(features_test, y_test)

        # Append to data structure
        polynomials.append((col, degree, score))

# Sort and subset the data structure to find the top 7
top_7_polynomials = sorted(polynomials, key=lambda record: record[-1], reverse=True)[:7]
print("Top 7 polynomials:")
print(top_7_polynomials)
```
### Add the Best Polynomials

If there are duplicate column values in the results above, don't add multiple of them to the model, to avoid creating duplicate columns. (For example, if column `A` appeared in your list as both a 2nd and 3rd degree polynomial, adding both would result in `A` squared being added to the features twice.) Just add in the polynomial that results in the highest R-Squared.

```python
# Convert to DataFrame for easier manipulation
top_polynomials = pd.DataFrame(top_7_polynomials, columns=["Column", "Degree", "R^2"])
top_polynomials
```
```python
# Drop duplicate columns
top_polynomials.drop_duplicates(subset="Column", inplace=True)
top_polynomials
```
```python

# Loop over remaining results
for (col, degree, _) in top_polynomials.values:
    # Create polynomial terms
    poly = PolynomialFeatures(degree, include_bias=False)
    col_transformed_train = pd.DataFrame(poly.fit_transform(X_train[[col]]),
                                        columns=poly.get_feature_names([col]))
    col_transformed_test = pd.DataFrame(poly.transform(X_test[[col]]),
                                    columns=poly.get_feature_names([col]))
    # Concat new polynomials to X_train and X_test
    X_train = pd.concat([X_train.drop(col, axis=1), col_transformed_train], axis=1)
    X_test = pd.concat([X_test.drop(col, axis=1), col_transformed_test], axis=1)

```
Check out your final data set and make sure that your interaction terms as well as your polynomial terms are included.

```python
X_train.head()
```
## Full Model R-Squared

Check out the $R^2$ of the full model with your interaction and polynomial terms added. Print this value for both the train and test data.

```python
lr = LinearRegression()
lr.fit(X_train, y_train)

print("Train R^2:", lr.score(X_train, y_train))
print("Test R^2: ", lr.score(X_test, y_test))
```
It looks like we may be overfitting some now. Let's try some feature selection techniques.

## Feature Selection

First, test out `RFE` ([documentation here](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html)) with several different `n_features_to_select` values. For each value, print out the train and test $R^2$ score and how many features remain.

```python
for n in [5, 10, 15, 20, 25]:
    rfe = RFE(LinearRegression(), n_features_to_select=n)
    X_rfe_train = rfe.fit_transform(X_train, y_train)
    X_rfe_test = rfe.transform(X_test)

    lr = LinearRegression()
    lr.fit(X_rfe_train, y_train)

    print("Train R^2:", lr.score(X_rfe_train, y_train))
    print("Test R^2: ", lr.score(X_rfe_test, y_test))
    print(f"Using {n} out of {X_train.shape[1]} features")
    print()
```
Now test out `Lasso` ([documentation here](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)) with several different `alpha` values.

```python
for alpha in [1, 10, 100, 1000, 10000]:
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train, y_train)

    print("Train R^2:", lasso.score(X_train, y_train))
    print("Test R^2: ", lasso.score(X_test, y_test))
    print(f"Using {sum(abs(lasso.coef_) < 10**(-10))} out of {X_train.shape[1]} features")
    print("and an alpha of", alpha)
    print()
```
Compare the results. Which features would you choose to use?

```python
"""
For RFE the model with the best test R-Squared was using 20 features

For Lasso the model with the best test R-Squared was using an alpha of 10000

The Lasso result was a bit better so let's go with that and the 14 features
that it selected
"""
```
## Evaluate Final Model on Validation Data

### Data Preparation

At the start of this lab, we created `X_val` and `y_val`. Prepare `X_val` the same way that `X_train` and `X_test` have been prepared. This includes scaling, adding interactions, and adding polynomial terms.

```python

# Scale X_val
X_val_scaled = scaler.transform(X_val)
X_val = pd.DataFrame(X_val_scaled, columns=X.columns)

# Add interactions to X_val
for record in top_7_interactions:
    col1, col2 = record[0]
    new_col_name = col1 + "_" + col2
    X_val[new_col_name] = X_val[col1] * X_val[col2]

# Add polynomials to X_val
for (col, degree, _) in top_polynomials.values:
    poly = PolynomialFeatures(degree, include_bias=False)
    col_transformed_val = pd.DataFrame(poly.fit_transform(X_val[[col]]),
                                        columns=poly.get_feature_names([col]))
    X_val = pd.concat([X_val.drop(col, axis=1), col_transformed_val], axis=1)
```
### Evaluation

Using either `RFE` or `Lasso`, fit a model on the complete train + test set, then find R-Squared and MSE values for the validation set.

```python
final_model = Lasso(alpha=10000)
final_model.fit(pd.concat([X_train, X_test]), pd.concat([y_train, y_test]))

print("R-Squared:", final_model.score(X_val, y_val))
print("MSE:", mean_squared_error(y_val, final_model.predict(X_val)))
```
## Level Up Ideas (Optional)

### Create a Lasso Path

From this section, you know that when using `Lasso`, more parameters shrink to zero as your regularization parameter goes up. In scikit-learn there is a function `lasso_path()` which visualizes the shrinkage of the coefficients while $alpha$ changes. Try this out yourself!

https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_coordinate_descent_path.html#sphx-glr-auto-examples-linear-model-plot-lasso-coordinate-descent-path-py

### AIC and BIC for Subset Selection

This notebook shows how you can use AIC and BIC purely for feature selection. Try this code out on our Ames housing data!

https://xavierbourretsicotte.github.io/subset_selection.html

## Summary

Congratulations! You now know how to apply concepts of bias-variance tradeoff using extensions to linear models and feature selection.


-----File-Boundary-----
