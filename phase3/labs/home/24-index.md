# Fitting a Logistic Regression Model - Lab

## Introduction

In the last lesson you were given a broad overview of logistic regression. This included an introduction to two separate packages for creating logistic regression models. In this lab, you'll be investigating fitting logistic regressions with `statsmodels`. For your first foray into logistic regression, you are going to attempt to build a model that classifies whether an individual survived the [Titanic](https://www.kaggle.com/c/titanic/data) shipwreck or not (yes, it's a bit morbid).


## Objectives

In this lab you will:

* Implement logistic regression with `statsmodels`
* Interpret the statistical results associated with model parameters

## Import the data

Import the data stored in the file `'titanic.csv'` and print the first five rows of the DataFrame to check its contents.

```python
# Import the data
import pandas as pd

df = pd.read_csv('titanic.csv')
df.head()
```
## Define independent and target variables

Your target variable is in the column `'Survived'`. A `0` indicates that the passenger didn't survive the shipwreck. Print the total number of people who didn't survive the shipwreck. How many people survived?

```python
# Total number of people who survived/didn't survive
df['Survived'].value_counts()
```
Only consider the columns specified in `relevant_columns` when building your model. The next step is to create dummy variables from categorical variables. Remember to drop the first level for each categorical column and make sure all the values are of type `float`:

```python
# Create dummy variables
relevant_columns = ['Pclass', 'Age', 'SibSp', 'Fare', 'Sex', 'Embarked', 'Survived']
dummy_dataframe = pd.get_dummies(df[relevant_columns], drop_first=True, dtype=float)

dummy_dataframe.shape
```
Did you notice above that the DataFrame contains missing values? To keep things simple, simply delete all rows with missing values.

> NOTE: You can use the [`.dropna()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.dropna.html) method to do this.

```python
# Drop missing rows
dummy_dataframe = dummy_dataframe.dropna()
dummy_dataframe.shape
```
Finally, assign the independent variables to `X` and the target variable to `y`:

```python
# Split the data into X and y
y = dummy_dataframe['Survived']
X = dummy_dataframe.drop(columns=['Survived'], axis=1)
```
## Fit the model

Now with everything in place, you can build a logistic regression model using `statsmodels` (make sure you create an intercept term as we showed in the previous lesson).

> Warning: Did you receive an error of the form "LinAlgError: Singular matrix"? This means that `statsmodels` was unable to fit the model due to certain linear algebra computational problems. Specifically, the matrix was not invertible due to not being full rank. In other words, there was a lot of redundant, superfluous data. Try removing some features from the model and running it again.

```python
# Build a logistic regression model using statsmodels
import statsmodels.api as sm
X = sm.tools.add_constant(X)
logit_model = sm.Logit(y, X)
result = logit_model.fit()
```
## Analyze results

Generate the summary table for your model. Then, comment on the p-values associated with the various features you chose.

```python
# Summary table
result.summary()
```
```python
# Based on our P-values, most of the current features appear to be significant based on a .05 significance level.
# That said, the 'Embarked' and 'Fare' features were not significant based on their higher p-values.
```
## Level up (Optional)

Create a new model, this time only using those features you determined were influential based on your analysis of the results above. How does this model perform?

```python
# Your code here
relevant_columns = ['Pclass', 'Age', 'SibSp', 'Sex', 'Survived']
dummy_dataframe = pd.get_dummies(df[relevant_columns], drop_first=True, dtype=float)

dummy_dataframe = dummy_dataframe.dropna()

y = dummy_dataframe['Survived']
X = dummy_dataframe.drop(columns=['Survived'], axis=1)

X = sm.tools.add_constant(X)
logit_model = sm.Logit(y, X)
result = logit_model.fit()

result.summary()
```
```python
# Comments:
# Note how removing the insignificant features had little impact on the $R^2$ value
# of our model.
```
## Summary

Well done! In this lab, you practiced using `statsmodels` to build a logistic regression model. You then interpreted the results, building upon your previous stats knowledge, similar to linear regression. Continue on to take a look at building logistic regression models in Scikit-learn!


-----File-Boundary-----
# Logistic Regression in scikit-learn - Lab

## Introduction

In this lab, you are going to fit a logistic regression model to a dataset concerning heart disease. Whether or not a patient has heart disease is indicated in the column labeled `'target'`. 1 is for positive for heart disease while 0 indicates no heart disease.

## Objectives

In this lab you will:

- Fit a logistic regression model using scikit-learn


## Let's get started!

Run the following cells that import the necessary functions and import the dataset:

```python
# Import necessary functions
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
```
```python
# Import data
df = pd.read_csv('heart.csv')
df.head()
```
## Define appropriate `X` and `y`

Recall the dataset contains information about whether or not a patient has heart disease and is indicated in the column labeled `'target'`. With that, define appropriate `X` (predictors) and `y` (target) in order to model whether or not a patient has heart disease.

```python
# Split the data into target and predictors
y = df['target']
X = df.drop(columns=['target'], axis=1)
```
## Normalize the data

Normalize the data (`X`) prior to fitting the model.

```python
X = X.apply(lambda x : (x - x.min()) /(x.max() - x.min()), axis=0)
X.head()
```
## Train- test split

- Split the data into training and test sets
- Assign 25% to the test set
- Set the `random_state` to 0

```python
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
```
## Fit a model

- Instantiate `LogisticRegression`
  - Make sure you don't include the intercept
  - set `C` to a very large number such as `1e12`
  - Use the `'liblinear'` solver
- Fit the model to the training data

```python
# Instantiate the model
logreg = LogisticRegression(fit_intercept=False, C=1e12, solver='liblinear')

# Fit the model
logreg.fit(X_train, y_train)
```
## Predict
Generate predictions for the training and test sets.

```python
# Generate predictions
y_hat_train = logreg.predict(X_train)
y_hat_test = logreg.predict(X_test)
```
## How many times was the classifier correct on the training set?

```python
# We could subtract the two columns. If values or equal, difference will be zero. Then count number of zeros.
residuals = np.abs(y_train - y_hat_train)
print(pd.Series(residuals).value_counts())
print('------------------------------------')
print(pd.Series(residuals).value_counts(normalize=True))
# 194 correct, ~ 85% accuracy
```
## How many times was the classifier correct on the test set?

```python
# We could subtract the two columns. If values or equal, difference will be zero. Then count number of zeros.
residuals = np.abs(y_test - y_hat_test)
print(pd.Series(residuals).value_counts())
print('------------------------------------')
print(pd.Series(residuals).value_counts(normalize=True))
# 62 correct, ~ 82% accuracy
```
## Analysis
Describe how well you think this initial model is performing based on the training and test performance. Within your description, make note of how you evaluated performance as compared to your previous work with regression.

```python
"""
Answers will vary. In this instance, our model has 85% accuracy on the train set and 83% on the test set.
You can also see that our model has a reasonably even number of False Positives and False Negatives,
with slightly more False Positives for both the training and testing validations.
"""
```
## Summary

In this lab, you practiced a standard data science pipeline: importing data, split it into training and test sets, and fit a logistic regression model. In the upcoming labs and lessons, you'll continue to investigate how to analyze and tune these models for various scenarios.


-----File-Boundary-----
# Gradient Descent - Lab

## Introduction

In this lab, you'll continue to formalize your knowledge of gradient descent by coding the algorithm yourself. In the upcoming labs, you'll apply similar procedures to implement logistic regression on your own.


## Objectives

In this lab you will:


- Implement gradient descent from scratch to minimize OLS

## Use gradient descent to minimize OLS

To practice gradient descent, you'll investigate a simple regression case in which you're looking to minimize the Residual Sum of Squares (RSS) between the predictions and the actual values. Remember that this is referred to as Ordinary Least Squares (OLS) regression. You'll compare two simplistic models and use gradient descent to improve upon these initial models.


## Load the dataset

- Import the file `'movie_data.xlsx'` using Pandas
- Print the first five rows of the data

> You can use the `read_excel()` function to import an Excel file.

```python
# Import the data
import pandas as pd

# Print the first five rows of the data
df = pd.read_excel('movie_data.xlsx')
df.head()
```
## Two simplistic models

Imagine someone is attempting to predict the domestic gross sales of a movie based on the movie's budget, or at least further investigate how these two quantities are related. Two models are suggested and need to be compared.
The two models are:

$\text{domgross} = 1.575 \cdot \text{budget}$
$\text{domgross} = 1.331 \cdot \text{budget}$


Here's a graph of the two models along with the actual data:

```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

x = np.linspace(start=df['budget'].min(), stop=df['budget'].max(), num=10**5)
plt.scatter(x, 1.575*x, label='Mean Ratio Model') # Model 1
plt.scatter(x, 1.331*x, label='Median Ratio Model') # Model 2
plt.scatter(df['budget'], df['domgross'], label='Actual Data Points')
plt.title('Gross Domestic Sales vs. Budget', fontsize=18)
plt.xlabel('Budget', fontsize=16)
plt.ylabel('Gross Domestic Sales', fontsize=16)
plt.legend(bbox_to_anchor=(1, 1))
plt.show()
```
## Error/Loss functions

To compare the two models (and future ones), a metric for evaluating and comparing models to each other is needed. Traditionally, this is the residual sum of squares. As such you are looking to minimize  $ \sum(\hat{y}-y)^2$.
Write a function `rss()` which calculates the residual sum of squares for a simplistic model:

$\text{domgross} = m \cdot \text{budget}$

```python
def rss(m, X=df['budget'], y=df['domgross']):
    model = m * X
    residuals = model - y
    total_rss = residuals.map(lambda x: x**2).sum()
    return total_rss
```
## Find the RSS for the two models
Which of the two models is better?

```python
# Your code here
print('Model 1 RSS:', rss(1.575))
print('Model 2 RSS:', rss(1.331))
```
```python
# Your response here
"""
The second model is mildly better.
"""
```
## Gradient descent

Now that you have a loss function, you can use numerical methods to find a minimum to the loss function. By minimizing the loss function, you have achieved an optimal solution according to the problem formulation. Here's the outline of gradient descent from the previous lesson:

1. Define initial parameters:
    1. pick a starting point
    2. pick a step size $\alpha$ (alpha)
    3. choose a maximum number of iterations; the algorithm will terminate after this many iterations if a minimum has yet to be found
    4. (optionally) define a precision parameter; similar to the maximum number of iterations, this will terminate the algorithm early. For example, one might define a precision parameter of 0.00001, in which case if the change in the loss function were less than 0.00001, the algorithm would terminate. The idea is that we are very close to the bottom and further iterations would make a negligible difference
2. Calculate the gradient at the current point (initially, the starting point)
3. Take a step (of size alpha) in the direction of the gradient
4. Repeat steps 2 and 3 until the maximum number of iterations is met, or the difference between two points is less then your precision parameter

To start, visualize the cost function. Plot the cost function output for a range of m values from -3 to 5.

```python
# Your code here
x = np.linspace(start=-3, stop=5, num=10**3)
y = [rss(xi) for xi in x]
plt.plot(x, y)
plt.title('RSS Loss Function for Various Values of m')
plt.show()
```
As you can see, this is a simple cost function. The minimum is clearly around 1. With that, it's time to implement gradient descent in order to find the optimal value for m.

```python
# The algorithm starts at x=1.5
cur_x = 1.5

# Initialize a step size
alpha = 1*10**(-7)

# Initialize a precision
precision = 0.0000000001

# Helpful initialization
previous_step_size = 1

# Maximum number of iterations
max_iters = 10000

# Iteration counter
iters = 0

# Create a loop to iterate through the algorithm until either the max_iteration or precision conditions is met
while (previous_step_size > precision) & (iters < max_iters):
    print('Current value: {} RSS Produced: {}'.format(cur_x, rss(cur_x)))
    prev_x = cur_x
    # Calculate the gradient. This is often done by hand to reduce computational complexity.
    # For here, generate points surrounding your current state, then calculate the rss of these points
    # Finally, use the np.gradient() method on this survey region.
    # This code is provided here to ease this portion of the algorithm implementation
    x_survey_region = np.linspace(start = cur_x - previous_step_size , stop = cur_x + previous_step_size , num = 101)
    rss_survey_region = [np.sqrt(rss(m)) for m in x_survey_region]
    gradient = np.gradient(rss_survey_region)[50]
    cur_x -= alpha * gradient # Move opposite the gradient
    previous_step_size = abs(cur_x - prev_x)
    iters+=1


# The output for the above will be: ('The local minimum occurs at', 1.1124498053361267)
print("The local minimum occurs at", cur_x)
```
## Plot the minimum on your graph
Replot the RSS cost curve as above. Add a red dot for the minimum of this graph using the solution from your gradient descent function above.

```python
# Your code here
x = np.linspace(start=-3, stop=5, num=10**3)
y = [rss(xi) for xi in x]
plt.plot(x, y)
plt.scatter(1.1124498053361267, rss(1.1124498053361267), c='red')
plt.title('RSS Loss Function for Various Values of m, with minimum marked')
plt.show()
```
## Summary

In this lab, you coded up a gradient descent algorithm from scratch! In the next lab, you'll apply this to logistic regression in order to create a full implementation yourself!


-----File-Boundary-----
