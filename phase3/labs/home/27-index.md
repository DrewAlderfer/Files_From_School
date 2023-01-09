# Distance Metrics - Lab

## Introduction

In this lab, you'll calculate various distances between multiple points using the distance metrics you learned about!

## Objectives

In this lab you will:

- Calculate Manhattan distance between two points
- Calculate Euclidean distance between two points
- Calculate Minkowski distance between two points

## Getting Started

You'll start by writing a generalized function to calculate any of the three distance metrics you've learned about. Let's review what you know so far:

> The **_Manhattan distance_** and **_Euclidean distance_** are both special cases of **_Minkowski distance_**.


Take a look at the formula for Minkowski distance below:

$$\large d(x,y) = \left(\sum_{i=1}^{n}|x_i - y_i|^c\right)^\frac{1}{c}$$

**_Manhattan distance_** is a special case where $c=1$ in the equation above (which means that you can remove the root operation and just keep the summation).

**_Euclidean distance_** is a special case where $c=2$ in the equation above.

Knowing this, you can create a generalized `distance()` function that calculates Minkowski distance, and takes in `c` as a parameter. That way, you can use the same function for every problem, and still calculate Manhattan and Euclidean distance metrics by just passing in the appropriate values for the `c` parameter!

In the cell below:

* Complete the `distance()` function which should implement the Minkowski distance equation above to return the distance, a single number
* This function should take in 4 arguments:
    * `a`: a tuple or array that describes a vector in n-dimensional space
    * `b`: a tuple or array that describes a vector in n-dimensional space (this must be the same length as `a`!)
    * `c`: which tells us the norm to calculate the vector space (if set to `1`, the result will be Manhattan, while `2` will calculate Euclidean distance)
    * `verbose`: set to `True` by default. If true, the function should print out if the distance metric returned is a measurement of Manhattan, Euclidean, or Minkowski distance

* Since euclidean distance is the most common distance metric used, this function should default to using `c=2` if no value is set for `c`


**_HINT:_**

1. You can avoid using a `for` loop like we did in the previous lesson by simply converting the tuples to NumPy arrays

2. Use `np.power()` as an easy way to implement both squares and square roots. `np.power(a, 3)` will return the cube of `a`, while `np.power(a, 1/3)` will return the cube root of 3. For more information on this function, refer the [NumPy documentation](https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.power.html)!

```python
import numpy as np

def distance(a, b, c=2, verbose=True):
    if len(a) != len(b):
        raise ValueError("Both vectors must be of equal length!")

    if verbose:
        if c == 1:
            print("Calculating Manhattan distance:")
        elif c == 2:
            print("Calculating Euclidean distance:")
        else:
            print(f"Calcuating Minkowski distance (c={c}):")

    return np.power(np.sum(np.power(np.abs(np.array(a) - np.array(b)), c)), 1/c)

test_point_1 = (1, 2)
test_point_2 = (4, 6)
print(distance(test_point_1, test_point_2)) # Expected Output: 5.0
print(distance(test_point_1, test_point_2, c=1)) # Expected Output: 7.0
print(distance(test_point_1, test_point_2, c=3)) # Expected Output: 4.497941445275415
```
Great job!

Now, use your function to calculate distances between points:

## Problem 1

Calculate the **_Euclidean distance_** between the following points in 5-dimensional space:

Point 1: (-2, -3.4, 4, 15, 7)

Point 2: (3, -1.2, -2, -1, 7)

```python
print(distance((-2, -3.4, 4, 15, 7), (3, -1.2, -2, -1, 7))) # Expected Output: 17.939899665271266
```
## Problem 2

Calculate the **_Manhattan distance_** between the following points in 10-dimensional space:

Point 1: \[0, 0, 0, 7, 16, 2, 0, 1, 2, 1\]
Point 2: \[1, -1, 5, 7, 14, 3, -2, 3, 3, 6\]

```python
print(distance( [0, 0, 0, 7, 16, 2, 0, 1, 2, 1],  [1, -1, 5, 7, 14, 3, -2, 3, 3, 6], c=1)) # Expected Output: 20.0
```
## Problem 3

Calculate the **_Minkowski distance_** with a norm of 3.5 between the following points:

Point 1: (-2, 7, 3.4)
Point 2: (3, 4, 1.5)

```python
print(distance((-2, 7, 3.4), (3, 4, 1.5), c=3.5)) # Expected Output: 5.268789659188307
```
## Summary

Great job! Now that you know about the various distance metrics, you can use them to writing a K-Nearest Neighbors classifier from scratch!


-----File-Boundary-----
# K-Nearest Neighbors - Lab

## Introduction

In this lesson, you'll build a simple version of a **_K-Nearest Neigbors classifier_** from scratch, and train it to make predictions on a dataset!

## Objectives

In this lab you will:

* Implement a basic KNN algorithm from scratch

## Getting Started

You'll begin this lab by creating a classifier. To keep things simple, you'll be using a helper function, `euclidean()`, from the `spatial.distance` module of the `scipy` library. Import this function in the cell below:

```python
from scipy.spatial.distance import euclidean
import numpy as np
```
## Create the `KNN` class

You will now:

* Create an class called `KNN`
* This class should contain two empty methods -- `fit` and `predict`

```python
# Define the KNN class with two empty methods - fit and predict
class KNN:

    def fit():
        pass

    def predict():
        pass
```
## Comple the `fit()` method

Recall that when "fitting" a KNN classifier, all you're really doing is storing the points and their corresponding labels. There's no actual "fitting" involved here, since all you need to do is store the data so that you can use it to calculate the nearest neighbors when the `predict()` method is called.

The inputs for this function should be:

* `self`: since this will be an instance method inside the `KNN` class
* `X_train`: an array, each row represents a _vector_ for a given point in space
* `y_train`: the corresponding labels for each vector in `X_train`. The label at `y_train[0]` is the label that corresponds to the vector at `X_train[0]`, and so on

In the cell below, complete the `fit` method:

```python
def fit(self, X_train, y_train):
    self.X_train = X_train
    self.y_train = y_train

# This line updates the knn.fit method to point to the function you've just written
KNN.fit = fit
```
### Helper functions

Next, you will write three helper functions to make things easier when completing the `predict()` method.

In the cell below, complete the `_get_distances()` function. This function should:

* Take in two arguments: `self` and `x`
* Create an empty array, `distances`, to hold all the distances you're going to calculate
* Enumerate through every item in `self.X_train`. For each item:
    * Use the `euclidean()` function to get the distance between x and the current point from `X_train`
    * Create a tuple containing the index and the distance (in that order!) and append it to the `distances` array
* Return the `distances` array when a distance has been generated for all items in `self.X_train`

```python
def _get_distances(self, x):
    distances = []
    for ind, val in enumerate(self.X_train):
        dist_to_i = euclidean(x, val)
        distances.append((ind, dist_to_i))
    return distances

# This line attaches the function you just created as a method to KNN class
KNN._get_distances = _get_distances
```
Well done! You will now create a `_get_k_nearest()` function to retrieve indices of the k-nearest points. This function should:

* Take three arguments:
    * `self`
    * `dists`: an array of tuples containing (index, distance), which will be output from the `_get_distances()` method.
    * `k`: the number of nearest neighbors you want to return
* Sort the `dists` array by distances values, which are the second element in each tuple
* Return the first `k` tuples from the sorted array

**_Hint:_** To easily sort on the second item in the tuples contained within the `dists` array, use the `sorted()` function and pass in lambda for the `key=` parameter. To sort on the second element of each tuple, you can just use `key=lambda x: x[1]`!

```python
def _get_k_nearest(self, dists, k):
    sorted_dists = sorted(dists, key=lambda x: x[1])
    return sorted_dists[:k]

# This line attaches the function you just created as a method to KNN class
KNN._get_k_nearest = _get_k_nearest
```
The final helper function you'll create will get the labels that correspond to each of the k-nearest point, and return the class that occurs the most.

Complete the `_get_label_prediction()` function in the cell below. This function should:

* Create a list containing the labels from `self.y_train` for each index in `k_nearest` (remember, each item in `k_nearest` is a tuple, and the index is stored as the first item in each tuple)
* Get the total counts for each label (use `np.bincount()` and pass in the label array created in the previous step)
* Get the index of the label with the highest overall count in counts (use `np.argmax()` for this, and pass in the counts created in the previous step)

```python
def _get_label_prediction(self, k_nearest):

    labels = [self.y_train[i] for i, _ in k_nearest]
    counts = np.bincount(labels)
    return np.argmax(counts)

# This line attaches the function you just created as a method to KNN class
KNN._get_label_prediction = _get_label_prediction
```
Great! Now, you now have all the ingredients needed to complete the `predict()` method.

## Complete the `predict()` method

This method does all the heavy lifting for KNN, so this will be a bit more complex than the `fit()` method. Here's an outline of how this method should work:

* In addition to `self`, our `predict` function should take in two arguments:
    * `X_test`: the points we want to classify
    * `k`: which specifies the number of neighbors we should use to make the classification.  Set `k=3` as a default, but allow the user to update it if they choose
* Your method will need to iterate through every item in `X_test`. For each item:
    * Calculate the distance to all points in `X_train` by using the `._get_distances()` helper method
    * Find the k-nearest points in `X_train` by using the `._get_k_nearest()` method
    * Use the index values contained within the tuples returned by `._get_k_nearest()` method to get the corresponding labels for each of the nearest points
    * Determine which class is most represented in these labels and treat that as the prediction for this point. Append the prediction to `preds`
* Once a prediction has been generated for every item in `X_test`, return `preds`

Follow these instructions to complete the `predict()` method in the cell below:

```python
def predict(self, X_test, k=3):
    preds = []
    # Iterate through each item in X_test
    for i in X_test:
        # Get distances between i and each item in X_train
        dists = self._get_distances(i)
        k_nearest = self._get_k_nearest(dists, k)
        predicted_label = self._get_label_prediction(k_nearest)
        preds.append(predicted_label)
    return preds

# This line updates the knn.predict method to point to the function you've just written
KNN.predict = predict
```
Great! Now, try out your new KNN classifier on a sample dataset to see how well it works!

## Test the KNN classifier

In order to test the performance of your model, import the **_Iris dataset_**. Specifically:

- Use the `load_iris()` function, which can be found inside of the `sklearn.datasets` module. Then call this function, and use the object it returns
- Import `train_test_split()` from `sklearn.model_selection`, as well as `accuracy_score()` from `sklearn.metrics`
- Assign the `.data` attribute of `iris` to `data` and the `.target` attribute to `target`

Note that there are **_3 classes_** in the Iris dataset, making this a multi-categorical classification problem. This means that you can't use evaluation metrics that are meant for binary classification problems. For this, just stick to accuracy for now.

```python
# Import the necessary functions
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()
data = iris.data
target = iris.target
```
Use `train_test_split()` to split the data into training and test sets. Pass in the `data` and `target`, and set the `test_size` to 0.25 and `random_state` to 0.

```python
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=0)
```
Now, instantiate the `KNN` class, and `fit` it to the data in `X_train` and the labels in `y_train`.

```python
# Instantiate and fit KNN
knn = KNN()
knn.fit(X_train, y_train)
```
In the cell below, use the `.predict()` method to generate predictions for the data stored in `X_test`:

```python
# Generate predictions
preds = knn.predict(X_test)
```
Finally, the moment of truth! Test the accuracy of your predictions. In the cell below, complete the call to `accuracy_score()` by passing in `y_test` and `preds`!

```python
print("Testing Accuracy: {}".format(accuracy_score(y_test, preds)))
# Expected Output: Testing Accuracy: 0.9736842105263158
```
Over 97% accuracy! Not bad for a handwritten machine learning classifier!

## Summary

That was great! Next, you'll dive a little deeper into evaluating performance of a KNN algorithm!


-----File-Boundary-----
# KNN with scikit-learn - Lab

## Introduction

In this lab, you'll learn how to use scikit-learn's implementation of a KNN classifier on the classic Titanic dataset from Kaggle!


## Objectives

In this lab you will:

- Conduct a parameter search to find the optimal value for K
- Use a KNN classifier to generate predictions on a real-world dataset
- Evaluate the performance of a KNN model


## Getting Started

Start by importing the dataset, stored in the `titanic.csv` file, and previewing it.

```python
# Import pandas and set the standard alias
import pandas as pd

# Import the data from 'titanic.csv' and store it in a pandas DataFrame
raw_df = pd.read_csv('titanic.csv')

# Print the head of the DataFrame to ensure everything loaded correctly
raw_df.head()
```
Great!  Next, you'll perform some preprocessing steps such as removing unnecessary columns and normalizing features.

## Preprocessing the data

Preprocessing is an essential component in any data science pipeline. It's not always the most glamorous task as might be an engaging data visual or impressive neural network, but cleaning and normalizing raw datasets is very essential to produce useful and insightful datasets that form the backbone of all data powered projects. This can include changing column types, as in:


```python
df['col_name'] = df['col_name'].astype('int')
```
Or extracting subsets of information, such as:

```python
import re
df['street'] = df['address'].map(lambda x: re.findall('(.*)?\n', x)[0])
```

> **Note:** While outside the scope of this particular lesson, **regular expressions** (mentioned above) are powerful tools for pattern matching! See the [regular expressions official documentation here](https://docs.python.org/3.6/library/re.html).

Since you've done this before, you should be able to do this quite well yourself without much hand holding by now. In the cells below, complete the following steps:

1. Remove unnecessary columns (`'PassengerId'`, `'Name'`, `'Ticket'`, and `'Cabin'`)
2. Convert `'Sex'` to a binary encoding, where female is `0` and male is `1`
3. Detect and deal with any missing values in the dataset:
    * For `'Age'`, replace missing values with the median age for the dataset
    * For `'Embarked'`, drop the rows that contain missing values
4. One-hot encode categorical columns such as `'Embarked'`
5. Store the target column, `'Survived'`, in a separate variable and remove it from the DataFrame

```python
# Drop the unnecessary columns
df = raw_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=False)
df.head()
```
```python
# Convert Sex to binary encoding
df['Sex'] = df['Sex'].map({'female': 0, 'male': 1})
df.head()
```
```python
# Find the number of missing values in each column
df.isna().sum()
```
```python
# Impute the missing values in 'Age'
df['Age'] = df['Age'].fillna(df.Age.median())
df.isna().sum()
```
```python
# Drop the rows missing values in the 'Embarked' column
df = df.dropna()
df.isna().sum()
```
```python
# One-hot encode the categorical columns
one_hot_df = pd.get_dummies(df)
one_hot_df.head()
```
```python
labels = one_hot_df['Survived']
one_hot_df.drop('Survived', axis=1, inplace=True)
```
## Create training and test sets

Now that you've preprocessed the data, it's time to split it into training and test sets.

In the cell below:

* Import `train_test_split` from the `sklearn.model_selection` module
* Use `train_test_split()` to split the data into training and test sets, with a `test_size` of `0.25`. Set the `random_state` to 42

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(one_hot_df, labels, test_size=0.25, random_state=42)
```
## Normalizing the data

The final step in your preprocessing efforts for this lab is to **_normalize_** the data. We normalize **after** splitting our data into training and test sets. This is to avoid information "leaking" from our test set into our training set (read more about data leakage [here](https://machinelearningmastery.com/data-leakage-machine-learning/) ). Remember that normalization (also sometimes called **_Standardization_** or **_Scaling_**) means making sure that all of your data is represented at the same scale. The most common way to do this is to convert all numerical values to z-scores.

Since KNN is a distance-based classifier, if data is in different scales, then larger scaled features have a larger impact on the distance between points.

To scale your data, use `StandardScaler` found in the `sklearn.preprocessing` module.

In the cell below:

* Import and instantiate `StandardScaler`
* Use the scaler's `.fit_transform()` method to create a scaled version of the training dataset
* Use the scaler's `.transform()` method to create a scaled version of the test dataset
* The result returned by `.fit_transform()` and `.transform()` methods will be numpy arrays, not a pandas DataFrame. Create a new pandas DataFrame out of this object called `scaled_df`. To set the column names back to their original state, set the `columns` parameter to `one_hot_df.columns`
* Print the head of `scaled_df` to ensure everything worked correctly

```python
# Import StandardScaler
from sklearn.preprocessing import StandardScaler

# Instantiate StandardScaler
scaler = StandardScaler()

# Transform the training and test sets
scaled_data_train = scaler.fit_transform(X_train)
scaled_data_test = scaler.transform(X_test)

# Convert into a DataFrame
scaled_df_train = pd.DataFrame(scaled_data_train, columns=one_hot_df.columns)
scaled_df_train.head()
```
You may have noticed that the scaler also scaled our binary/one-hot encoded columns, too! Although it doesn't look as pretty, this has no negative effect on the model. Each 1 and 0 have been replaced with corresponding decimal values, but each binary column still only contains 2 values, meaning the overall information content of each column has not changed.

## Fit a KNN model

Now that you've preprocessed the data it's time to train a KNN classifier and validate its accuracy.

In the cells below:

* Import `KNeighborsClassifier` from the `sklearn.neighbors` module
* Instantiate the classifier. For now, you can just use the default parameters
* Fit the classifier to the training data/labels
* Use the classifier to generate predictions on the test data. Store these predictions inside the variable `test_preds`

```python
# Import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier

# Instantiate KNeighborsClassifier
clf = KNeighborsClassifier()

# Fit the classifier
clf.fit(scaled_data_train, y_train)

# Predict on the test set
test_preds = clf.predict(scaled_data_test)
```
## Evaluate the model

Now, in the cells below, import all the necessary evaluation metrics from `sklearn.metrics` and complete the `print_metrics()` function so that it prints out **_Precision, Recall, Accuracy, and F1-Score_** when given a set of `labels` (the true values) and `preds` (the models predictions).

Finally, use `print_metrics()` to print the evaluation metrics for the test predictions stored in `test_preds`, and the corresponding labels in `y_test`.

```python
# Import the necessary functions
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
```
```python
# Complete the function
def print_metrics(labels, preds):
    print("Precision Score: {}".format(precision_score(labels, preds)))
    print("Recall Score: {}".format(recall_score(labels, preds)))
    print("Accuracy Score: {}".format(accuracy_score(labels, preds)))
    print("F1 Score: {}".format(f1_score(labels, preds)))

print_metrics(y_test, test_preds)
```
> Interpret each of the metrics above, and explain what they tell you about your model's capabilities. If you had to pick one score to best describe the performance of the model, which would you choose? Explain your answer.

Write your answer below this line:


________________________________________________________________________________



## Improve model performance

While your overall model results should be better than random chance, they're probably mediocre at best given that you haven't tuned the model yet. For the remainder of this notebook, you'll focus on improving your model's performance. Remember that modeling is an **_iterative process_**, and developing a baseline out of the box model such as the one above is always a good start.

First, try to find the optimal number of neighbors to use for the classifier. To do this, complete the `find_best_k()` function below to iterate over multiple values of K and find the value of K that returns the best overall performance.

The function takes in six arguments:
* `X_train`
* `y_train`
* `X_test`
* `y_test`
* `min_k` (default is 1)
* `max_k` (default is 25)

> **Pseudocode Hint**:
1. Create two variables, `best_k` and `best_score`
1. Iterate through every **_odd number_** between `min_k` and `max_k + 1`.
    1. For each iteration:
        1. Create a new `KNN` classifier, and set the `n_neighbors` parameter to the current value for k, as determined by the loop
        1. Fit this classifier to the training data
        1. Generate predictions for `X_test` using the fitted classifier
        1. Calculate the **_F1-score_** for these predictions
        1. Compare this F1-score to `best_score`. If better, update `best_score` and `best_k`
1. Once all iterations are complete, print the best value for k and the F1-score it achieved

```python
def find_best_k(X_train, y_train, X_test, y_test, min_k=1, max_k=25):
    best_k = 0
    best_score = 0.0
    for k in range(min_k, max_k+1, 2):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        preds = knn.predict(X_test)
        f1 = f1_score(y_test, preds)
        if f1 > best_score:
            best_k = k
            best_score = f1

    print("Best Value for k: {}".format(best_k))
    print("F1-Score: {}".format(best_score))
```
```python
find_best_k(scaled_data_train, y_train, scaled_data_test, y_test)
# Expected Output:

# Best Value for k: 17
# F1-Score: 0.7468354430379746
```
If all went well, you'll notice that model performance has improved by 3 percent by finding an optimal value for k. For further tuning, you can use scikit-learn's built-in `GridSearch()` to perform a similar exhaustive check of hyperparameter combinations and fine tune model performance. For a full list of model parameters, see the [sklearn documentation !](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)



## (Optional) Level Up: Iterating on the data

As an optional (but recommended!) exercise, think about the decisions you made during the preprocessing steps that could have affected the overall model performance. For instance, you were asked to replace the missing age values with the column median. Could this have affected the overall performance? How might the model have fared if you had just dropped those rows, instead of using the column median? What if you reduced the data's dimensionality by ignoring some less important columns altogether?

In the cells below, revisit your preprocessing stage and see if you can improve the overall results of the classifier by doing things differently. Consider dropping certain columns, dealing with missing values differently, or using an alternative scaling function. Then see how these different preprocessing techniques affect the performance of the model. Remember that the `find_best_k()` function handles all of the fitting; use this to iterate quickly as you try different strategies for dealing with data preprocessing!

```python
```
```python
```
```python
```
```python
```
## Summary

Well done! In this lab, you worked with the classic Titanic dataset and practiced fitting and tuning KNN classification models using scikit-learn! As always, this gave you another opportunity to continue practicing your data wrangling skills and model tuning skills using Pandas and scikit-learn!


-----File-Boundary-----
