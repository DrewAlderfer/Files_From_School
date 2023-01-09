# Model Tuning and Pipelines - Introduction

## Introduction

Now that you have learned the basics of a supervised learning workflow, it's time to get into some more-advanced techniques! In this section you'll learn about tools for tuning model hyperparameters, building pipelines, and persisting your trained model on disk.

## Tuning Model Hyperparameters with GridSearchCV

With non-parametric models such as decision trees and k-nearest neighbors, you have seen that there are various hyperparameters that you can specify when you instantiate the model. For example, the maximum depth of the tree, or the number of neighbors. Often these hyperparameters help to balance the bias-variance trade-off between underfitting and overfitting and are important for finding the optimal model.

With so many different hyperparameter combinations to try out, it can be difficult to write clean, readable code. Fortunately there is a tool from scikit-learn called `GridSearchCV` that is specifically designed to search through a "grid" of hyperparameters! In this section we'll introduce how to use this tool.


## Machine Learning Pipelines

Pipelines are extremely useful for allowing data scientists to quickly and consistently transform data, train machine learning models, and make predictions.

By now, you know that the data science process is a flow of activities, from inspecting the data to cleaning it, transforming it, running a model, and discussing the results. Wouldn't it be nice if there was a streamlined process to create nice machine learning workflows? Enter the `Pipeline` class in scikit-learn!

In this section, you'll learn how you can use a pipeline to integrate several steps of the machine learning workflow. Additionally, you'll compare several classification techniques with each other, and integrate grid search in your pipeline so you can tune several hyperparameters in each of the machine learning models while also avoiding data leakage.

## Pickle and Model Deployment

So far, as soon as you shut down your notebook kernel, your model ceases to exist. If you wanted to use the model to make predictions again, you would need to re-train the model. This is time-consuming and makes your model a lot less useful.

Luckily there are techniques to *pickle* your model -- basically, to store the model for later, so that it can be loaded and can make predictions without being trained again. Pickled models are also typically used in the context of model deployment, where your model can be used as the backend of an API!

## Summary

This section only scratches the surface of the advanced modeling tools you might use as a data scientist. Get ready to optimize your workflow and get beyond the basics!


-----File-Boundary-----
# GridSearchCV

## Introduction

In this lesson, we'll explore the concept of parameter tuning to maximize our model performance using a combinatorial grid search!

## Objectives

You will be able to:


- Design a parameter grid for use with scikit-learn's `GridSearchCV`
- Use `GridSearchCV` to increase model performance through parameter tuning


## Parameter tuning

By now, you've seen that the process of building and training a supervised learning model is an iterative one. Your first model rarely performs the best! There are multiple ways we can potentially improve model performance. Thus far, most of the techniques we've used have been focused on our data. We can get better data, or more data, or both. We can engineer certain features, or clean up the data by removing rows/variables that hurt model performance, like multicollinearity.

The other major way to potentially improve model performance is to find good parameters to set when creating the model. For example, if we allow a decision tree to have too many leaves, the model will almost certainly overfit the data. Too few, and the model will underfit. However, each modeling problem is unique -- the same parameters could cause either of those situations, depending on the data, the task at hand, and the complexity of the model needed to best fit the data.

In this lesson, we'll learn how we can use a **_combinatorial grid search_** to find the best combination of parameters for a given model.

## Grid search

When we set parameters in a model, the parameters are not independent of one another -- the value set for one parameter can have significant effects on other parameters, thereby affecting overall model performance. Consider the following grid.

|     Parameter     |    1    |    2       |  3  |  4  |
|:-----------------:|:------:|:---------:|:--:|:--:|
|     criterion     | "gini" | "entropy" |      |
|     max_depth     |    1  |     2     |  5 |  10 |
| min_samples_split |    1   |     5     | 10 | 20 |

All the parameters above work together to create the framework of the decision tree that will be trained. For a given problem, it may be the case that increasing the value of the parameter for `min_samples_split` generally improves model performance up to a certain point, by reducing overfitting. However, if the value for `max_depth` is too low or too high, this may doom the model to overfitting or underfitting, by having a tree with too many arbitrary levels and splits that overfit on noise, or limiting the model to nothing more than a "stump" by only allowing it to grow to one or two levels.

So how do we know which combination of parameters is best? The only way we can really know for sure is to try **_every single combination!_** For this reason, grid search is sometimes referred to as an **_exhaustive search_**.


## Use `GridSearchCV`

The `sklearn` library provides an easy way to tune model parameters through an exhaustive search by using its [`GridSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) class, which can be found inside the `model_selection` module. `GridsearchCV` combines **_K-Fold Cross-Validation_** with a grid search of parameters. In order to do this, we must first create a **_parameter grid_** that tells `sklearn` which parameters to tune, and which values to try for each of those parameters.

The following code snippet demonstrates how to use `GridSearchCV` to perform a parameter grid search using a sample parameter grid, `param_grid`. Our parameter grid should be a dictionary, where the keys are the parameter names, and the values are the different parameter values we want to use in our grid search for each given key. After creating the dictionary, all you need to do is pass it to `GridSearchCV()` along with the classifier. You can also use K-fold cross-validation during this process, by specifying the `cv` parameter. In this case, we choose to use 3-fold cross-validation for each model created inside our grid search.

```python
clf = DecisionTreeClassifier()

param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [1, 2, 5, 10],
    'min_samples_split': [1, 5, 10, 20]
}

gs_tree = GridSearchCV(clf, param_grid, cv=3)
gs_tree.fit(train_data, train_labels)

gs_tree.best_params_
```


This code will run all combinations of the parameters above. The first model to be trained would be `DecisionTreeClassifier(criterion='gini', max_depth=1, min_samples_split=1)` using a 3-fold cross-validation, and recording the average score. Then, it will change one parameter, and repeat the process (e.g., `DecisionTreeClassifier(criterion='gini', max_depth=1, min_samples_split=5)`, and so on), keeping track of the overall performance of each model. Once it has tried every combination, the `GridSearchCV` object we created will automatically default the model that had the best score. We can even access the best combination of parameters by checking the `best_params_` attribute!


## Drawbacks of `GridSearchCV`

GridSearchCV is a great tool for finding the best combination of parameters. However, it is only as good as the parameters we put in our parameter grid -- so we need to be very thoughtful during this step!

The main drawback of an exhaustive search such as `GridsearchCV` is that there is no way of telling what's best until we've exhausted all possibilities! This means training many versions of the same machine learning model, which can be very time consuming and computationally expensive. Consider the example code above -- we have three different parameters, with 2, 4, and 4 variations to try, respectively. We also set the model to use cross-validation with a value of 3, meaning that each model will be built 3 times, and their performances averaged together. If we do some simple math, we can see that this simple grid search we see above actually results in `2 * 4 * 4 * 3 =` **_96 different models trained!_** For projects that involve complex models and/or very large datasets, the time needed to run a grid search can often be prohibitive. For this reason, be very thoughtful about the parameters you set -- sometimes the extra runtime isn't worth it -- especially when there's no guarantee that the model performance will improve!

## Summary

In this lesson, you learned about grid search, how to perform grid search, and the drawbacks associated with the method!


-----File-Boundary-----
# Introduction to Pipelines

## Introduction

You've learned a substantial number of different supervised learning algorithms. Now, it's time to learn about a handy tool used to integrate these algorithms into a single manageable pipeline.

## Objectives

You will be able to:

- Explain how pipelines can be used to combine various parts of a machine learning workflow

## Why Use Pipelines?

Pipelines are extremely useful tools to write clean and manageable code for machine learning. Recall how we start preparing our dataset: we want to clean our data, transform it, potentially use feature selection, and then run a machine learning algorithm. Using pipelines, you can do all these steps in one go!

Pipeline functionality can be found in scikit-learn's `Pipeline` module. Pipelines can be coded in a very simple way:

```python
from sklearn.pipeline import Pipeline

# Create the pipeline
pipe = Pipeline([('mms', MinMaxScaler()),
                 ('tree', DecisionTreeClassifier(random_state=123))])
```

This pipeline will ensure that first we'll apply a Min-Max scaler on our data before fitting a decision tree. However, the `Pipeline()` function above is only defining the sequence of actions to perform. In order to actually fit the model, you need to call the `.fit()` method like so:

```python
# Fit to the training data
pipe.fit(X_train, y_train)
```

Then, to score the model on test data, you can call the `.score()` method like so:

```python
# Calculate the score on test data
pipe.score(X_test, y_test)
```

A really good blog post on the basic ideas of pipelines can be found [here](https://www.kdnuggets.com/2017/12/managing-machine-learning-workflows-scikit-learn-pipelines-part-1.html).


## Integrating Grid Search in Pipelines

Note that the above pipeline simply creates one pipeline for a training set, and evaluates on a test set. Is it possible to create a pipeline that performs grid search? And cross-validation? Yes, it is!

First, you define the pipeline in the same way as above. Next, you create a parameter grid. When this is all done, you use the function `GridSearchCV()`, which you've seen before, and specify the pipeline as the estimator and the parameter grid. You also have to define how many folds you'll use in your cross-validation.

```python
# Create the pipeline
pipe = Pipeline([('mms', MinMaxScaler()),
                 ('tree', DecisionTreeClassifier(random_state=123))])

# Create the grid parameter
grid = [{'tree__max_depth': [None, 2, 6, 10],
         'tree__min_samples_split': [5, 10]}]


# Create the grid, with "pipe" as the estimator
gridsearch = GridSearchCV(estimator=pipe,
                          param_grid=grid,
                          scoring='accuracy',
                          cv=5)

# Fit using grid search
gridsearch.fit(X_train, y_train)

# Calculate the test score
gridsearch.score(X_test, y_test)
```

An article with a detailed workflow can be found [here](https://www.kdnuggets.com/2018/01/managing-machine-learning-workflows-scikit-learn-pipelines-part-2.html).

## Summary

Great, this wasn't too difficult! The proof of all this is in the pudding. In the next lab, you'll use this workflow to build pipelines applying classification algorithms you have learned so far in this module.


-----File-Boundary-----
# Pipelines in scikit-learn - Lab

## Introduction

In this lab, you will work with the [Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality). The goal of this lab is not to teach you a new classifier or even show you how to improve the performance of your existing model, but rather to help you streamline your machine learning workflows using scikit-learn pipelines. Pipelines let you keep your preprocessing and model building steps together, thus simplifying your cognitive load. You will see for yourself why pipelines are great by building the same KNN model twice in different ways.

## Objectives

- Construct pipelines in scikit-learn
- Use pipelines in combination with `GridSearchCV()`

## Import the data

Run the following cell to import all the necessary classes, functions, and packages you need for this lab.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings('ignore')
```
Import the `'winequality-red.csv'` dataset and print the first five rows of the data.

```python
# Import the data
df = None


# Print the first five rows
```
Use the `.describe()` method to print the summary stats of all columns in `df`. Pay close attention to the range (min and max values) of all columns. What do you notice?

```python
# Print the summary stats of all columns
```
As you can see from the data, not all features are on the same scale. Since we will be using k-nearest neighbors, which uses the distance between features to classify points, we need to bring all these features to the same scale. This can be done using standardization.



However, before standardizing the data, let's split it into training and test sets.

> Note: You should always split the data before applying any scaling/preprocessing techniques in order to avoid data leakage. If you don't recall why this is necessary, you should refer to the **KNN with scikit-learn - Lab.**

## Split the data

- Assign the target (`'quality'` column) to `y`
- Drop this column and assign all the predictors to `X`
- Split `X` and `y` into 75/25 training and test sets. Set `random_state` to 42

```python
# Split the predictor and target variables
y = None
X = None

# Split into training and test sets
X_train, X_test, y_train, y_test = None
```
## Standardize your data

- Instantiate a `StandardScaler()`
- Transform and fit the training data
- Transform the test data

```python
# Instantiate StandardScaler
scaler = None

# Transform the training and test sets
scaled_data_train = None
scaled_data_test = None

# Convert into a DataFrame
scaled_df_train = pd.DataFrame(scaled_data_train, columns=X_train.columns)
scaled_df_train.head()
```
## Train a model

- Instantiate a `KNeighborsClassifier()`
- Fit the classifier to the scaled training data

```python
# Instantiate KNeighborsClassifier
clf = None

# Fit the classifier
```
Use the classifier's `.score()` method to calculate the accuracy on the test set (use the scaled test data)

```python
# Print the accuracy on test set
```
Nicely done. This pattern (preprocessing and fitting models) is very common. Although this process is fairly straightforward once you get the hang of it, **pipelines** make this process simpler, intuitive, and less error-prone.

Instead of standardizing and fitting the model separately, you can do this in one step using `sklearn`'s `Pipeline()`. A pipeline takes in any number of preprocessing steps, each with `.fit()` and `transform()` methods (like `StandardScaler()` above), and a final step with a `.fit()` method (an estimator like `KNeighborsClassifier()`). The pipeline then sequentially applies the preprocessing steps and finally fits the model. Do this now.

## Build a pipeline (I)

Build a pipeline with two steps:

- First step: `StandardScaler()`
- Second step (estimator): `KNeighborsClassifier()`

```python
# Build a pipeline with StandardScaler and KNeighborsClassifier
scaled_pipeline_1 = None
```
- Transform and fit the model using this pipeline to the training data (you should use `X_train` here)
- Print the accuracy of the model on the test set (you should use `X_test` here)

```python
# Fit the training data to pipeline


# Print the accuracy on test set
```
If you did everything right, this answer should match the one from above!

Of course, you can also perform a grid search to determine which combination of hyperparameters can be used to build the best possible model. The way you define the pipeline still remains the same. What you need to do next is define the grid and then use `GridSearchCV()`. Let's do this now.

## Build a pipeline (II)

Again, build a pipeline with two steps:

- First step: `StandardScaler()`
- Second step (estimator): `RandomForestClassifier()`. Set `random_state=123` when instantiating the random forest classifier

```python
# Build a pipeline with StandardScaler and RandomForestClassifier
scaled_pipeline_2 = None
```
Use the defined `grid` to perform a grid search. We limited the hyperparameters and possible values to only a few values in order to limit the runtime.

```python
# Define the grid
grid = [{'RF__max_depth': [4, 5, 6],
         'RF__min_samples_split': [2, 5, 10],
         'RF__min_samples_leaf': [1, 3, 5]}]
```
Define a grid search now. Use:
- the pipeline you defined above (`scaled_pipeline_2`) as the estimator
- the parameter `grid`
- `'accuracy'` to evaluate the score
- 5-fold cross-validation

```python
# Define a grid search
gridsearch = None
```
After defining the grid values and the grid search criteria, all that is left to do is fit the model to training data and then score the test set. Do this below:

```python
# Fit the training data


# Print the accuracy on test set
```
## Summary

See how easy it is to define pipelines? Pipelines keep your preprocessing steps and models together, thus making your life easier. You can apply multiple preprocessing steps before fitting a model in a pipeline. You can even include dimensionality reduction techniques such as PCA in your pipelines. In a later section, you will work on this too!


-----File-Boundary-----
# Refactoring Your Code to Use Pipelines

## Introduction

In this lesson, you will learn how to use the core features of scikit-learn pipelines to refactor existing machine learning preprocessing code into a portable pipeline format.

## Objectives

You will be able to:

* Recall the benefits of using pipelines
* Describe the difference between a `Pipeline`, a `FeatureUnion`, and a `ColumnTransformer` in scikit-learn
* Iteratively refactor existing preprocessing code into a pipeline

## Pipelines in the Data Science Process

***If my code already works, why do I need a pipeline?***

As we covered previously, pipelines are a great way to organize your code in a DRY (don't repeat yourself) fashion. It also allows you to perform cross validation (including `GridSearchCV`) in a way that avoids leakage, because you are performing all preprocessing steps separately. Finally, it's helpful if you want to deploy your code, since it means that you only need to pickle the overall pipeline, rather than pickling the fitted model as well as all of the fitted preprocessing transformers.

***Then why not just write a pipeline from the start?***

Pipelines are designed for efficiency rather than readability, so they can become very confusing very quickly if something goes wrong. (All of the data is in NumPy arrays, not pandas dataframes, so there are no column labels by default.)

Therefore it's a good idea to write most of your code without pipelines at first, then refactor it. Eventually if you are very confident with pipelines you can save time by writing them from the start, but it's okay if you stick with the refactoring strategy!

## Code without Pipelines

Let's say we have the following (very-simple) dataset:

```python
import pandas as pd

example_data = pd.DataFrame([
    {"category": "A", "number": 7, "target": 1},
    {"category": "A", "number": 8, "target": 1},
    {"category": "B", "number": 9, "target": 0},
    {"category": "B", "number": 7, "target": 1},
    {"category": "C", "number": 4, "target": 0}
])

example_X = example_data.drop("target", axis=1)
example_y = example_data["target"]

example_X
```
### Preprocessing Steps without Pipelines

These steps should be a review of preprocessing steps you have learned previously.

#### One-Hot Encoding Categorical Data

If we just tried to apply a `StandardScaler` then a `LogisticRegression` to this dataset, we would get a `ValueError` because the values in `category` are not yet numeric.

So, let's use a `OneHotEncoder` to convert the `category` column into multiple dummy columns representing each of the categories present:

```python
from sklearn.preprocessing import OneHotEncoder

# Make a transformer
ohe = OneHotEncoder(categories="auto", handle_unknown="ignore", sparse=False)

# Create transformed dataframe
category_encoded = ohe.fit_transform(example_X[["category"]])
category_encoded = pd.DataFrame(
    category_encoded,
    columns=ohe.categories_[0],
    index=example_X.index
)

# Replace categorical data with encoded data
example_X.drop("category", axis=1, inplace=True)
example_X = pd.concat([category_encoded, example_X], axis=1)

# Visually inspect dataframe
example_X
```
#### Feature Engineering

Let's say for the sake of example that we wanted to add a new feature called `number_odd`, which is `1` when the value of `number` is odd and `0` when the value of `number` is even. (It's not clear why this would be useful, but you can imagine a more realistic example, e.g. a boolean flag related to a purchase threshold that triggers free shipping.)

We don't want to remove `number` and replace it with `number_odd`, we want an entire new feature `number_odd` to be added.

Let's make a custom transformer for this purpose. Specifically, we'll use a `FunctionTransformer` ([documentation here](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html)). As you might have guessed from the name, a `FunctionTransformer` takes in a function as an argument (similar to the `.apply` dataframe method) and uses that function to transform the data. Unlike just using `.apply`, this transformer has the typical `.fit_transform` interface and can be used just like any other transformer (including being used in a pipeline).

```python
from sklearn.preprocessing import FunctionTransformer

def is_odd(data):
    """
    Helper function that returns 1 if odd, 0 if even
    """
    return data % 2

# Instantiate transformer
func_transformer = FunctionTransformer(is_odd)

# Create transformed column
number_odd = func_transformer.fit_transform(example_X["number"])

# Add engineered column
example_X["number_odd"] = number_odd
example_X
```
#### Scaling

Then let's say we want to scale all of the features after the previous steps have been taken:

```python
from sklearn.preprocessing import StandardScaler

# Instantiate transformer
scaler = StandardScaler()

# Create transformed dataset
data_scaled = scaler.fit_transform(example_X)

# Replace dataset with transformed one
example_X = pd.DataFrame(
    data_scaled,
    columns=example_X.columns,
    index=example_X.index
)
example_X
```
#### Bringing It All Together

Here is the full preprocessing example without a pipeline:

```python
def preprocess_data_without_pipeline(X):

    transformers = []

    ### Encoding categorical data ###

    # Make a transformer
    ohe = OneHotEncoder(categories="auto", handle_unknown="ignore", sparse=False)

    # Create transformed dataframe
    category_encoded = ohe.fit_transform(X[["category"]])
    category_encoded = pd.DataFrame(
        category_encoded,
        columns=ohe.categories_[0],
        index=X.index
    )
    transformers.append(ohe)

    # Replace categorical data with encoded data
    X.drop("category", axis=1, inplace=True)
    X = pd.concat([category_encoded, X], axis=1)

    ### Feature engineering ###

    def is_odd(data):
        """
        Helper function that returns 1 if odd, 0 if even
        """
        return data % 2

    # Instantiate transformer
    func_transformer = FunctionTransformer(is_odd)

    # Create transformed column
    number_odd = func_transformer.fit_transform(X["number"])
    transformers.append(func_transformer)

    # Add engineered column
    X["number_odd"] = number_odd

    ### Scaling ###

    # Instantiate transformer
    scaler = StandardScaler()

    # Create transformed dataset
    data_scaled = scaler.fit_transform(X)
    transformers.append(scaler)

    # Replace dataset with transformed one
    X = pd.DataFrame(
        data_scaled,
        columns=X.columns,
        index=X.index
    )

    return X, transformers

# Reset value of example_X
example_X = example_data.drop("target", axis=1)
# Test out our function
result, transformers = preprocess_data_without_pipeline(example_X)
result
```
Now let's rewrite that with pipeline logic!

## Pieces of a Pipeline

### `Pipeline` Class

In a previous lesson, we introduced the most fundamental part of pipelines: the `Pipeline` class. This class is useful if you want to perform the same steps on every single column in your dataset. A simple example of just using a `Pipeline` would be:

```python
pipe = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])
```

However, many interesting datasets contain a mixture of kinds of data (e.g. numeric and categorical data), which means you often do not want to perform the same steps on every column. For example, one-hot encoding is useful for converting categorical data into a format that is usable in ML models, but one-hot encoding numeric data is a bad idea. You also usually want to apply different feature engineering processes to different features.

In order to apply different data cleaning and feature engineering steps to different columns, we'll use the `FeatureUnion` and `ColumnTransformer` classes.

### `ColumnTransformer` Class

The core idea of a `ColumnTransformer` is that you can **apply different preprocessing steps to different columns of the dataset**.

Looking at the preprocessing steps above, we only want to apply the `OneHotEncoder` to the `category` column, so this is a good use case for a `ColumnTransformer`:

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Reset value of example_X
example_X = example_data.drop("target", axis=1)

# Create a column transformer
col_transformer = ColumnTransformer(transformers=[
    ("ohe", OneHotEncoder(categories="auto", handle_unknown="ignore"), ["category"])
], remainder="passthrough")

# Create a pipeline containing the single column transformer
pipe = Pipeline(steps=[
    ("col_transformer", col_transformer)
])

# Use the pipeline to fit and transform the data
transformed_data = pipe.fit_transform(example_X)
transformed_data
```
The pipeline returns a NumPy array, but we can convert it back into a dataframe for readability if we want to:

```python
import numpy as np

# Extract the category labels from the OHE within the pipeline
encoder = col_transformer.named_transformers_["ohe"]
category_labels = encoder.categories_[0]

# Make a dataframe with the relevant columns
pd.DataFrame(transformed_data, columns=np.append(category_labels, "number"))
```
#### Interpreting the `ColumnTransformer` Example

Let's go back and look at each of those steps more closely.

First, creating a column transformer. Here is what that code looked like above:

```python
# Create a column transformer
col_transformer = ColumnTransformer(transformers=[
    ("ohe", OneHotEncoder(categories="auto", handle_unknown="ignore"), ["category"])
], remainder="passthrough")
```

Here is the same code, spread out so we can add more comments explaining what's happening:

```python
# Create a column transformer
col_transformer = ColumnTransformer(
    # ColumnTransformer takes a list of "transformers", each of which
    # is represented by a three-tuple (not just a transformer object)
    transformers=[
        # Each tuple has three parts
        (
            # (1) This is a string representing the name. It is there
            # for readability and so you can extract information from
            # the pipeline later. scikit-learn doesn't actually care
            # what the name is.
            "ohe",
            # (2) This is the actual transformer
            OneHotEncoder(categories="auto", handle_unknown="ignore"),
            # (3) This is the list of columns that the transformer should
            # apply to. In this case, there is only one column, but it
            # still needs to be in a list
            ["category"]
        )
        # If we wanted to perform multiple different transformations
        # on different columns, we would add more tuples here
    ],
    # By default, any column that is not specified in the list of
    # transformer tuples will be dropped, but we can indicate that we
    # want them to stay as-is if we set remainder="passthrough"
    remainder="passthrough"
)
```

Next, putting the column transformer into a pipeline. Here is that original code:

```python
# Create a pipeline containing the single column transformer
pipe = Pipeline(steps=[
    ("col_transformer", col_transformer)
])
```

And again, here it is with more comments:

```python
# Create a pipeline containing the single column transformer
pipe = Pipeline(
    # Pipeline takes a list of "steps", each of which is
    # represented by a two-tuple (not just a transformer or
    # estimator object)
    steps=[
        # Each tuple has two parts
        (
            # (1) This is name of the step. Again, this is for
            # readability and retrieving steps later, so just
            # choose a name that makes sense to you
            "col_transformer",
            # (2) This is the actual transformer or estimator.
            # Note that a transformer can be a simple one like
            # StandardScaler, or a composite one like a
            # ColumnTransformer (shown here), a FeatureUnion,
            # or another Pipeline.
            # Typically the last step will be an estimator
            # (i.e. a model that makes predictions)
            col_transformer
        )
    ]
)
```

### `FeatureUnion` Class

A `FeatureUnion` **concatenates together the results of multiple different transformers**. While `Pipeline` and a `ColumnTransformer` are usually enough to perform basic *data cleaning* forms of preprocessing, it's also helpful to be able to use a `FeatureUnion` for *feature engineering* forms of preprocessing.

Let's use a `FeatureUnion` to add on the `number_odd` feature from before. Because we only want this transformation to apply to the `number` column, we need to wrap it in a `ColumnTransformer` again. Let's call this new one `feature_eng` to indicate what it is doing:

```python
# Create a ColumnTransformer for feature engineering
feature_eng = ColumnTransformer(transformers=[
    ("add_number_odd", FunctionTransformer(is_odd), ["number"])
], remainder="drop")
```
Let's also rename the other `ColumnTransformer` to `original_features_encoded` to make it clearer what it is responsible for:

```python
# Create a ColumnTransformer to encode categorical data
# and keep numeric data as-is
original_features_encoded = ColumnTransformer(transformers=[
    ("ohe", OneHotEncoder(categories="auto", handle_unknown="ignore"), ["category"])
], remainder="passthrough")
```
Now we can combine those two into a `FeatureUnion`:

```python
from sklearn.pipeline import FeatureUnion

feature_union = FeatureUnion(transformer_list=[
    ("encoded_features", original_features_encoded),
    ("engineered_features", feature_eng)
])
```
And put that `FeatureUnion` into a `Pipeline`:

```python
# Create a pipeline containing union of encoded
# original features and engineered features
pipe = Pipeline(steps=[
    ("feature_union", feature_union)
])

# Use the pipeline to fit and transform the data
transformed_data = pipe.fit_transform(example_X)
transformed_data
```
Again, here it is as a more-readable dataframe:

```python
# Extract the category labels from the OHE within the pipeline
encoder = original_features_encoded.named_transformers_["ohe"]
category_labels = encoder.categories_[0]

# Make a dataframe with the relevant columns
all_cols = list(category_labels) + ["number", "number_odd"]
pd.DataFrame(transformed_data, columns=all_cols)
```
#### Interpreting the `FeatureUnion` Example

Once more, here was the code used to create the `FeatureUnion`:

```python
feature_union = FeatureUnion(transformer_list=[
    ("encoded_features", original_features_encoded),
    ("engineered_features", feature_eng)
])
```

And here it is spread out with more comments:

```python
feature_union = FeatureUnion(
    # FeatureUnion takes a "transformer_list" containing
    # two-tuples (not just transformers)
    transformer_list=[
        # Each tuple contains two elements
        (
            # (1) Name of the feature. If you make this "drop",
            # the transformer will be ignored
            "encoded_features",
            # (2) The actual transformer (in this case, a
            # ColumnTransformer). This one will produce the
            # numeric features as-is and the categorical
            # features one-hot encoded
            original_features_encoded
        ),
        # Here is another tuple
        (
            # (1) Name of the feature
            "engineered_features",
            # (2) The actual transformer (again, a
            # ColumnTransformer). This one will produce just
            # the flag of whether the number is even or odd
            feature_eng
        )
    ]
)
```

### Adding Final Steps to Pipeline

If we want to scale all of the features at the end, this doesn't require any additional `ColumnTransformer` or `FeatureUnion` objects, it just means we need to add another step in our `Pipeline` like this:

```python
# Create a pipeline containing union of encoded
# original features and engineered features, then
# all features scaled
pipe = Pipeline(steps=[
    ("feature_union", feature_union),
    ("scale", StandardScaler())
])

# Use the pipeline to fit and transform the data
transformed_data = pipe.fit_transform(example_X)
transformed_data
```
Additionally, if we want to add an estimator (model) as the last step, we can do it like this:

```python
from sklearn.linear_model import LogisticRegression

# Create a pipeline containing union of encoded
# original features and engineered features, then
# all features scaled, then feed into a model
pipe = Pipeline(steps=[
    ("feature_union", feature_union),
    ("scale", StandardScaler()),
    ("model", LogisticRegression())
])

# Use the pipeline to fit the model and score it
pipe.fit(example_X, example_y)
pipe.score(example_X, example_y)
```
## Complete Refactored Pipeline Example

Below is the complete pipeline (without the estimator), which produces the same output as the original full preprocessing example:

```python
def preprocess_data_with_pipeline(X):

    ### Encoding categorical data ###
    original_features_encoded = ColumnTransformer(transformers=[
        ("ohe", OneHotEncoder(categories="auto", handle_unknown="ignore"), ["category"])
    ], remainder="passthrough")

    ### Feature engineering ###
    def is_odd(data):
        """
        Helper function that returns 1 if odd, 0 if even
        """
        return data % 2

    feature_eng = ColumnTransformer(transformers=[
        ("add_number_odd", FunctionTransformer(is_odd), ["number"])
    ], remainder="drop")

    ### Combine encoded and engineered features ###
    feature_union = FeatureUnion(transformer_list=[
        ("encoded_features", original_features_encoded),
        ("engineered_features", feature_eng)
    ])

    ### Pipeline (including scaling) ###
    pipe = Pipeline(steps=[
        ("feature_union", feature_union),
        ("scale", StandardScaler())
    ])

    transformed_data = pipe.fit_transform(X)

    ### Re-apply labels (optional step for readability) ###
    encoder = original_features_encoded.named_transformers_["ohe"]
    category_labels = encoder.categories_[0]
    all_cols = list(category_labels) + ["number", "number_odd"]
    return pd.DataFrame(transformed_data, columns=all_cols, index=X.index), pipe

# Reset value of example_X
example_X = example_data.drop("target", axis=1)
# Test out our new function
result, pipe = preprocess_data_with_pipeline(example_X)
result
```
Just to confirm, this produces the same result as the previous function:

```python
# Reset value of example_X
example_X = example_data.drop("target", axis=1)
# Compare result to old function
result, transformers = preprocess_data_without_pipeline(example_X)
result
```
We achieved the same thing in fewer lines of code, better prevention of leakage, and the ability to pickle the whole process!

Note that in both cases we returned the object or objects used for preprocessing so they could be used on test data. Without a pipeline, we would need to apply each of the transformers in `transformers`. With a pipeline, we would just need to use `pipe.transform` on test data.

## Summary

In this lesson, you learned how to make more-sophisticated pipelines using `ColumnTransformer` and `FeatureUnion` objects in addition to `Pipeline`s. We started with a preprocessing example that used sckit-learn code without pipelines, and rewrote it to use pipelines. Along the way we used `ColumnTransformer` to conditionally preprocess certain columns while leaving others alone, and `FeatureUnion` to combine engineered features with preprocessed versions of the original data. Now you should have a clearer idea of how pipelines can be used for non-trivial preprocessing tasks.


-----File-Boundary-----
# Pickle

## Introduction

Pickle is an invaluable tool for saving objects.  In this lesson you will learn how to use it on various different Python data types.

## Objectives

You will be able to:

* Describe the circumstances in which you would want to use a pickle file
* Write a pickle file
* Read a pickle file
* Use the `joblib` library to pickle and load a scikit-learn class

## Data Serialization

Think about the importance of being able to save data files to CSV, or another format. For example, you start with a raw dataset which you may have downloaded from the web. Then you painstakingly take hours preprocessing the data, cleaning it, constructing features, aggregates, and other views. In order to avoid having to rerun your entire process, you are apt to save the current final cleaned version of the dataset into a serialized format like CSV.

`pickle` allows you to go beyond just storing data in a format like CSV, and save any object that is currently loaded into your Python interpreter. Literally anything. You could save data stored in a `dict`, `list`, or `set` as a pickle file. You can also save functions or class instances as pickle files. Saving models is one of the important use cases of this technique.

## Pickling Base Python Objects

Let's say we have this nested data structure example (from the [Python docs](https://docs.python.org/3/library/pickle.html#examples)), which would not be suitable for storage as a CSV because the values are different data types:

```python
data_object = {
    'a': [1, 2.0, 3, 4+6j],
    'b': ('character string', b'byte string'),
    'c': {None, True, False}
}
```
### Importing `pickle`

`pickle` is a module built in to Python that is suitable for pickling base Python objects. You can import it like this:

```python
import pickle
```
### Writing Objects to Pickle

Let's store this object as a file on disk called `data.pickle`.

1. Open a file called `'data.pickle'`.
   1. The `.pickle` file extension is conventional for Python 3 objects. You can use a different file name or extension and it won't make a difference as far as Python is concerned, but we recommend using `.pickle` so it's clear what the file is
2. We'll need to open the file using mode `'wb'`.
   1. `w` because we want to write data to the file (and automatically create the file if it doesn't exist yet)
   2. `b` because we specifically want to write binary data. `pickle` uses a binary protocol, so it won't work if you don't include the `b`
3. Then we can use the `dump` function from the `pickle` module to write our data object.

```python
with open('data.pickle', 'wb') as f:
    pickle.dump(data_object, f)
```
### Importing Objects from Pickle Files

Go ahead and restart the kernel. Now if we try to access the original `data_object` it won't work:

```python
try:
    print(data_object)
except NameError as e:
    print(type(e), e)
```
But we can use `pickle` to load it back into memory with a new name, `data_object2`.

1. Open the file `data.pickle` since that is the file name we saved prior to restarting the kernel.
2. We'll need to open the file using mode `'rb'`.
   1. `r` for read
   2. `b` for binary
2. Then we can use the `load` function from the `pickle` module to read our data object.

```python
import pickle
with open('data.pickle', 'rb') as f:
    data_object2 = pickle.load(f)
data_object2
```
***Important reminder:*** DO NOT open pickle files unless you trust the source (e.g. you created them yourself). They can contain malicious code and there are not any built-in security constraints on them.

## Pickle with scikit-learn

So far, your process has typically been to instantiate a model, train it, evaluate it, maybe make some predictions, then shut down the notebook. This means that the time and computational resources used to train the model are lost, and would need to be repeated if you ever wanted to use the model again.

If you pickle your fitted model instead, then all you will need to do is load it back into memory, then it will be all ready to make predictions!

### Instantiating and Fitting a Model

Below we fit a simple linear regression model:

```python
from sklearn.linear_model import LinearRegression

# y = x + 1
X = [[1],[2],[3],[4],[5]]
y = [2, 3, 4, 5, 6]

model = LinearRegression()
model.fit(X, y)

print(f"Fitted model is y = {model.coef_[0]}x + {model.intercept_}")
```
We can now use the model to make predictions:

```python
model.predict([[7], [8], [9]])
```
### Importing `joblib`

For scikit-learn models, it is possible to use the `pickle` module but not recommended because it is less efficient. (See documentation [here](https://scikit-learn.org/stable/modules/model_persistence.html).) Instead, we'll use the `joblib` library.

(Note that we still often use the language of "pickle file" and "pickling" even if we are using a different library such as `joblib`.)

```python
import joblib
```
### Writing Objects with `joblib`

Let's save `model` using `joblib`.

1. Once again we need to open a file to store the model in.
   1. Instead of `'data'`, we'll call this `'regression_model'` so it's clear what the file contains
   2. Instead of the `.pickle` file extension, we'll use `.pkl`
      1. This is the conventional file ending for scikit-learn models, and used to be the standard for all Python objects in Python 2
      2. Neither Python nor `joblib` nor scikit-learn will enforce this file ending. So you also might see examples with `.pickle` or `.joblib` file extensions, or some other ending, even though it was serialized using this technique
      3. If you see a serialized model ending with `.zip`, `.pb`, or `.h5`, that means it is likely not a scikit-learn model and probably was not serialized using `joblib` or `pickle`
2. The mode (`'wb'`) is the same as with `pickle`.
3. The function name (`dump`) is also the same as with `pickle`.

```python
with open('regression_model.pkl', 'wb') as f:
    joblib.dump(model, f)
```
### Importing Objects with `joblib`

Now, restart the kernel. Once again, we would expect an error if we tried to use the `model` variable:

```python
try:
    print(model.predict([[10], [11], [12]]))
except NameError as e:
    print(type(e), e)
```
But we can load the model from the pickled file:

1. Open file `'regression_model.pkl'`.
2. Use mode `'rb'`.
3. Use the `load` function (this time from `joblib`).

```python
import joblib
with open('regression_model.pkl', 'rb') as f:
    model2 = joblib.load(f)

print(f"Loaded model is y = {model2.coef_[0]}x + {model2.intercept_}")
```
Note that the coefficient and intercept are the same as the original model. While this would have been a simple model to re-fit, you can imagine how this would save significant time with a more-complex model or with large production datasets.

Now we can make predictions again!

```python
model2.predict([[10], [11], [12]])
```
## Additional Resources

* [Pickle Documentation](https://docs.python.org/3/library/pickle.html)
* [scikit-learn Persistence Documentation (using `joblib`)](https://scikit-learn.org/stable/modules/model_persistence.html)

## Summary

In this brief lesson you saw how to both save objects to pickle files and import objects from pickle files. This can be particularly useful for saving models that are non deterministic and would otherwise be difficult or impossible to reproduce exact replicas of.


-----File-Boundary-----
# Pickling and Deploying Pipelines

## Introduction

Now that you have learned about scikit-learn pipelines and model pickling, it's time to bring it all together in a professional ML workflow!

## Objectives

In this lesson you will:

* Understand the purpose of deploying a machine learning model
* Understand the cloud function approach to model deployment
* Pickle a scikit-learn pipeline
* Create a cloud function

## Model Deployment

Previously when we covered pickling, we introduced the idea of model ***persistence*** -- essentially that if you serialize a model after training it, you can later load the model and make predictions without wasting time or computational resources re-training the model.

In some contexts, model persistence is all you need. For example, if the end-user of your model is a data scientist with a full Python environment setup, they can launch up their own notebook, call `.load` on the model file, and start making predictions.

Model ***deployment*** goes beyond model persistence to allow your model to be used in many more contexts. Here are just a few examples:

* A mobile app that uses a model to power its game AI
* A website that uses a model to decide which ads to serve to a given viewer
* A CRM (customer relationship management) platform that uses a model to decide when to send out coupons

In order for these applications to work, your model needs to use a deployment strategy that is more complex than simply pickling a Python model.

### Model Deployment with Cloud Functions

A cloud function is a very popular way to deploy a machine learning model. When you deploy a model as a cloud function, that means that you are setting up an ***HTTP API backend***. This is the same kind of API that you have previously queried using the `requests` library.

The advantage of a cloud function approach is that the cloud provider handles the actual web server maintenance underpinning the API, and you just have to provide a small amount of function code. If you need a lot of customization, you may need to write and deploy an actual web server, but for many use cases you can just use the cloud function defaults.

When a model is deployed as an API, that means that **it can be queried from any language** that can use HTTP APIs. This means that even though you wrote the model in Python, an app written in Objective C or a website written in JavaScript can use it!

## Creating a Cloud Function

Let's go ahead and create one! We are going to use the format required by Google Cloud Functions ([documentation here](https://cloud.google.com/functions)).

This is by no means the only platform for hosting a cloud function -- feel free to investigate AWS Lambda or Azure Functions or other cloud provider options! However Google Cloud Functions are a convenient option because their free tier allows up to 2 million API calls per day, and they promise not to charge for additional API calls unless given authorization. This is useful when you're learning how to deploy models and don't want to spend money on cloud services.

### Components of a Cloud Function for Model Deployment

In order to deploy a model, you will need:

1. A pickled model file
2. A Python file defining the function
3. A requirements file

### Our Pipeline

Let's say the model we have developed is a multi-class classifier trained on the [iris dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html#sklearn.datasets.load_iris) from scikit-learn.

```python
import pandas as pd
from sklearn.datasets import load_iris

data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="class")
pd.concat([X, y], axis=1)
```
```python
X.describe()
```
And let's say we are using logistic regression, with default regularization applied.

As you can see, even with such a simple dataset, we need to scale the data before passing it into the classifier, since the different features do not currently have the same scale. Let's go ahead and use a pipeline for that:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])

pipe.fit(X, y)
```
Now the pipeline is ready to make predictions on new data!

In the example below, we are sending in X values from a record in the training data. We know that the classification *should* be 0, so this is a quick check to make sure that the model works and we are getting the results we expect:

```python
example = [[5.1, 3.5, 1.4, 0.2]]
pipe.predict(example)[0]
```
It worked!

(Note that when you call `.predict` on a pipeline, it is actually transforming the input data then making the prediction. You do not need to apply a separate `.transform` step.)

### Pickling Our Pipeline

In this case, because raw data needs to be preprocessed before our model can use it, we'll pickle the entire pipeline, not just the model.

Sometimes you will see something referred to as a "pickled model" when it's actually a pickled pipeline, so we'll use that language interchangeably.

First, let's import the `joblib` library:

```python
import joblib
```
Then we can serialize the pipeline into a file called `model.pkl`.

(Recall that `.pkl` is a conventional file ending for a pickled scikit-learn model. [This blog post](https://towardsdatascience.com/guide-to-file-formats-for-machine-learning-columnar-training-inferencing-and-the-feature-store-2e0c3d18d4f9) covers many other file formats that you might see that use different libraries.)

```python
with open("model.pkl", "wb") as f:
    joblib.dump(pipe, f)
```
That's it! Sometimes you will also see approaches that pickle the model along with its current metrics and possibly some test data (in order to confirm that the model performance is the same when un-pickled) but for now we'll stick to the most simple approach.

### Creating Our Function

The serialized model is not sufficient on its own for the HTTP API server to know what to do. You also need to write the actual Python function for it to execute.

Let's write a function and test it out. In this case, let's say the function:

* takes in 4 arguments representing the 4 features
* returns a dictionary with the format `{"predicted_class": <value>}` where `<value>` is 0, 1, or 2

```python
def iris_prediction(sepal_length, sepal_width, petal_length, petal_width):
    """
    Given sepal length, sepal width, petal length, and petal width,
    predict the class of iris
    """

    # Load the model from the file
    with open("model.pkl", "rb") as f:
        model = joblib.load(f)

    # Construct the 2D matrix of values that .predict is expecting
    X = [[sepal_length, sepal_width, petal_length, petal_width]]

    # Get a list of predictions and select only 1st
    predictions = model.predict(X)
    prediction = predictions[0]

    return {"predicted_class": prediction}
```
Now let's test it out!

```python
iris_prediction(5.1, 3.5, 1.4, 0.2)
```
The specific next steps needed to incorporate this function into a cloud function platform will vary. For Google Cloud Functions specifically, it looks like this. All of this code would need to be written in a file called `main.py`:

```python
import json
import joblib

def iris_prediction(sepal_length, sepal_width, petal_length, petal_width):
    """
    Given sepal length, sepal width, petal length, and petal width,
    predict the class of iris
    """

    # Load the model from the file
    with open("model.pkl", "rb") as f:
        model = joblib.load(f)

    # Construct the 2D matrix of values that .predict is expecting
    X = [[sepal_length, sepal_width, petal_length, petal_width]]

    # Get a list of predictions and select only 1st
    predictions = model.predict(X)
    prediction = int(predictions[0])

    return {"predicted_class": prediction}

def predict(request):
    """
    `request` is an HTTP request object that will automatically be passed
    in by Google Cloud Functions

    You can find all of its properties and methods here:
    https://flask.palletsprojects.com/en/1.0.x/api/#flask.Request
    """
    # Get the request data from the user in JSON format
    request_json = request.get_json()

    # We are expecting the request to look like this:
    # {"sepal_length": <x1>, "sepal_width": <x2>, "petal_length": <x3>, "petal_width": <x4>}
    # Send it to our prediction function using ** to unpack the arguments
    result = iris_prediction(**request_json)

    # Return the result as a string with JSON format
    return json.dumps(result)
```
(Unusually for a `.py` file, we don't need to include an `if __name__ == "__main__":` statement in the file. This is due to the particular configuration of Google Cloud Functions, which operations a web server behind the scenes. Instead, if you want to deploy the function, you'll need to configure it in the console so that the `predict` function will be invoked.)

### Creating Our Requirements File

One last thing we need before we can upload our cloud function is a requirements file. As we have developed this model, we have likely been using some massive environment like `learn-env` that includes lots of packages that we don't actually need.

Let's make a file that specifies only the packages we need. Which ones are those?

#### scikit-learn

We used scikit-learn to build our pickled model. (Technically, a model pipeline.) Let's figure out what exact version it was:

```python
import sklearn
sklearn.__version__
```
#### joblib

We also used joblib to serialize the model. We'll repeat the same step:

```python
joblib.__version__
```
#### Creating the File

At the time of this writing, we are using scikit-learn 0.23.2 and joblib 0.17.0. (If you see different numbers there, that means we have updated the environment, so use those numbers instead!) We create a file called `requirements.txt` containing these lines, with pip-style versions ([documentation here](https://pip.pypa.io/en/stable/reference/requirements-file-format/#requirements-file-format)):

```
scikit-learn==0.23.2
joblib==0.17.0
```

### Putting It All Together

Now we have:

1. `model.pkl` (a pickled model file)
2. `main.py` (a Python file defining the function)
3. `requirements.txt` (a requirements file)

Copies of each of these files are available in this repository.

If you want to deploy these on Google Cloud Functions, you'll want to combine them all into a single zipped archive. For example, to do this with the `zip` utility on Mac or Linux, run this command in the terminal:

```bash
zip archive model.pkl main.py requirements.txt
```

That will create an archive called `archive.zip` which can be uploaded following [these instructions](https://cloud.google.com/functions/docs/deploying/console). An already-created `archive.zip` is available in this repository if you just want to practice following the Google Cloud Function instructions.

You will want to specify an executed function of `predict` and also select the checkbox for "Allow unauthenticated invocations" if you want to make a public API.

Then the code to test out your deployed function would be something like this, where you replace the `url` value with your actual URL.

```python
import requests
response = requests.post(
    url="https://<name here>.cloudfunctions.net/function-1",
    json={"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
)
response
```

## Summary

In this lesson, we discussed the purpose of model deployment, and cloud functions in particular. Then we walked through the process of pickling a scikit-learn pipeline and using it in a deployed cloud function.


-----File-Boundary-----
# Model Tuning and Pipelines - Recap

## Key Takeaways

The key takeaways from this section include:

* Machine learning ***pipelines*** create a nice workflow to combine data manipulations, preprocessing, and modeling
* Machine learning pipelines can be used along with ***grid search*** to evaluate several parameter settings
  * Grid search can considerably blow up computation time when computing for several parameters along with cross-validation
  * Some models are very sensitive to hyperparameter changes, so they should be chosen with care, and even with big grids a good outcome isn't always guaranteed
* Machine learning pipelines can also be ***pickled*** so that they can be used in the future without re-training
* Model ***deployment*** can be something as simple as pickling a model, or a more complex approach like a ***cloud function*** that exposes model predictions through an HTTP API


-----File-Boundary-----
