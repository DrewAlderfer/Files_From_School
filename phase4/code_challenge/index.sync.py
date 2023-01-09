# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python (learn-env)
#     language: python
#     name: learn-env
# ---

# %% [markdown] nbgrader={"grade": false, "grade_id": "cell-93e35ef07b6a9f79", "locked": true, "schema_version": 3, "solution": false, "task": false}
# # Phase 4 Code Challenge
#
# This code challenge is designed to test your understanding of the Phase 4 material. It covers:
#
# * Principal Component Analysis
# * Clustering
# * Time Series
# * Natural Language Processing
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

# %% nbgrader={"grade": false, "grade_id": "cell-8324b5fef3a46de1", "locked": true, "schema_version": 3, "solution": false, "task": false}
# Run this cell without changes to import the necessary libraries

from numbers import Number
from random import Random
import matplotlib, sklearn, scipy, pickle
from pandas.core.common import random_state
from scipy.sparse import random
import numpy as np
import pandas as pd

# %matplotlib inline

# %% [markdown] nbgrader={"grade": false, "grade_id": "cell-0312e6ab3947bffa", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ---
#
# ## Part 1: Principal Component Analysis [Suggested Time: 15 minutes]
#
# ---
#
# In this part, you will use Principal Component Analysis on the wine dataset. 

# %% nbgrader={"grade": false, "grade_id": "cell-1c655cf7834874d7", "locked": true, "schema_version": 3, "solution": false, "task": false}
# Run this cell without changes

# Relevant imports
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load data
wine = load_wine()
X, y = load_wine(return_X_y=True)
X = pd.DataFrame(X, columns=wine.feature_names)
y = pd.Series(y)
y.name = 'class'

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Scaling
scaler_1 = StandardScaler()
X_train_scaled = pd.DataFrame(scaler_1.fit_transform(X_train), columns=X_train.columns)

# Inspect the first five rows of the scaled dataset
X_train_scaled.head()

# %% [markdown] nbgrader={"grade": false, "grade_id": "cell-adac39f3ffb2589c", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ### 1.1) Create a PCA object `wine_pca` and fit it using `X_train_scaled`.
#
# Use parameter defaults with `n_components=0.9` and `random_state=1` for your classifier. You must use the Scikit-learn PCA (docs [here](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)).

# %% nbgrader={"grade": false, "grade_id": "cell-fc96080dfc176b32", "locked": true, "schema_version": 3, "solution": false, "task": false}
# CodeGrade step1.1
# Your code here
pca_scaler = StandardScaler()

wine_pca = PCA(n_components=0.9, random_state=1)

# Fit
X_train_scaled = pca_scaler.fit_transform(X_train)
X_test_scaled = pca_scaler.transform(X_test)

wine_pca.fit_transform(X_train_scaled)
wine_pca.explained_variance_ratio_

# %%
# This test confirms that you have created a PCA object named wine_pca

assert type(wine_pca) == PCA

# This test confirms that you have set random_state to 1

assert wine_pca.get_params()['random_state'] == 1

# This test confirms that wine_pca has been fit

sklearn.utils.validation.check_is_fitted(wine_pca)

# %% [markdown] nbgrader={"grade": false, "grade_id": "cell-afa7eb5b2df5dc78", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ### 1.2) Create a numeric variable `wine_pca_ncomps` containing the number of components in `wine_pca`
#
# _Hint: Look at the list of attributes of trained `PCA` objects in the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)_

# %% nbgrader={"grade": false, "grade_id": "cell-0dc95483da95ec65", "locked": true, "schema_version": 3, "solution": false, "task": false}
# CodeGrade step1.2
# Replace None with appropriate code

wine_pca_ncomps = wine_pca.n_components_

# %%
# This test confirms that you have created a numeric variable named wine_pca_ncomps

assert isinstance(wine_pca_ncomps, Number)

# %% [markdown] nbgrader={"grade": false, "grade_id": "cell-9db04f9af71bb32f", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ### 1.3) Short Answer: Is PCA more useful or less useful when you have high multicollinearity among your features? Explain why.

# %%
# Your answer here
"""
It is more useful because the colinearity among features is to some degree "folded" into the 
variance principal components. The process of dimensionality reduction collapses the latent space 
that colinearity implies and in that way reduces its influence on predictions made by that data.
"""
clear_output = True

# %% [markdown] nbgrader={"grade": false, "grade_id": "cell-2be033309999869a", "locked": true, "schema_version": 3, "solution": false, "task": false}
# --- 
#
# ## Part 2: Clustering [Suggested Time: 20 minutes]
#
# ---
#
# In this part, you will answer general questions about clustering.

# %% nbgrader={"grade": false, "grade_id": "cell-7fb56c6a144a1ff1", "locked": true, "schema_version": 3, "solution": false, "task": false}
# Run this cell without changes

from sklearn.cluster import KMeans

# %% [markdown] nbgrader={"grade": false, "grade_id": "cell-f5977bb7be24f780", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ### 2.1) Short Answer: Describe the steps of the k-means clustering algorithm.
#
# Hint: Refer to the animation below, which visualizes the process.
#
# <img src='https://raw.githubusercontent.com/learn-co-curriculum/dsc-cc-images/main/phase_4/centroid.gif'>

# %%
# Your answer here
"""
- The Alogrithm starts by assigning a number of centroids (defined as a hyper-parameter of the model)
  to the points closest to them in the dataset.
- Each grouping becomes considered a "cluster" and the distance between each point in a cluster and
  its centroid is calculated and the centroid is moved along a vector representing the mean of that
  distance.
- The inertia is the magnitude of that vector.
- This is repeated until the inertia of each centroid approaches zero or some other minimum threshold.
"""
clear_output = True


# %% [markdown] nbgrader={"grade": false, "grade_id": "cell-0d929a59f2b64837", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ### 2.2) Write a function `get_labels()` that meets the requirements below to find `k` clusters in a dataset of features `X`, and return the cluster assignment labels for each row of `X`. 
#
# Review the doc-string in the function below to understand the requirements of this function.
#
# _Hint: Within the function, you'll need to:_
# * instantiate a [scikit-learn KMeans object](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html), using `random_state = 1` for reproducibility
# * fit the object to the data
# * return the cluster assignment labels for each row of `X` 

# %% nbgrader={"grade": false, "grade_id": "cell-7d131ed1c76ccc52", "locked": true, "schema_version": 3, "solution": false, "task": false}
# CodeGrade step2.2
# Replace None with appropriate code

def get_labels(k, X):
    """ 
    Finds the labels from a k-means clustering model 

    Parameters: 
    -----------
    k: float object
        number of clusters to use in the k-means clustering model
    X: Pandas DataFrame or array-like object
        Data to cluster

    Returns: 
    --------
    labels: array-like object
        Labels attribute from the k-means model

    """

    # Instantiate a k-means clustering model with random_state=1 and n_clusters=k
    kmeans = KMeans(n_clusters=k, random_state=1)

    # Fit the model to the data
    kmeans.fit(X)
    kmeans.predict(X)

    # Return the predicted labels for each row in the data produced by the model
    return kmeans.labels_


# %%
# This test confirms that you have created a function named get_labels

assert callable(get_labels) 

# This test confirms that get_labels can take the correct parameter types

get_labels(1, np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]]))

# %% [markdown] nbgrader={"grade": false, "grade_id": "cell-3e44a061e098167b", "locked": true, "schema_version": 3, "solution": false, "task": false}
# The next cell uses your `get_labels` function to cluster the wine data, looping through all $k$ values from 2 to 9. It saves the silhouette scores for each $k$ value in a list `silhouette_scores`.

# %% nbgrader={"grade": false, "grade_id": "cell-e668bf9ba032a378", "locked": true, "schema_version": 3, "solution": false, "task": false}
# Run this cell without changes

from sklearn.metrics import silhouette_score

# Preprocessing is needed. Scale the data
scaler_2 = StandardScaler()
X_scaled = scaler_2.fit_transform(X)

# Create empty list for silhouette scores
silhouette_scores = []

# Range of k values to try
k_values = range(2, 10)

for k in k_values:
    labels = get_labels(k, X_scaled)
    score = silhouette_score(X_scaled, labels, metric='euclidean')
    silhouette_scores.append(score)

# %% [markdown] nbgrader={"grade": false, "grade_id": "cell-38e582c973d5e62e", "locked": true, "schema_version": 3, "solution": false, "task": false}
# Next, we plot the silhouette scores obtained for each different value of $k$, against $k$, the number of clusters we asked the algorithm to find. 

# %% nbgrader={"grade": false, "grade_id": "cell-89d3669094b3d4e0", "locked": true, "schema_version": 3, "solution": false, "task": false}
# Run this cell without changes

import matplotlib.pyplot as plt
# %matplotlib inline

plt.plot(k_values, silhouette_scores, marker='o')
plt.title('Silhouette scores vs number of clusters')
plt.xlabel('k (number of clusters)')
plt.ylabel('silhouette score');

# %% [markdown] nbgrader={"grade": false, "grade_id": "cell-74ff2d4b4db6f745", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ### 2.3) Create numeric variable `wine_nclust` containing the value of $k$ you would choose based on the above plot of silhouette scores. 

# %% nbgrader={"grade": false, "grade_id": "cell-3d86a102cb0b9d05", "locked": true, "schema_version": 3, "solution": false, "task": false}
# CodeGrade step2.3
# Replace None with appropriate code

wine_nclust = 3

# %%
# This test confirms that you have created a numeric variable named wine_nclust

assert isinstance(wine_nclust, Number)

# %% [markdown] nbgrader={"grade": false, "grade_id": "cell-b70729833605d576", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ---
#
# ## Part 3: Natural Language Processing [Suggested Time: 20 minutes]
#
# ---
#
# In this third section we will attempt to classify text messages as "SPAM" or "HAM" using TF-IDF Vectorization.

# %% nbgrader={"grade": false, "grade_id": "cell-2bcac79fa0ec69f4", "locked": true, "schema_version": 3, "solution": false, "task": false}
# Run this cell without changes

# Import necessary libraries 
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize

# Generate a list of stopwords 
nltk.download('stopwords')
stops = stopwords.words('english') + list(string.punctuation)

# Read in data
df_messages = pd.read_csv('./spam.csv', usecols=[0,1])

# Convert string labels to 1 or 0 
le = LabelEncoder()
df_messages['target'] = le.fit_transform(df_messages['v1'])

# Examine our data
print(df_messages.head())

# Separate features and labels 
X = df_messages['v2']
y = df_messages['target']

# Create test and train datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state=1)

# %% [markdown] nbgrader={"grade": false, "grade_id": "cell-fdb9d8950abce1f2", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ### 3.1) Create CSR matrices `tf_idf_train` and `tf_idf_test` by using a `TfidfVectorizer` with stop word list `stops` to vectorize `X_train` and `X_test`, respectively.
#
# Besides using the stop word list, use paramater defaults for your TfidfVectorizer. Refer to the documentation about [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html).

# %%
# CodeGrade step3.1
# Replace None with appropriate code

vectorizer = TfidfVectorizer(stop_words=stops)

tf_idf_train = vectorizer.fit_transform(X_train)

tf_idf_test = vectorizer.transform(X_test)

# %%
# These tests confirm that you have created CSR matrices tf_idf_train and tf_idf_test

assert type(tf_idf_train) == scipy.sparse.csr.csr_matrix
assert type(tf_idf_test) == scipy.sparse.csr.csr_matrix

# %% [markdown] nbgrader={"grade": false, "grade_id": "cell-4c0469e57522c867", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ### 3.2) Create an array `y_preds` containing predictions from an untuned `RandomForestClassifier` that uses `tf_idf_train` and `tf_idf_test`.
#
# Use parameter defaults with `random_state=1` for your classifier. Refer to the documentation on [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).

# %% nbgrader={"grade": false, "grade_id": "cell-8b45b5691fce29ee", "locked": true, "schema_version": 3, "solution": false, "task": false}
# CodeGrade step3.2
# Replace None with appropriate code

classifier = RandomForestClassifier(random_state=1)

#Fit on training
classifier.fit(tf_idf_train, y_test)


# Predict using test    
y_preds = classifier.predict(tf_idf_test)

# %%
# This test confirms that you have created an array named y_preds

assert type(y_preds) == np.ndarray

# %% [markdown] nbgrader={"grade": false, "grade_id": "cell-18c4bf3c4e9875a2", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ### 3.3) Short Answer: What would it mean if the word "genuine" had the highest TF-IDF value of all words in one document from our test data?

# %%
# Your answer here
"""
It would mean that "genuine" rarely appeared in other documents but appeared often in this particular
document. Because TF-IDF is a corpus weighted score it would imply that genuine was semantically
important in that particular document.
"""
no_output = True

# %% [markdown] nbgrader={"grade": false, "grade_id": "cell-f190415dece92737", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ---
#
# ## Part 4: Time Series [Suggested Time: 20 minutes]
#
# ---
# In this part you will analyze the price of one stock over time. Each row of the dataset has four prices tracked for each day: 
#
# * Open: The price when the market opens.
# * High: The highest price over the course of the day.
# * Low: The lowest price over the course of the day.
# * Close: The price when the market closes.
#
# <!---Create stock_df and save as .pkl
# stocks_df = pd.read_csv("raw_data/all_stocks_5yr.csv")
# stocks_df["clean_date"] = pd.to_datetime(stocks_df["date"], format="%Y-%m-%d")
# stocks_df.drop(["date", "clean_date", "volume", "Name"], axis=1, inplace=True)
# stocks_df.rename(columns={"string_date": "date"}, inplace=True)
# pickle.dump(stocks_df, open("write_data/all_stocks_5yr.pkl", "wb"))
# --->

# %% nbgrader={"grade": false, "grade_id": "cell-fd9493a8ea890a36", "locked": true, "schema_version": 3, "solution": false, "task": false}
# Run this cell without changes

stocks_df = pd.read_csv('./stocks_5yr.csv')
stocks_df.head()

# %% [markdown] nbgrader={"grade": false, "grade_id": "cell-f6bc3b15110435d3", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ### 4.1) For `stocks_df`, create a DatetimeIndex from the `date` column.
#
# The resulting DataFrame should not have a `date` column, only `open`, `high`, `low`, and `close` columns. 
#
# Hint: First convert the `date` column to Datetime datatype, then set it as the index.

# %% nbgrader={"grade": false, "grade_id": "cell-15921f7c4cf5e767", "locked": true, "schema_version": 3, "solution": false, "task": false}
# CodeGrade step4.1
# Replace None with appropriate code

stocks_df.index = pd.to_datetime(stocks_df['date'])
stocks_df = stocks_df.drop(columns='date')

# %%
# This test confirms that stocks_df has a DatetimeIndex

assert type(stocks_df.index) == pd.core.indexes.datetimes.DatetimeIndex

# This test confirms that stocks_df only has `open`, `high`, `low`, and `close` columns.

assert list(stocks_df.columns) == ['open', 'high', 'low', 'close']

# %% [markdown] nbgrader={"grade": false, "grade_id": "cell-56237f4da08165ef", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ### 4.2) Create a DataFrame `stocks_monthly_df` that resamples `stocks_df` each month with the 'MS' DateOffset to calculate the mean of the four features over each month.
#
# Refer to the [resample documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.resample.html).

# %% nbgrader={"grade": false, "grade_id": "cell-24dbe2526545b9bb", "locked": true, "schema_version": 3, "solution": false, "task": false}
# CodeGrade step4.2
# Replace None with appropriate code

stocks_monthly_df = stocks_df.resample("MS").mean()
stocks_monthly_df.head()

# %%
# This test confirms that you have created a DataFrame named stocks_monthly_df

assert type(stocks_monthly_df) == pd.DataFrame

# This test confirms that stocks_monthy_df has the correct dimensions

assert stocks_monthly_df.shape == (61, 4)

# %% [markdown] nbgrader={"grade": false, "grade_id": "cell-a33f13a6897659d2", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ### 4.3) Create a matplotlib figure `rolling_open_figure` containing a line graph that visualizes the rolling quarterly mean of open prices from `stocks_monthly_df`.
#
# You will use this graph to determine whether the average monthly open stock price is stationary or not.
#
# Hint: use a window size of 3 to represent one quarter of a year

# %% nbgrader={"grade": false, "grade_id": "cell-60d8542e250c354f", "locked": true, "schema_version": 3, "solution": false, "task": false}
# CodeGrade step4.3
# Your code here
# Create rolling mean with window 3 for the open column
roll_mean = stocks_monthly_df.rolling(window=3).mean()

rolling_open_figure, ax = plt.subplots(figsize=(10, 6))
#Plot
ax.plot(roll_mean)
ax.legend()
plt.title('Average Monthly Open Stock Prices (Feb 2013 - Feb 2018)')
plt.show()

# %%
# This test confirms that you have created a figure named rolling_open_figure

assert type(rolling_open_figure) == plt.Figure

# This test confirms that the figure contains exactly one axis

assert len(rolling_open_figure.axes) == 1

# %% [markdown] nbgrader={"grade": false, "grade_id": "cell-0aef1dacb1d8361f", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ### 4.4) Short Answer: Based on your graph from Question 4.3, does the monthly open stock price look stationary? Explain your answer, what statistical test could you use to determine?

# %%
# Your answer here
"""
It does not look stationary because the mean is increasing over time.

You could use the adfuller function to determine a datasets stationarity if it was less obvious than
in the above graph.
"""
no_output = True
