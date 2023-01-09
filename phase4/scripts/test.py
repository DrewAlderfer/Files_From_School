import datetime
import math
import os
import random
import re
import string
import time
import warnings

import keras
import matplotlib.image as mpimg
import matplotlib.pylab as plt
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import pandas.tseries
import pyspark
import seaborn as sns
import statsmodels.api as sm
from keras import models, optimizers, regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential, load_model
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from keras.wrappers.scikit_learn import KerasRegressor
from matplotlib.pylab import rcParams
from mpl_toolkits.mplot3d import Axes3D
from nltk import word_tokenize
from nltk.collocations import *
from nltk.corpus import gutenberg, stopwords
from nltk.tokenize import word_tokenize
from pandas import Series
from sklearn import metrics, svm
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.datasets import load_breast_cancer, load_digits, make_blobs
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             mean_squared_error)
from sklearn.model_selection import (GridSearchCV, cross_val_predict,
                                     train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, StandardScaler
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from surprise import Dataset, Reader
from surprise.model_selection import GridSearchCV
from surprise.prediction_algorithms import KNNBaseline, KNNBasic, KNNWithMeans
from tensorflow.keras.preprocessing import image
