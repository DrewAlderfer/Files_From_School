# Solving Systems of Linear Equations with NumPy - Lab

## Introduction

Now you've gathered all the required skills needed to solve systems of linear equations. You saw why there was a need to calculate inverses of matrices, followed by matrix multiplication to figure out the values of unknown variables.

The exercises in this lab present some problems that can be converted into a system of linear equations.

## Objectives
You will be able to:

- Use matrix algebra and NumPy to solve a system of linear equations given a real-life example
- Use NumPy's linear algebra solver to solve for systems of linear equations

## Exercise 1

A coffee shop is having a sale on coffee and tea.

On day 1, 29 bags of coffee and 41 bags of tea were sold, for a total of 490 dollars.

On day 2, they sold 23 bags of coffee and 41 bags of tea, for which customers paid a total of 448 dollars.

How much does each bag cost?

```python
# Create and solve the relevant system of equations

# Let x be the price of a bag of coffee and y be the price of a bag of tea.

# 29x + 41y = 490

# 23x + 41y = 448

#  Create numpy matrices from above equations
import numpy as np
A = np.matrix([[29, 41], [23, 41]])
B = np.matrix([[490, 448]])

# Calculate inverse of A and take the dot product
A_inv = np.linalg.inv(A)
X = A_inv.dot(B.T)
print(X)

# Verify the answer linalg.solve()
np.linalg.solve(A, B.T)
```
```python
# Describe your result
# bag of coffee = $7 , bag of tea = $7
```
## Exercise 2

The cost of admission to a popular music concert was 162 dollars for 12 children and 3 adults.

The admission was 122 dollars for 8 children and 3 adults in the same music concert.

How much was the admission for each child and adult?

```python
# Create and solve the relevant system of equations

# Let x be the price per child and y be the price per adult

# 12x + 3y = 162
#
# 8x + 3y = 122

# Create matrices in numpy
A = np.matrix([[12, 3],[8, 3]])
B = np.matrix([162, 122])

# Calculate inverse of A and take the dot product
A_inv = np.linalg.inv(A)
X = A_inv.dot(B.T)
print (X)

# Verify the answer linalg.solve()
np.linalg.solve(A, B.T)
```
```python
# Describe your result
# price per child = $10, price per adult = $14
```
## Exercise 3

You want to make a soup containing tomatoes, carrots, and onions.

Suppose you don't know the exact mix to put in, but you know there are 7 individual pieces of vegetables, and there are twice as many tomatoes as onions, and that the 7 pieces of vegetables cost 5.25 USD in total.
You also know that onions cost 0.5 USD each, tomatoes cost 0.75 USD and carrots cost 1.25 USD each.

Create a system of equations to find out exactly how many of each of the vegetables are in your soup.

```python
# Create and solve the relevant system of equations

# Let o represent onions, t - tomatoes and c - carrots.  p--> c .   b--> o, 0---> t

# t + c + o = 7

# .5o + .75t + 1.25c = 5.25

#  t  = 2o which is equal to: -2o + t + 0c = 0

# Create matrices in numpy
A = np.matrix([[1,1,1],[0.5, 0.75, 1.25], [-2,1,0]])
B = np.matrix([[7, 5.25, 0]])

# Calculate inverse of A and take the dot product
A_inv = np.linalg.inv(A)
X = A_inv.dot(B.T)
print (X)

# Verify the answer linalg.solve()
np.linalg.solve(A,B.T)
```
```python
# Describe your result
# onions = 2, tomatoes = 4, carrots = 1 , needed to make the soup
```
## Exercise 4

A landlord owns 3 properties: a 1-bedroom, a 2-bedroom, and a 3-bedroom house.

The total rent he receives is 1240 USD.

He needs to make some repairs, where those repairs cost 10% of the 1-bedroom house\N{RIGHT SINGLE QUOTATION MARK}s rent. The 2-bedroom repairs cost 20% of the 2-bedroom rental price and 30% of the 3-bedroom house's rent for its repairs.  The total repair bill for all three houses was 276 USD.

The 3-bedroom house's rent is twice the 1-bedroom house\N{RIGHT SINGLE QUOTATION MARK}s rent.

How much is the individual rent for three houses?

```python
# Create and solve the relevant system of equations

# Let x,y,z represent rent value for house 1,2 and 3 respectively

# x + y + z = 1240

# .1x + .2y + .3z = 276

# 2x +0y -z = 0

# Create matrices in numpy
A = np.matrix([[1, 1, 1],[0.1, 0.2, 0.3], [2, 0, -1]])
B = np.matrix([[1240, 276, 0]])

# Calculate inverse of A and take the dot product
A_inv = np.linalg.inv(A)
X = A_inv.dot(B.T)
print (X)

# Verify the answer linalg.solve()
np.linalg.solve(A, B.T)
```
```python
# Describe your result
# Rent: house1 = 280, house2 = 400, house3 = 560
```
## Summary
In this lab, you learned how to use NumPy to solve linear equations by taking inverses and matrix multiplication and also using numpy's `solve()` function. You'll now take these skills forward and see how you can define a simple regression problem using linear algebra and solve it with Numpy.


-----File-Boundary-----
# Regression with Linear Algebra - Lab

## Introduction

In this lab, you'll apply regression analysis using simple matrix manipulations to fit a model to given data, and then predict new values for previously unseen data. You'll follow the approach highlighted in the previous lesson where you used NumPy to build the appropriate matrices and vectors and solve for the $\beta$ (unknown variables) vector. The beta vector will be used with test data to make new predictions. You'll also evaluate the model fit.
In order to make this experiment interesting, you'll use NumPy at every single stage of this experiment, i.e., loading data, creating matrices, performing train-test split, model fitting, and evaluation.


## Objectives

In this lab you will:

- Use matrix algebra to calculate the parameter values of a linear regression


First, let's import necessary libraries:

```python
import csv # for reading csv file
import numpy as np
```
## Dataset

The dataset you'll use for this experiment is "**Sales Prices in the City of Windsor, Canada**", something very similar to the Boston Housing dataset. This dataset contains a number of input (independent) variables, including area, number of bedrooms/bathrooms, facilities(AC/garage), etc. and an output (dependent) variable, **price**.  You'll formulate a linear algebra problem to find linear mappings from input features using the equation provided in the previous lesson.

This will allow you to find a relationship between house features and house price for the given data, allowing you to find unknown prices for houses, given the input features.

A description of the dataset and included features is available [here](https://rdrr.io/cran/Ecdat/man/Housing.html).

In your repository, the dataset is available as `windsor_housing.csv`. There are 11 input features (first 11 columns):

	lotsize	bedrooms  bathrms  stories	driveway  recroom	fullbase  gashw	 airco  garagepl   prefarea

and 1 output feature i.e. **price** (12th column).

The focus of this lab is not really answering a preset analytical question, but to learn how you can perform a regression experiment, using mathematical manipulations - similar to the one you performed using `statsmodels`. So you won't be using any `pandas` or `statsmodels` goodness here. The key objectives here are to:

- Understand regression with matrix algebra and
- Mastery in NumPy scientific computation

## Stage 1: Prepare data for modeling

Let's give you a head start by importing the dataset. You'll perform the following steps to get the data ready for analysis:

* Initialize an empty list `data` for loading data
* Read the csv file containing complete (raw) `windsor_housing.csv`. [Use `csv.reader()` for loading data.](https://docs.python.org/3/library/csv.html). Store this in `data` one row at a time

* Drop the first row of csv file as it contains the names of variables (header) which won't be used during analysis (keeping this will cause errors as it contains text values)

* Append a column of all **1**s to the data (bias) as the first column

* Convert `data` to a NumPy array and inspect first few rows

> NOTE: `read.csv()` reads the csv as a text file, so you should convert the contents to float.

```python
# Create Empty lists for storing X and y values
data = []

# Read the data from the csv file
with open('windsor_housing.csv') as f:
    raw = csv.reader(f)
    # Drop the very first line as it contains names for columns - not actual data
    next(raw)
    # Read one row at a time. Append one to each row
    for row in raw:
        ones = [1.0]
        for r in row:
            ones.append(float(r))
        # Append the row to data
        data.append(ones)
data = np.array(data)
data[:5,:]
```
## Step 2: Perform a 80/20 train-test split

Explore NumPy's official documentation to manually split a dataset using a random sampling method of your choice. Some useful methods are located in the [numpy.random library](https://docs.scipy.org/doc/numpy-1.14.0/reference/routines.random.html).
* Perform a **random** 80/20 split on data using a method of your choice in NumPy
* Split the data to create `x_train`, `y_train`, `x_test`, and `y_test` arrays
* Inspect the contents to see if the split performed as expected

> Note: When randomly splitting data, it's always recommended to set a seed in order to ensure reproducibility

```python
# Set a seed
np.random.seed(42)
# Perform an 80/20 split
# Make array of indices
all_idx = np.arange(data.shape[0])
# Randomly choose 80% subset of indices without replacement for training
training_idx = np.random.choice(all_idx, size=round(546*.8), replace=False)
# Choose remaining 20% of indices for testing
test_idx = all_idx[~np.isin(all_idx, training_idx)]
# Subset data
training, test = data[training_idx,:], data[test_idx,:]

# Check the shape of datasets
print ('Raw data Shape: ', data.shape)
print ('Train/Test Split:', training.shape, test.shape)

# Create x and y for test and training sets
x_train = training[:,:-1]
y_train = training [:,-1]

x_test = test[:,:-1]
y_test = test[:,-1]

# Check the shape of datasets
print ('x_train, y_train, x_test, y_test:', x_train.shape, y_train.shape, x_test.shape, y_test.shape)
```
## Step 3: Calculate the `beta`

With $X$ and $y$ in place, you can now compute your beta values with $x_\text{train}$ and $y_\text{train}$ as:
#### $\beta = (x_\text{train}^T. x_\text{train})^{-1} . x_\text{train}^T . y_\text{train}$

* Using NumPy operations (transpose, inverse) that we saw earlier, compute the above equation in steps
* Print your beta values

```python
# Calculate Xt.X and Xt.y for beta = (XT . X)-1 . XT . y - as seen in previous lessons
Xt = np.transpose(x_train)
XtX = np.dot(Xt,x_train)
Xty = np.dot(Xt,y_train)

# Calculate inverse of Xt.X
XtX_inv = np.linalg.inv(XtX)

# Take the dot product of XtX_inv with Xty to compute beta
beta = XtX_inv.dot(Xty)

# Print the values of computed beta
print(beta)
```
## Step 4: Make predictions
Great, you now have a set of coefficients that describe the linear mappings between $X$ and $y$. You can now use the calculated beta values with the test datasets that we left out to calculate $y$ predictions. Next, use all features in turn and multiply it with this beta. The result will give a prediction for each row which you can append to a new array of predictions.

$\hat{y} = x\beta = \beta_0 + \beta_1 x_1 +  \beta_2 x_2 + \ldots + \beta_m x_m $

* Create a new empty list (`y_pred`) for saving predictions
* For each row of `x_test`, take the dot product of the row with beta to calculate the prediction for that row
* Append the predictions to `y_pred`
* Print the new set of predictions

```python
# Calculate and print predictions for each row of X_test
y_pred = []
for row in x_test:
    pred = row.dot(beta)
    y_pred.append(pred)
```
## Step 5: Evaluate model

### Visualize actual vs. predicted values
This is exciting, now your model can use the beta value to predict the price of houses given the input features. Let's plot these predictions against the actual values in `y_test` to see how much our model deviates.

```python
# Plot predicted and actual values as line plots
import matplotlib.pyplot as plt
%matplotlib inline
from pylab import rcParams
rcParams['figure.figsize'] = 15, 10
plt.style.use('ggplot')

plt.plot(y_pred, linestyle='-', marker='o', label='predictions')
plt.plot(y_test, linestyle='-', marker='o', label='actual values')
plt.title('Actual vs. predicted values')
plt.legend()
plt.show()
```
This doesn't look so bad, does it? Your model, although isn't perfect at this stage, is making a good attempt to predict house prices although a few prediction seem a bit out. There could be a number of reasons for this. Let's try to dig a bit deeper to check model's predictive abilities by comparing these prediction with actual values of `y_test` individually. That will help you calculate the RMSE value (root mean squared error) for your model.

### Root Mean Squared Error
Here is the formula for RMSE:

$$ \large RMSE = \sqrt{\sum^N_{i=1}\dfrac{ (\text{Predicted}_i-\text{Actual}_i)^2}{N}}$$

* Initialize an empty array `err`
* For each row in `y_test` and `y_pred`, take the squared difference and append error for each row in the `err` array
* Calculate $RMSE$ from `err` using the formula shown above

```python
# Due to random split, your answers may vary
# Calculate RMSE
err = []
for pred,actual in zip(y_pred,y_test):
    sq_err = (pred - actual) ** 2
    err.append(sq_err)
mean_sq_err = np.array(err).mean()
root_mean_sq_err = np.sqrt(mean_sq_err)
root_mean_sq_err

# Due to random split, your answers may vary
# RMSE = 14868.172645765708
```
### Normalized root mean squared error
The above error is clearly in terms of the dependent variable, i.e., the final house price. You can also use a normalized mean squared error in case of multiple regression which can be calculated from RMSE using following the formula:

$$ \large NRMSE = \dfrac{RMSE}{max_i y_i - min_i y_i} $$

* Calculate normalized RMSE

```python

root_mean_sq_err/(y_train.max() - y_train.min())

# Due to random split, your answers may vary
# 0.09011013724706489
```
There it is. A complete multiple regression analysis using nothing but NumPy. Having good programming skills in NumPy allows you to dig deeper into analytical algorithms in machine learning and deep learning. Using matrix multiplication techniques you saw here, you can easily build a whole neural network from scratch.

## Level up (Optional)

* Calculate the R-squared and adjusted R-squared for the above model
* Plot the residuals (similar to `statsmodels`) and comment on the variance and heteroscedasticity
* Run the experiment in `statsmodels` and compare the performance of both approaches in terms of computational cost

## Summary

In this lab, you built a predictive model for predicting house prices. Remember this is a very naive implementation of regression modeling. The purpose here was to get an introduction to the applications of linear algebra into machine learning and predictive analysis. There are a number of shortcomings in this modeling approach and you can further apply a number of data modeling techniques to improve this model.


-----File-Boundary-----
# Introduction to Derivatives - Lab

## Introduction
In this lab, we will practice our knowledge of derivatives. Remember that our key formula for derivatives, is
$f'(x) = \dfrac{\Delta y}{\Delta x} =  \dfrac{f(x + \Delta x) - f(x)}{\Delta x}$.  So in driving towards this formula, we will do the following:

1. Learn how to represent linear and nonlinear functions in code
2. Then, because our calculation of a derivative relies on seeing the output at an initial value and the output at that value plus $\Delta x$, we need an `output_at` function
3. Then we will be able to code the $\Delta f$ function that sees the change in output between the initial $x$ and that initial $x$ plus the $\Delta x$
4. Finally, we will calculate the derivative at a given $x$ value, `derivative_at`

## Objectives

You will be able to:

- Use python functions to demonstrate derivatives of functions
- Describe what a derivative means in the context of a real-world example

## Let's begin: Starting with functions

### 1. Representing Functions

We are about to learn to take the derivative of a function in code.  But before doing so, we need to learn how to express any kind of function in code.  This way when we finally write our functions for calculating the derivative, we can use them with both linear and nonlinear functions.

For example, we want to write the function $f(x) = 2x^2 + 4x - 10 $ in a way that allows us to easily determine the exponent of each term.

This is our technique: write the formula as a numpy array. For example, for a function $f(x)= 7x^3$:

```python
arr = np.array([7, 3])
arr[0] # 7
arr[1] # 3
```

Take the following function as an example:

$$f(x) = 4x^2 + 4x - 10 $$

We can use a [N-dimensional array](https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.ndarray.html) to represent this:

```python
import numpy as np
array_1 = np.array([[4, 2], [4, 1], [-10, 0]])
```
```python
np.shape(array_1)
```
So each row in the `np.array` represents a different term in the function.  The first column is the term's constant and the second column is the term's exponent.  Thus $4x^2$ translates to `[4, 2]` and  $-10$ translates to `[-10, 0]` because $-10$ equals $-10*x^0$.
> We'll refer to this `np.array` as "array of terms", or `array_of_terms`.

Ok, so give this a shot. Write $ f(x) = 4x^3 + 11x^2 $ as an array of terms.  Assign it to the variable `array_2`.

```python
array_2 = np.array([[4, 3], [11, 2]])
```
### 2. Evaluating a function at a specific point

Now that we can represent a function in code, let's write a Python function called `term_output` that can evaluate what a single term equals at a value of $x$.

* For example, when $x = 2$, the term $3x^2 = 3*2^2 = 12 $.
* So we represent $3x^2$ in code as `(3, 2)`, and:
* `term_output((3, 2), 2)` should return 12

```python
def term_output(array, input_value):
    return array[0]*input_value**array[1]
```
```python
term_output(np.array([3, 2]), 2) # 12
```
> **Hint:** To raise a number to an exponent in python, like 3^2 use the double star, as in:
```python
3**2 # 9
```

Now write a function called `output_at`, when passed an `array_of_terms` and a value of $x$, calculates the value of the function at that value.
* For example, we'll use `output_at` to calculate $f(x) = 3x^2 - 11$.
* Then `output_at([np.array([[3, 2], [-11, 0]]), 2)` should return $f(2) = 3*2^2 - 11 = 1$. Store `np.array([[3, 2], [-11, 0]])` as `array_3`.

```python
def output_at(array_of_terms, x_value):
    outputs = []
    for i in range(int(np.shape(array_of_terms)[0])):
        outputs.append(array_of_terms[i][0]*x_value**array_of_terms[i][1])
    return sum(outputs)
```
```python
array_3 = np.array([[3, 2], [-11, 0]])
```
Verify that $f(2) = 3*2^2 - 11 = 1$.

```python
output_at(array_3, 2) # 1
```
What value does $f(3)$ return?

```python
output_at(array_3, 3) # 16
```
Now we can use our `output_at` function to display our function graphically.  We simply declare a list of `x_values` and then calculate `output_at` for each of the `x_values`.

```python
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12,6))
x_values = np.linspace(-30, 30, 100)
y_values = list(map(lambda x: output_at(array_3, x), x_values))

plt.plot(x_values, y_values, label = "3x^2 - 11")

ax.legend(loc="upper center",fontsize='large')
plt.show()
```
## Moving to derivatives of linear functions

Let's start with a function, $f(x) = 4x + 15$.  We represent the function as the following:

```python
lin_function = np.array([[4, 1], [15, 0]])
```
We can plot the function by calculating outputs at a range of $x$ values.  Note that we use our `output_at` function to calculate the output at each individual $x$ value.

```python
fig, ax = plt.subplots(figsize=(12,6))
x_values = np.linspace(0, 5, 100)
y_values = list(map(lambda x: output_at(lin_function, x), x_values))

plt.plot(x_values, y_values, label = "4x + 15")

ax.legend(loc="upper center",fontsize='large')

plt.show()
```
Ok, time to do what we are here for: *derivatives*.  Remember that the derivative is the instantaneous rate of change of a function, and is expressed as:

$$ f'(x) = \frac{\Delta f}{\Delta x}  = \frac{f(x + \Delta x) - f(x)}{\Delta x}  $$

### Writing a function for $\Delta f$

We can see from the formula above that  $\Delta f = f(x + \Delta x ) - f(x) $.  Write a function called `delta_f` that, given a `list_of_terms`, an `x_value`, and a value $\Delta x $, returns the change in the output over that period.
> **Hint** Don't forget about the `output_at` function.  The `output_at` function takes a list of terms and an $x$ value and returns the corresponding output.  So really **`output_at` is equivalent to $f(x)$**, provided a function and a value of x.

```python
def delta_f(array_of_terms, x_value, delta_x):
    return output_at(array_of_terms, x_value + delta_x) - output_at(array_of_terms, x_value)
```
```python
delta_f(lin_function, 2, 1) # 4
```
So for $f(x) = 4x + 15$, when $x$ = 2, and $\Delta x = 1$, $\Delta f$ is 4.

### Plotting our function, delta f, and delta x

Let's show $\Delta f$ and $\Delta x$ graphically.

```python
x_value = 2
delta_x = 1
```
```python
fig, ax = plt.subplots(figsize=(10,6))

x_values = np.linspace(0, 5, 100)
y_values = list(map(lambda x: output_at(lin_function, x), x_values))

plt.plot(x_values, y_values, label = "4x + 15")

# delta x
y_val = output_at(lin_function, x_value)
hline_lab= 'delta x = ' + str(delta_x)
plt.hlines(y=y_val, xmin= x_value, xmax= x_value + delta_x, color="lightgreen", label = hline_lab)

# delta f
y_val_max = output_at(lin_function, x_value + delta_x)
vline_lab =  'delta f = ' + str(y_val_max-y_val)
plt.vlines(x = x_value + delta_x , ymin= y_val, ymax=y_val_max, color="darkorange", label = vline_lab)
ax.legend(loc='upper left', fontsize='large')

plt.show()
```
### Calculating the derivative

Write a function, `derivative_at` that calculates $\dfrac{\Delta f}{\Delta x}$ when given a `array_of_terms`, an `x_value` for the value of $(x)$ the derivative is evaluated at, and `delta_x`, which represents $\Delta x$.

Let's try this for $f(x) = 4x + 15 $.  Round the result to three decimal places.

```python
def derivative_of(array_of_terms, x_value, delta_x):
    delta = delta_f(array_of_terms, x_value, delta_x)
    return round(delta/delta_x, 3)
```
Now let's use this function along with our stored `x_value` and `delta_x`.

```python
derivative_of(lin_function, x_value=x_value, delta_x=delta_x) # 4.0
```
### Building more plots

Ok, now that we have written a Python function that allows us to plot our list of terms, we can write a function called `tangent_line` that outputs the necessary terms to plot the slope of the function between initial $x$ and $x$ plus $\Delta x$. We'll walk you through this one.

```python
def tangent_line(array_of_terms, x_value, line_length = 4, delta_x = .01):
    y = output_at(array_of_terms, x_value)
    derivative_at = derivative_of(array_of_terms, x_value, delta_x)

    x_dev = np.linspace(x_value - line_length/2, x_value + line_length/2, 50)
    tan = y + derivative_at *(x_dev - x_value)
    return {'x_dev':x_dev, 'tan':tan, 'lab': " f' (x) = " + str(derivative_at)}
```
> Our `tangent_line` function takes as arguments `list_of_terms`, `x_value`, which is where our line should be tangent to our function, `line_length` as the length of our tangent line, and `delta_x` which is our $\Delta x$.


> The return value of `tangent_line` is a dictionary that represents the tangent line at that value of $x$. It uses `output_at()` to calculate the function value at a particular $x$ and the `derivative_of()` function you wrote above to calculate the slope of the tangent line.
Next, it uses `line_length` along with the `np.linspace` to generate an array of x-values to be used as an input to generate the tangent line `tan`.

Let's look at the output of the `tangent_line()`, using our `lin_function`,  $x$ equal to 2, $\Delta_x$ equal to 0.1 and `line_length` equal to 2.

```python
tan_line = tangent_line(lin_function, 2, line_length = 2, delta_x = .1)
tan_line
```
Now, let's plot our function, $\Delta f$ and $\Delta x$ again along with our `rate_of_change` line.

```python
fig, ax = plt.subplots(figsize=(10,6))

x_values = np.linspace(0, 5, 100)
y_values = list(map(lambda x: output_at(lin_function, x), x_values))

plt.plot(x_values, y_values, label = "4x + 15")
# tangent_line
plt.plot(tan_line['x_dev'], tan_line['tan'], color = "yellow", label = tan_line['lab'])

# delta x
y_val = output_at(lin_function, x_value)
hline_lab= 'delta x = ' + str(delta_x)
plt.hlines(y=y_val, xmin= x_value, xmax= x_value + delta_x, color="lightgreen", label = hline_lab)

# delta f
y_val_max = output_at(lin_function, x_value + delta_x)
vline_lab =  'delta f = ' + str(y_val_max-y_val)
plt.vlines(x = x_value + delta_x , ymin= y_val, ymax=y_val_max, color="darkorange", label = vline_lab)
ax.legend(loc='upper left', fontsize='large')

plt.show()
```
So that function highlights the rate of change is moving at precisely the point $x = 2$. Sometimes it is useful to see how the derivative is changing across all $x$ values.  With linear functions, we know that our function is always changing by the same rate, and therefore the rate of change is constant.  Let's write a function that allows us to see the function and the derivative side by side.

```python
fig, ax = plt.subplots(figsize=(10,4))

x_values = np.linspace(0, 5, 100)
function_values = list(map(lambda x: output_at(lin_function, x),x_values))
derivative_values = list(map(lambda x: derivative_of(lin_function, x, delta_x), x_values))

# plot 1
plt.subplot(121)
plt.plot(x_values, function_values, label = "f (x)")
plt.legend(loc="upper left", bbox_to_anchor=[0, 1], ncol=2, fancybox=True)

# plot 2
plt.subplot(122)
plt.plot(x_values, derivative_values,color="darkorange", label = "f '(x)")
plt.legend(loc="upper left");

plt.show()
```
## Summary

In this section, we coded out our function for calculating and plotting the derivative.  We started by seeing how we can represent different types of functions.  Then we moved onto writing the `output_at` function which evaluates a provided function at a value of x.  We calculated `delta_f` by subtracting the output at initial x value from the output at that initial x plus delta x.  After calculating `delta_f`, we moved onto our `derivative_at` function, which simply divided `delta_f` from `delta_x`.

In the final section, we plotted out some of our findings. We introduced the `tangent_line` function to get the slope for a function between an initial $x$, and $x + \Delta x $


-----File-Boundary-----
# Gradient Descent: Step Sizes - Lab

## Introduction

In this lab, you'll practice applying gradient descent.  As you know, gradient descent begins with an initial regression line and moves to a "best fit" regression line by changing values of $m$ and $b$ and evaluating the RSS.  So far, we have illustrated this technique by changing the values of $m$ and evaluating the RSS.  In this lab, you will work through applying this technique by changing the value of $b$ instead.  Let's get started.

## Objectives

You will be able to:

- Use gradient descent to find the optimal parameters for a linear regression model
- Describe how to use an RSS curve to find the optimal parameters for a linear regression model

```python
import sys
import numpy as np
np.set_printoptions(formatter={'float_kind':'{:f}'.format})
import matplotlib.pyplot as plt
```
## Setting up Our Initial Regression Line

Once again, we'll take a look at revenues (our data example), which looks like this:

```python
np.random.seed(225)

x = np.random.rand(30, 1).reshape(30)
y_randterm = np.random.normal(0,3,30)
y = 3 + 50*x + y_randterm

fig, ax = plt.subplots()
ax.scatter(x, y, marker=".", c="b")
ax.set_xlabel("x", fontsize=14)
ax.set_ylabel("y", fontsize=14)
fig.suptitle("Revenues");
```
We can start with some values for an initial not-so-accurate regression line, $y = 43x + 12$.

```python
def regression_formula(x):
    return 43*x + 12
```
We plot this line with the same data below:

```python
fig, ax = plt.subplots()
ax.scatter(x, y, marker=".", c="b")
ax.plot(x, regression_formula(x), color="orange", label=r'$y = 43x + 12$')
ax.set_xlabel("x", fontsize=14)
ax.set_ylabel("y", fontsize=14)
fig.suptitle("Revenues", fontsize=16)
ax.legend();
```
As you can see, this line is near the data, but not quite right. Let's evaluate that more formally using RSS.

```python
def errors(x_values, y_values, m, b):
    y_line = (b + m*x_values)
    return (y_values - y_line)

def squared_errors(x_values, y_values, m, b):
    return errors(x_values, y_values, m, b)**2

def residual_sum_squares(x_values, y_values, m, b):
    return sum(squared_errors(x_values, y_values, m, b))
```
Now using the `residual_sum_squares`, function, we calculate the RSS to measure the accuracy of the regression line to our data.  Let's take another look at that function:

```python
residual_sum_squares(x, y , 43, 12)
```
So, for a $b$ of 12, we are getting an RSS of 1117.8. Let's see if we can do better than that!

### Building a cost curve

Now let's use the `residual_sum_squares` function to build a cost curve.  Keeping the $m$ value fixed at $43$, write a function called `rss_values`.
* `rss_values` passes our dataset with the `x_values` and `y_values` arguments.
* It also takes a list of values of $b$, and an initial $m$ value as arguments.
* It outputs a NumPy array with a first column of `b_values` and second column of `rss_values`. For example, this input:
  ```python
  rss_values(x, y, 43, [1, 2, 3])
  ```
  Should produce this output:
  ```python
  array([[1.000000, 1368.212664],
       [2.000000, 1045.452004],
       [3.000000, 782.691343]])
  ```
  Where 1, 2, and 3 are the b values an 1368.2, 1045.5 and 782.7 are the associated RSS values.

*Hint:* Check out `np.zeros` ([documentation here](https://numpy.org/doc/stable/reference/generated/numpy.zeros.html)).

```python
def rss_values(x_values, y_values, m, b_values):
    # Make an "empty" 2D NumPy array with 2 columns
    # and as many rows as there are values in b_values
    # (Instead of being truly empty, fill it with zeros)
    table = np.zeros(
        (len(b_values), # One row for every value
        2))             # Two columns

    # Loop over all of the values in b_values
    for idx, b_val in enumerate(b_values):
        # Add the current b value and associated RSS to the
        # NumPy array
        table[idx, 0] = b_val
        table[idx, 1] = residual_sum_squares(x_values, y_values, m, b_val)

    # Return the NumPy array
    return table
```
```python
example_rss = rss_values(x, y, 43, [1,2,3])

# Should return a NumPy array
assert type(example_rss) == np.ndarray

# Specifically a 2D array
assert example_rss.ndim == 2

# The shape should match the number of b values passed in
assert example_rss.shape == (3, 2)

example_rss
```
Now let's make more of an attempt to find the actual best b value for our `x` and `y` data.

Make an array `b_val` that contains values between 0 and 14 with steps of 0.5.

*Hint:* Check out `np.arange` ([documentation here](https://numpy.org/doc/stable/reference/generated/numpy.arange.html))

```python
b_val = np.arange(0, 14.5, step=0.5)
b_val
```
Now use your `rss_values` function to find the RSS values for each value in `b_val`. Continue to use the m value of 43.

We have included code to print out the resulting table.

```python
bval_rss = rss_values(x, y, 43, b_val)
np.savetxt(sys.stdout, bval_rss, '%16.2f') # this line is to round your result, which will make things look nicer.
```
This represents our cost curve!

Let's plot this out using a a line chart.

```python
fig, ax = plt.subplots(figsize=(10,7))
ax.plot(bval_rss[:,0], bval_rss[:,1])
ax.set_xlabel(r'$b$ values', fontsize=14)
ax.set_ylabel("RSS", fontsize=14)
fig.suptitle("RSS with Changes to Intercept", fontsize=16);
```
## Looking at the Slope of Our Cost Curve

In this section, we'll work up to building a gradient descent function that automatically changes our step size.  To get you started, we'll provide a function called `slope_at` that calculates the slope of the cost curve at a given point on the cost curve.

Use the `slope_at` function for b-values 3 and 6 (continuing to use an m of 43).

```python
def slope_at(x_values, y_values, m, b):
    delta = .001
    base_rss = residual_sum_squares(x_values, y_values, m, b)
    delta_rss = residual_sum_squares(x_values, y_values, m, b + delta)
    numerator = delta_rss - base_rss
    slope = numerator/delta
    return slope
```
```python
slope_at(x, y, 43, 3)
```
```python
slope_at(x, y, 43, 6)
```
The `slope_at` function takes in our dataset, and returns the slope of the cost curve at that point.  So the numbers -232.73 and -52.73 reflect the slopes at the cost curve when b is 3 and 6 respectively.

Below, we plot these on the cost curve.

```python
# Setting up to repeat the same process for 3 and 6
# (You can change these values to see other tangent lines)
b_vals = [3, 6]

def plot_slope_at_b_vals(x, y, m, b_vals, bval_rss):
    # Find the slope at each of these values
    slopes = [slope_at(x, y, m, b) for b in b_vals]
    # Find the RSS at each of these values
    rss_values = [residual_sum_squares(x, y, m, b) for b in b_vals]

    # Calculate the actual x and y locations for plotting
    x_values = [np.linspace(b-1, b+1, 100) for b in b_vals]
    y_values = [rss_values[i] + slopes[i]*(x_values[i] - b) for i, b in enumerate(b_vals)]

    # Plotting the same RSS curve as before
    fig, ax = plt.subplots(figsize=(10,7))
    ax.plot(bval_rss[:,0], bval_rss[:,1])
    ax.set_xlabel(r'$b$ values', fontsize=14)
    ax.set_ylabel("RSS", fontsize=14)

    # Adding tangent lines for the selected b values
    for i in range(len(b_vals)):
        ax.plot(x_values[i], y_values[i], label=f"slope={round(slopes[i], 2)}", linewidth=3)

    ax.legend(loc='upper right', fontsize='large')
    fig.suptitle(f"RSS with Intercepts {[round(b, 3) for b in b_vals]} Highlighted", fontsize=16)

plot_slope_at_b_vals(x, y, 43, b_vals, bval_rss)
```
Let's look at the above graph.  When the curve is steeper and downwards at $b = 3$, the slope is around -232.73.  And at $b = 6$ with our cost curve becoming flatter, our slope is around -52.73.

## Moving Towards Gradient Descent

Now that we are familiar with our `slope_at` function and how it calculates the slope of our cost curve at a given point, we can begin to use that function with our gradient descent procedure.

Remember that gradient descent works by starting at a regression line with values m, and b, which corresponds to a point on our cost curve.  Then we alter our m or b value (here, the b value) by looking to the slope of the cost curve at that point.  Then we look to the slope of the cost curve at the new b value to indicate the size and direction of the next step.

So now let's write a function called `updated_b`.  The function will tell us the step size and direction to move along our cost curve.  The `updated_b` function takes as arguments an initial value of $b$, a learning rate, and the `slope` of the cost curve at that value of $m$.  Its return value is the next value of `b` that it calculates.

```python
def updated_b(b, learning_rate, cost_curve_slope):
    change_to_b = -1 * learning_rate * cost_curve_slope
    return change_to_b + b
```
Test out your function below. Each time we update `current_b` and step a little closer to the optimal value.

```python
b_vals = []

current_b = 3
b_vals.append(current_b)

current_cost_slope = slope_at(x, y, 43, current_b)
new_b = updated_b(current_b, .01, current_cost_slope)
print(f"""
Current b: {round(current_b, 3)}
Cost slope for current b: {round(current_cost_slope, 3)}
Updated b: {round(new_b, 3)}
""")

# Same code repeated 3 times
current_b = new_b
b_vals.append(current_b)

current_cost_slope = slope_at(x, y, 43, current_b)
new_b = updated_b(current_b, .01, current_cost_slope)
print(f"""
Current b: {round(current_b, 3)}
Cost slope for current b: {round(current_cost_slope, 3)}
Updated b: {round(new_b, 3)}
""")

current_b = new_b
b_vals.append(current_b)

current_cost_slope = slope_at(x, y, 43, current_b)
new_b = updated_b(current_b, .01, current_cost_slope)
print(f"""
Current b: {round(current_b, 3)}
Cost slope for current b: {round(current_cost_slope, 3)}
Updated b: {round(new_b, 3)}
""")

current_b = new_b
b_vals.append(current_b)

current_cost_slope = slope_at(x, y, 43, current_b)
new_b = updated_b(current_b, .01, current_cost_slope)
print(f"""
Current b: {round(current_b, 3)}
Cost slope for current b: {round(current_cost_slope, 3)}
Updated b: {round(new_b, 3)}
""")
```
Take a careful look at how we use the `updated_b` function.  By using our updated value of $b$ we are quickly converging towards an optimal value of $b$.

In the cell below, we plot each of these b values and their associated cost curve slopes. Note how the tangent lines get closer together as the steps approach the minimum.

```python
plot_slope_at_b_vals(x, y, 43, b_vals, bval_rss)
```
We can visualize the actual lines created by those b values against the data like this:

```python
fig, ax = plt.subplots(figsize=(10,7))
ax.scatter(x, y, marker=".", c="b")
colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
for i, b in enumerate(b_vals):
    ax.plot(x, x*43 + b, color=colors[i], label=f'$y = 43x + {round(b, 3)}$', linewidth=3)
ax.set_xlabel("x", fontsize=14)
ax.set_ylabel("y", fontsize=14)
fig.suptitle("Revenues", fontsize=16)
ax.legend();
```
Now let's write another function called `gradient_descent`.  The inputs of the function are `x_values`, `y_values`, `steps`, the `m` we are holding constant, the `learning_rate`, and the `current_b` that we are looking at.  The `steps` arguments represent the number of steps the function will take before the function stops.  We can get a sense of the return value in the cell below.  It is a list of dictionaries, with each dictionary having a key of the current `b` value, the `slope` of the cost curve at that `b` value, and the `rss` at that `b` value.

```python
def gradient_descent(x_values, y_values, steps, current_b, learning_rate, m):
    cost_curve = []
    for i in range(steps):
        current_cost_slope = slope_at(x_values, y_values, m, current_b)
        current_rss = residual_sum_squares(x_values, y_values, m, current_b)
        cost_curve.append({'b': current_b, 'rss': round(current_rss,2), 'slope': round(current_cost_slope,2)})
        current_b = updated_b(current_b, learning_rate, current_cost_slope)
    return cost_curve
```
```python
descent_steps = gradient_descent(x, y, 15, 0, learning_rate = .005, m = 43)
descent_steps

#[{'b': 0, 'rss': 1750.97, 'slope': -412.73},
# {'b': 2.063653301142949, 'rss': 1026.94, 'slope': -288.91},
# {'b': 3.5082106119386935, 'rss': 672.15, 'slope': -202.24},
# {'b': 4.519400729495828, 'rss': 498.29, 'slope': -141.57},
# {'b': 5.2272338117862205, 'rss': 413.1, 'slope': -99.1},
# {'b': 5.72271696938941, 'rss': 371.35, 'slope': -69.37},
# {'b': 6.06955517971187, 'rss': 350.88, 'slope': -48.56},
# {'b': 6.312341926937677, 'rss': 340.86, 'slope': -33.99},
# {'b': 6.482292649996282, 'rss': 335.94, 'slope': -23.79},
# {'b': 6.601258156136964, 'rss': 333.53, 'slope': -16.66},
# {'b': 6.684534010435641, 'rss': 332.35, 'slope': -11.66},
# {'b': 6.742827108444089, 'rss': 331.77, 'slope': -8.16},
# {'b': 6.7836322770506285, 'rss': 331.49, 'slope': -5.71},
# {'b': 6.812195895074922, 'rss': 331.35, 'slope': -4.0},
# {'b': 6.832190427692808, 'rss': 331.28, 'slope': -2.8}]
```
Looking at our b-values, you get a pretty good idea of how our gradient descent function works.  It starts far away with $b = 0$, and the step size is relatively large, as is the slope of the cost curve.  As the $b$ value updates such that it approaches a minimum of the RSS, the slope of the cost curve and the size of each step both decrease.

Compared to the initial RSS of 1117.8 when $b$ was 12, we are down to 331.3!

Remember that each of these steps indicates a change in our regression line's slope value towards a "fit" that more accurately matches our dataset.  Let's plot the final regression line as found before, with $m=43$ and $b=6.83$

```python
fig, ax = plt.subplots()
ax.scatter(x, y, marker=".", c="b")
ax.plot(x, x*43 + 6.83, color='#17becf', label=f'$y = 43x + 6.83$')
ax.set_xlabel("x", fontsize=14)
ax.set_ylabel("y", fontsize=14)
fig.suptitle("Revenues", fontsize=16)
ax.legend();
```
As you can see, this final intercept value of around $b=6.8$ matches our data much better than the previous guess of 12. Remember that the slope was kept constant. You can see that lifting the slope upwards could probably even lead to a better fit!

## Summary

In this lesson, we learned some more about gradient descent.  We saw how gradient descent allows our function to improve to a regression line that better matches our data.  We see how to change our regression line, by looking at the Residual Sum of Squares related to the current regression line. We update our regression line by looking at the rate of change of our RSS as we adjust our regression line in the right direction -- that is, the slope of our cost curve.  The larger the magnitude of our rate of change (or slope of our cost curve) the larger our step size.  This way, we take larger steps the further away we are from our minimizing our RSS, and take smaller steps as we converge towards our minimum RSS.


-----File-Boundary-----
# Applying Gradient Descent - Lab

## Introduction

In the last lesson, we derived the functions that we help us descend along our cost functions efficiently.  Remember that this technique is not so different from what we saw with using the derivative to tell us our next step size and direction in two dimensions.

<img src="https://raw.githubusercontent.com/learn-co-curriculum/dsc-applying-gradient-descent-lab/master/images/slopes.png" alt="RSS with changes to slope" />

When descending along our cost curve in two dimensions, we used the slope of the tangent line at each point, to tell us how large of a step to take next.  And with the cost curve being a function of $m$ and $b$, we had to use the gradient to determine each step.

<img src="https://raw.githubusercontent.com/learn-co-curriculum/dsc-applying-gradient-descent-lab/master/images/new_gradientdescent.png" alt="gradient descent in 3d with absolute minimum highlighted" width="600">

But really it's an analogous approach.  Just like we can calculate the use derivative of a function $f(x)$ to calculate the slope at a given value of $x$ on the graph and thus our next step.  Here, we calculated the partial derivative with respect to both variables, our slope and y-intercept, to calculate the amount to move next in either direction and thus to steer us towards our minimum.

## Objectives

You will be able to:

* Create functions to perform a simulation of gradient descent for an actual dataset
* Represent RSS as a multivariable function and take partial derivatives to perform gradient descent

## Reviewing our gradient descent formulas

Luckily for us, we already did the hard work of deriving these formulas.  Now we get to see the fruit of our labor.  The following formulas tell us how to update regression variables of $m$ and $b$ to approach a "best fit" line.

- $ \frac{dJ}{dm}J(m,b) = -2\sum_{i = 1}^n x_i(y_i - (mx_i + b)) = -2\sum_{i = 1}^n x_i*\epsilon_i$
- $ \frac{dJ}{db}J(m,b) = -2\sum_{i = 1}^n(y_i - (mx_i + b)) = -2\sum_{i = 1}^n \epsilon_i $

Now the formulas above tell us to take some dataset, with values of $x$ and $y$, and then given a regression formula with values $m$ and $b$, iterate through our dataset, and use the formulas to calculate an update to $m$ and $b$.  So ultimately, to descend along the cost function, we will use the calculations:

`current_m` = `old_m` $ -  (-2*\sum_{i=1}^n x_i*\epsilon_i )$

`current_b` =  `old_b` $ - ( -2*\sum_{i=1}^n \epsilon_i )$

Ok let's turn this into code.  First, let's initialize our data like we did before:

```python
import numpy as np
np.set_printoptions(formatter={'float_kind':'{:f}'.format})
import matplotlib.pyplot as plt
%matplotlib inline
np.random.seed(225)

x = np.random.rand(30, 1).reshape(30)
y_randterm = np.random.normal(0,3,30)
y = 3 + 50* x + y_randterm

data = np.array([y, x])
data = np.transpose(data)

plt.plot(x, y, '.b')
plt.xlabel("x", fontsize=14)
plt.ylabel("y", fontsize=14);
```
Now

- Let's set our initial regression line by initializing $m$ and $b$ variables as zero.  Store them in `b_current` and `m_current`.
- Let's next initialize updates to these variables by setting the variables, `update_to_b` and `update_to_m` equal to 0.
- Define an `error_at` function which returns the error $\epsilon_i$ for a given $i$. The parameters are:
> point: a row of the particular data set
> $b$: the intercept term
> $m$: the slope

- Them, use this `error_at` function to iterate through each of the points in the dataset, and at each iteration change our `update_to_b` by $2*\epsilon$ and change our `update_to_m` by $2*x*\epsilon$.

```python
# initial variables of our regression line
b_current = 0
m_current = 0

#amount to update our variables for our next step
update_to_b = 0
update_to_m = 0

# Define the error_at function
def error_at(point, b, m):
    return (point[0]- (m * point[1]  + b))

# iterate through data to change update_to_b and update_to_m
for i in range(0, len(data)):
    update_to_b += -2*(error_at(data[i], b_current, m_current))
    update_to_m += -2*(error_at(data[i], b_current, m_current))*data[i][1]

# Create new_b and new_m by subtracting the updates from the current estimates
new_b = b_current - update_to_b
new_m = m_current - update_to_m
```
In the last two lines of the code above, we calculate our `new_b` and `new_m` values by updating our taking our current values and adding our respective updates.  We define a function called `error_at`, which we can use in the error component of our partial derivatives above.

The code above represents **just one** update to our regression line, and therefore just one step towards our best fit line.  We'll just repeat the process to take multiple steps.  But first, we have to make a couple of other changes.

## Tweaking our approach

Ok, the above code is very close to what we want, but we just need to make tweaks to our code before it's perfect.

The first one is obvious if we think about what these formulas are really telling us to do.  Look at the graph below, and think about what it means to change each of our $m$ and $b$ variables by at least the sum of all of the errors, of the $y$ values that our regression line predicts and our actual data.  That would be an enormous change.  To ensure that we drastically updating our regression line with each step, we multiply each of these partial derivatives by a learning rate.  As we have seen before, the learning rate is just a small number, like $.
01$ which controls how large our updates to the regression line will be.  The learning rate is  represented by the Greek letter eta, $\eta$, or alpha $\alpha$.  We'll use eta, so $\eta = .01$ means the learning rate is $.01$.

Multiplying our step size by our learning rate works fine, so long as we multiply both of the partial derivatives by the same amount.  This is because without gradient,  $ \nabla J(m,b)$, we think of as steering us in the correct direction.  In other words, our derivatives ensure we are making the correct **proportional** changes to $m$ and $b$.  So scaling down these changes to make sure we don't update our regression line too quickly works fine, so long as we keep me moving in the correct direction.  While we're at it, we can also get rid of multiplying our partials by 2.  As mentioned, so long as our changes are proportional we're in good shape.

For our second tweak, note that in general the larger the dataset, the larger the sum of our errors would be.  But that doesn't mean our formulas are less accurate, and there deserve larger changes.  It just means that the total error is larger.  But we should really think accuracy as being proportional to the size of our dataset.  We can correct for this effect by dividing the effect of our update by the size of our dataset, $n$.

Make these changes below:

```python
#amount to update our variables for our next step
update_to_b = 0
update_to_m = 0

# define learning rate and n
learning_rate = .01
n = len(data)

# create update_to_b and update_to_m
for i in range(0, n):
    update_to_b += -(1/n)*(error_at(data[i], b_current, m_current))
    update_to_m += -(1/n)*(error_at(data[i], b_current, m_current)*data[i][0])

# create new_b and new_m
new_b = b_current - (learning_rate * update_to_b)
new_m = m_current - (learning_rate * update_to_m)
```
So our code now reflects what we know about our gradient descent process.  Start with an initial regression line with values of $m$ and $b$.  Then for each point, calculate how the regression line fares against the actual point (that is, find the error).  Update what the next step to the respective variable should be by using the partial derivative.  And after iterating through all of the points, update the value of $b$ and $m$ appropriately, scaled down by a learning rate.

## Seeing our gradient descent formulas in action

As mentioned earlier, the code above represents just one update to our regression line, and therefore just one step towards our best fit line.  To take multiple steps we wrap the process we want to duplicate in a function called `step_gradient` and then can call that function as much as we want. With this function:

- Include a learning_rate of 0.1
- Return a tuple of (b,m)
The parameters should be:
> b_current : the starting value of b
> m_current : the starting value of m
> points : the number of points at which we want to check our gradient

See if you can use your `error_at` function within the `step_gradient` function!

```python
def step_gradient(b_current, m_current, points):
    b_gradient = 0
    m_gradient = 0
    learning_rate = .1
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i][1]
        y = points[i][0]
        b_gradient += -(1/N) * (y - (m_current * x + b_current))
        m_gradient += -(1/N) * x * (y -  (m_current * x + b_current))
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return (new_b, new_m)
```
Now let's initialize `b` and `m` as 0 and run a first iteration of the `step_gradient` function.

```python
b = 0
m = 0
first_step = step_gradient(b, m, data) # {'b': 0.0085, 'm': 0.6249999999999999}

print(first_step[0])
print(first_step[1])
# b= 3.02503, m= 2.07286
```
So just looking at input and output, we begin by setting $b$ and $m$ to 0 and 0.  Then from our step_gradient function, we receive new values of $b$ and $m$ of 3.02503 and 2.0728.  Now what we need to do, is take another step in the correct direction by calling our step gradient function with our updated values of $b$ and $m$.

```python
updated_b = first_step[0]
updated_m = first_step[1]
step_gradient(updated_b, updated_m, data)
# b = 5.63489, m= 3.902265
```
Let's do this, say, 1000 times.

```python
# set our initial step with m and b values, and the corresponding error.
b = 0
m = 0
iterations = []
for i in range(1000):
    iteration = step_gradient(b, m, data)
    b = iteration[0]
    m = iteration[1]
    # update values of b and m
    iterations.append(iteration)
```
Let's take a look at the estimates in the last iteration.

```python
iterations[999]
```
As you can see, our  m  and  b  values both update with each step. Not only that, but with each step, the size of the changes to  m and  b  decrease. This is because they are approaching a best fit line.

## Let's include 2 predictors, $x_1$ and $x_2$

Below, we generated a problem where we have 2 predictors. We generated data such that the best fit line is around $\hat y = 3x_1 -4x_2 +2$, noting that there is random noise introduced, so the final result will never be exactly that. Let's build what we built previously, but now create a `step_gradient_multi` function that can take an *arbitrary* number of predictors (so the function should be able to include more than 2 predictors as well). Good luck!

```python
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(11)

x1 = np.random.rand(100,1).reshape(100)
x2 = np.random.rand(100,1).reshape(100)
y_randterm = np.random.normal(0,0.2,100)
y = 2+ 3* x1+ -4*x2 + y_randterm

data = np.array([y, x1, x2])
data = np.transpose(data)
```
```python
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
ax1.set_title('x_1')
ax1.plot(x1, y, '.b')
ax2.set_title('x_2')
ax2.plot(x2, y, '.b');
```
Note that, for our gradients, when having multiple predictors $x_j$ with $j \in 1,\ldots, k$

$$ \frac{dJ}{dm_j}J(m_j,b) = -2\sum_{i = 1}^n x_{j,i}(y_i - (\sum_{j=1}^km{x_{j,i}} + b)) = -2\sum_{i = 1}^n x_{j,i}*\epsilon_i$$
$$ \frac{dJ}{db}J(m_j,b) = -2\sum_{i = 1}^n(y_i - (\sum_{j=1}^km{x_{j,i}} + b)) = -2\sum_{i = 1}^n \epsilon_i $$


So we'll have one gradient per predictor along with the gradient for the intercept!

Create the `step_gradient_multi` function below. As we said before, this means that we have more than one feature that we are using as an independent variable in the regression. This function will have the same inputs as `step_gradient`, but it will be able to handle having more than one value for m. It should return the final values for b and m in the form of a tuple.

- `b_current` refers to the y-intercept at the current step
- `m_current` refers to the slope at the current step
- `points` are the data points to which we want to fit a line

You might have to refactor your `error` at function if you want to use it with multiple m values.

```python
def step_gradient_multi(b_current, m_current ,points):
    b_gradient = 0
    m_gradient = np.zeros(len(m_current))
    learning_rate = .1
    N = float(len(points))
    for i in range(0, len(points)):
        y = points[i][0]
        x = points[i][1:(len(m_current)+1)]
        b_gradient += -(1/N)  * (y -  (sum(m_current * x) + b_current))
        m_gradient += -(1/N) * x * (y -  (sum(m_current * x) + b_current))
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return (new_b, new_m)
```
Apply 1 step to our data

```python
b = [0, 0]
m = [0,0]
updated_b, updated_m = step_gradient_multi(b, m, data) # {'b': 0.0085, 'm': 0.6249999999999999}
updated_b, updated_m
```
Apply 500 steps to our data

```python
# set our initial step with m and b values, and the corresponding error.
b = [0, 0]
m = [0,0]
iterations = []
for i in range(500):
    iteration = step_gradient_multi(b, m, data)
    b= iteration[0]
    m = []
    for j in range(len(iteration)):
        m.append(iteration[1][j])
    iterations.append(iteration)
```
Look at the last step

```python
iterations[499]
```
## Level up - optional

Try your own gradient descent algorithm on the Boston Housing data set, and compare with the result from scikit learn!
Be careful to test on a few continuous variables at first, and see how you perform. Scikit learn has built-in "regularization" parameters to make optimization more feasible for many parameters.

## Summary

In this section, we saw our gradient descent formulas in action.  The core of the gradient descent functions is understanding the two lines:

$$ \frac{dJ}{dm}J(m,b) = -2\sum_{i = 1}^n x(y_i - (mx_i + b)) = -2\sum_{i = 1}^n x_i*\epsilon_i$$
$$ \frac{dJ}{db}J(m,b) = -2\sum_{i = 1}^n(y_i - (mx_i + b)) = -2\sum_{i = 1}^n \epsilon_i $$

Which both look to the errors of the current regression line for our dataset to determine how to update the regression line next.  These formulas came from our cost function, $J(m,b) = \sum_{i = 1}^n(y_i - (mx_i + b))^2 $, and using the gradient to find the direction of steepest descent.  Translating this into code, and seeing how the regression line continued to improve in alignment with the data, we saw the effectiveness of this technique in practice. Additionally, we saw how you can extend the gradient descent algorithm to multiple predictors.


-----File-Boundary-----
