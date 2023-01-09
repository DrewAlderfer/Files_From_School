# Classes and Instances - Lab

## Introduction

Okay, you've learned how to declare classes and create instances in the last lesson. Now it's time to put these new skills to the test!

## Objectives

In this lab you will:

* Create an instance of a class

## Classes


You're about to create your first package with class definitions! You've already seen how to import packages such as NumPy and Pandas, and you can organize your own code in a similar manner. For example, once you define the `Ride` class in a file **ride.py**, you can then import said code in another notebook or script using:

```python
# Import the entire file
import ride

# Import only the Ride class
from ride import Ride
```
In addition to **ride.py** file, we also created another file **driver.py** that contains the `Driver` class. Import this class in the cell below:

```python
# Import only the Driver class
from driver import Driver
```
Create a `Passenger` class that doesn't contain anything in the following cell:

> Note: By convention, you should use CamelCase to name the class. Also, you can't create an "empty" class. At the least, you need to specify the pass keyword to ensure the class definition is syntactically valid.

```python
# Create Passenger class
class Passenger:
    pass
```
## Instances

Now practice using these classes to create instances. First, make two instances of the `Passenger` class and assign them to the variables `meryl` and `daniel`, respectively:

```python
# Two instances of the Passenger class
meryl = Passenger()
daniel = Passenger()

print(meryl)
print(daniel)
```
Next, make one instance of the `Driver` class and assign it to the variable, `flatiron_taxi`.

```python
# One instance of the Driver class
flatiron_taxi = Driver()
print(flatiron_taxi)
```
Finally, make two instances of the `Ride` class and assign them to `ride_to_school` and `ride_home`.

```python
# Two instances of the Ride class
ride_to_school = Ride()
ride_home = Ride()

print(ride_to_school)
print(ride_home)
```
## Summary
Great! In this lab, you were able to define classes and create instances of those classes.


-----File-Boundary-----
# Instance Methods - Lab

## Introduction
In the last lesson, you learned about instance methods -- what they are and how to define them. In this lab, you are going to flesh out the `Driver` and `Passenger` classes by writing your own instance methods for these classes.

## Objectives

In this lab you will:

* Create an instance of a class
* Define and call an instance method

## Define classes and instance methods

You will now define classes and associated instance methods in the cell below:

> **Remember:** *as we learned in the previous lesson, we need to define our instance methods with at least one argument (`self`) in order to call them on an instance object.*

Define a class `Driver` with two instance methods:

- `greeting`: this should return the string `"Hey, how are you?"`
- `ask_for_destination`: this should return the string `"Where would you like to go today?"`

```python
# Define Driver class here
class Driver:

    def greeting(self):
        return "Hey, how are you?"

    def ask_for_destination(self):
        return "Where would you like to go today?"
```
Define a class `Passenger` with two instance methods:

- `reply_greeting`: this should return the string `"I am doing well!"`
- `in_a_hurry`: this should return the string `"Punch it! They're on our tail!"`

```python
# Define Passenger class here
class Passenger:

    def reply_greeting(self):
        return "I am doing well! Thanks for picking me up today!"

    def in_a_hurry(self):
        return "Punch it! They're on our tail!"
```
## Instantiate classes and methods

Great! You've now defined classes and the associated instance methods. You will now actually use them:

Start by instantiating a driver and a passenger. Assign the driver to the variable `daniel` and assign the passenger to `niky`.

```python
daniel = Driver() # driver
niky = Passenger() # passenger
```
Alright, you have the passengers and drivers! Now you need to put those instance methods to use. Try them out and assign the return values to the variables below.

- Have `daniel` greet his passenger, who is going to be `niky`. Assign the greeting to the variable `polite_greeting`
- Have `niky` respond by calling `in_a_hurry()`, and assign the return value to the variable, `no_time_to_talk`

```python
polite_greeting = daniel.greeting()
print(polite_greeting)
```
```python
no_time_to_talk = niky.in_a_hurry()
print(no_time_to_talk)
```
## Feel like doing more?

In the cells below, you'll create three different classes that represent animals in a zoo -- lions, tigers, and elephants. Each animal should have a method, `speak()`, which returns a string containing the sound they make (feel free to have some fun with this -- we don't know how to spell the sound an elephant makes any better than you do!).

```python
# Create Lion class
class Lion:

    def speak(self):
        return "Roar"
```
```python
# Create Tiger class
class Tiger:

    def speak(self):
        return "Meow"
```
```python
# Create Elephant class
class Elephant:

    def speak(self):
        return "woo-I'm-an-elephant!"
```
Now, in the cell below, create an instance of each animal:

```python
simba = Lion()
tony = Tiger()
dumbo = Elephant()
```
Now, add each of them into the list `zoo` in the cell below:

```python
zoo = [simba, tony, dumbo]
```
Now, loop through the `zoo` list and call out the `.speak()` method for every animal in the zoo. Make sure you print this in order to see the output!

```python
for animal in zoo:
    print(animal.speak())
```
## Summary
In this lab, you practiced defining classes and instance methods. You then instantiated instances of your classes and used them to practice calling your instance methods.


-----File-Boundary-----
# Object Attributes - Lab

## Introduction
In this lab, you'll practice defining classes and instance methods.

## Objectives

You will be able to:

* Define and call an instance method
* Define and access instance attributes

## Defining Classes and Instance Methods

In the cell below define a `Driver` class.

For this class, create a method called `greet_passenger()`, which returns the string `Hello! I'll be your driver today. My name is ` followed by that driver's first name and last name (i.e. `Hello! I'll be your driver today. My name is John Doe`). (Be sure to keep in mind that the driver's name will be stored under two separate attributes: first and last.)

```python
# Define Driver Class here with properties for each instance variable
class Driver():
    def greet_passenger(self):
        print("Hello! I'll be your driver today. My name is {} {}".format(self.first, self.last))
```
Great! Now create an instance of your driver class. Then, create the following attributes for your instance:
* `first` - the first name of the driver. Set it to Matthew.
* `last` - the last name of the driver. Set it to Mitchell.
* `miles_driven` - the number of miles driven by the driver. Set it to 100.
* `rating` - the driver's rating. Set it to 4.9

Finally, use your `greet_passenger()` method for your `Driver` instance object.

```python
driver = Driver()
driver.first = "Matthew"
driver.last = "Mitchell"
driver.miles_driven = 100
driver.rating = 4.9
driver.greet_passenger() # Hello! I'll be your driver today. My name is Matthew Mitchell
```
Now, create a passenger class with one method `yell_name()` which prints the passenger's first and last name in all caps. (Again first and last will be stored as separate attributes.)

```python
# Define Passenger Class here with properties for each instance variable
class Passenger():
    def yell_name(self):
        print("{} {}".format(self.first.upper(), self.last.upper()))
```
Create an instance of your passenger class. Then create an attribute "first" set to "Ron" and an attribute "last" set to "Burgundy". Then call the `yell_name()` method.

```python
passenger = Passenger()
passenger.first = "Ron"
passenger.last = "Burgundy"
passenger.yell_name() # "RON BURGUNDY"
```
Great work!

## Summary
In this lab, you practiced defining classes, creating instances of said classes, and using methods that made calls to object attributes.


-----File-Boundary-----
# Object Initialization - Lab

## Introduction
In this lab, you'll practice defining classes with `__init__` methods. You'll define two classes, `Driver` and `Passenger` in the cells below.

## Objectives

In this lab you will:

* Create instance variables in the `__init__` method
* Use default arguments in the `__init__` method

## Initializing Instance Objects

Start off by defining the `Driver` class, similar to as you've done before. This time, define an `__init__` method that initializes a driver with the attributes `first`, `last`, and `occupation` for their first name, last name, and occupation. Provide a default argument of `"driver"` for `occupation`.

```python
# Define Driver Class Here
class Driver():

    def __init__(self, first, last, occupation = 'driver'):
        self.first = first
        self.last = last
        self.occupation = occupation
```
Now, initialize a driver with the first name `"Dale"` and last name `"Earnhardt"`.

<img src="images/dale.gif" width="500">

gif from [Nascar](https://giphy.com/nascar)

```python
dale_earnhardt = Driver('Dale', 'Earnhardt') # Initialize Dale Earnhardt here
print(dale_earnhardt.first) # "Dale"
print(dale_earnhardt.last) # "Earnhardt"
print(dale_earnhardt.occupation) # "driving"
```
Next, define the `Passenger` class. Using the `__init__` method, ensure all instances contain the attributes `first`, `last`, `email`, and `rides_taken` for their first name, last name, email, and number of rides they have taken. Provide the `__init__` method with the default argument of `0` for the `rides_taken` attribute since new passengers should not have taken any rides.

```python
# Define Passenger class here
class Passenger():

    def __init__(self, first, last, email, rides_taken=0):
        self.first = first
        self.last = last
        self.email = email
        self.rides_taken = rides_taken
```
Now that you've defined a `Passenger` class, check it out by initializing a new passenger with the first name `"Jerry"`, the last name `"Seinfeld"`, and the email `"jerry.seinfeld@mailinator.com"`.

```python
jerry = Passenger('Jerry', 'Seinfeld', 'jerry.seinfeld@mailinator.com') # Initialize Mr. Seinfeld here
print(jerry.first) # "Jerry"
print(jerry.last) # "Seinfeld"
print(jerry.email) # "jerry.seinfeld@mailinator.com"
print(jerry.rides_taken) # 0
```
Great work! Mr. Seinfeld is now in the system and ready to request a ride!

## Summary

In this lab, you defined `__init__` methods that allowed you to initialize new instances with a set of predetermined attributes and default attributes.


-----File-Boundary-----
# Inheritance - Lab

## Introduction

In this lab, you'll use what you've learned about inheritance to model a zoo using superclasses, subclasses, and maybe even an abstract superclass!

## Objectives

In this lab you will:

- Create a domain model using OOP
- Use inheritance to write nonredundant code

## Modeling a Zoo

Consider the following scenario:  You've been hired by a zookeeper to build a program that keeps track of all the animals in the zoo.  This is a great opportunity to make use of inheritance and object-oriented programming!

## Creating an Abstract Superclass

Start by creating an abstract superclass, `Animal()`.  When your program is complete, all subclasses of `Animal()` will have the following attributes:

* `name`, which is a string set at instantation time
* `size`, which can be `'small'`, `'medium'`, `'large'`, or `'enormous'`
* `weight`, which is an integer set at instantiation time
* `species`, a string that tells us the species of the animal
* `food_type`, which can be `'herbivore'`, `'carnivore'`, or `'omnivore'`
* `nocturnal`, a boolean value that is `True` if the animal sleeps during the day, otherwise `False`

They'll also have the following behaviors:

* `sleep`, which prints a string saying if the animal sleeps during day or night
* `eat`, which takes in the string `'plants'` or `'meat'`, and returns `'{animal name} the {animal species} thinks {food} is yummy!'` or `'I don't eat this!'` based on the animal's `food_type` attribute

In the cell below, create an abstract superclass that meets these specifications.

**_NOTE:_** For some attributes in an abstract superclass such as `size`, the initial value doesn't matter -- just make sure that you remember to override it in each of the subclasses!

```python
class Animal(object):

    def __init__(self, name, weight):
        self.name = name
        self.weight = weight
        self.species = None
        self.size = None
        self.food_type = None
        self.nocturnal = False

    def sleep(self):
        if self.nocturnal:
            print("{} sleeps during the day!".format(self.name))
        else:
            print("{} sleeps during the night!".format(self.name))

    def eat(self, food):
        if self.food_type == 'omnivore':
            print("{} the {} thinks {} is Yummy!".format(self.name, self.species, food))
        elif (food == 'meat' and self.food_type == "carnivore") or (food == 'plants' and self.food_type == 'herbivore'):
            print("{} the {} thinks {} is Yummy!".format(self.name, self.species, food))
        else:
            print("I don't eat this!")
```
Great! Now that you have our abstract superclass, you can begin building out the specific animal classes.

In the cell below, complete the `Elephant()` class.  This class should:

* subclass `Animal`
* have a species of `'elephant'`
* have a size of `'enormous'`
* have a food type of `'herbivore'`
* set nocturnal to `False`

**_Hint:_** Remember to make use of `.super()` during initialization, and be sure to pass in the values it expects at instantiation time!

```python
class Elephant(Animal):

    def __init__(self, name, weight):
        super().__init__(name, weight)
        self.size = 'enormous'
        self.species = 'elephant'
        self.food_type = 'herbivore'
        self.nocturnal = False
```
Great! Now, in the cell below, create a `Tiger()` class.  This class should:

* subclass `Animal`
* have a species of `'tiger'`
* have a size of `'large'`
* have a food type of `'carnivore'`
* set nocturnal to `True`

```python
class Tiger(Animal):

    def __init__(self, name, weight):
        super().__init__(name, weight)
        self.size = 'large'
        self.species = 'tiger'
        self.food_type = 'carnivore'
        self.nocturnal = True
```
Great! Two more classes to go. In the cell below, create a `Raccoon()` class. This class should:

* subclass `Animal`
* have a species of `raccoon`
* have a size of `'small'`
* have a food type of `'omnivore'`
* set nocturnal to `True`

```python
class Raccoon(Animal):

    def __init__(self, name, weight):
        super().__init__(name, weight)
        self.size = 'small'
        self.species = 'raccoon'
        self.food_type = 'omnivore'
        self.nocturnal = True
```
Finally, create a `Gorilla()` class. This class should:

* subclass `Animal`
* have a species of `gorilla`
* have a size of `'large'`
* have a food type of `'herbivore'`
* set nocturnal to `False`

```python
class Gorilla(Animal):

    def __init__(self, name, weight):
        super().__init__(name, weight)
        self.size = 'large'
        self.species = 'gorilla'
        self.food_type = 'herbivore'
        self.nocturnal = False
```
## Using Our Objects

Now it's time to populate the zoo! To ease the creation of animal instances, create a function `add_animal_to_zoo()`.

This function should take in the following parameters:

* `zoo`, an array representing the current state of the zoo
* `animal_type`, a string.  Can be `'Gorilla'`, `'Raccoon'`, `'Tiger'`, or `'Elephant'`
* `name`, the name of the animal being created
* `weight`, the weight of the animal being created

The function should then:

* use `animal_type` to determine which object to create
* Create an instance of that animal, passing in the `name` and `weight`
* Append the newly created animal to `zoo`
* Return `zoo`

```python
def add_animal_to_zoo(zoo, animal_type, name, weight):
    animal = None
    if animal_type == 'Gorilla':
        animal = Gorilla(name, weight)
    elif animal_type == 'Raccoon':
        animal = Raccoon(name, weight)
    elif animal_type == 'Tiger':
        animal = Tiger(name, weight)
    else:
        animal = Elephant(name, weight)

    zoo.append(animal)

    return zoo
```
Great! Now, add some animals to your zoo.

Create the following animals and add them to your zoo.  The names and weights are up to you.

* 2 Elephants
* 2 Raccons
* 1 Gorilla
* 3 Tigers

```python
to_create = ['Elephant', 'Elephant', 'Raccoon', 'Raccoon', 'Gorilla', 'Tiger', 'Tiger', 'Tiger']

zoo = []

for i in to_create:
    zoo = add_animal_to_zoo(zoo, i, 'name', 100)

zoo
```
Great! Now that you have a populated zoo, you can do what the zookeeper hired you to do -- write a program that feeds the correct animals the right food at the right times!

To do this, write a function called `feed_animals()`. This function should take in two arguments:

* `zoo`, the zoo array containing all the animals
* `time`, which can be `'Day'` or `'Night'`.  This should default to day if nothing is entered for `time`

This function should:

* Feed only the non-nocturnal animals if `time='Day'`, or only the nocturnal animals if `time='Night'`
* Check the food type of each animal before feeding.  If the animal is a carnivore, feed it `'meat'`; otherwise, feed it `'plants'`. Feed the animals by using their `.eat()` method

```python
def feed_animals(zoo, time='Day'):
    for animal in zoo:
        if time == 'Day':
            # CASE: Daytime feeding -- Only feed the animals that aren't nocturnal
            if animal.nocturnal == False:
                # If the animal is a carnivore, feed it "meat".  Otherwise, feed it "plants"
                if animal.food_type == 'carnivore':
                    animal.eat('meat')
                else:
                    animal.eat('plants')
        else:
            # CASE: Night-time feeding -- feed only the nocturnal animals!
            if animal.nocturnal == True:
                if animal.food_type == 'carnivore':
                    animal.eat('meat')
                else:
                    animal.eat('plants')
```
Now, test out your program.  Call the function for a daytime feeding below.

```python
feed_animals(zoo)
```
If the elephants and gorrillas were fed then things should be good!

In the cell below, call `feed_animals()` again, but this time set `time='Night'`

```python
feed_animals(zoo, 'Night')
```
That's it! You've used OOP and inheritance to build a working program to help the zookeeper feed his animals with right food at the correct times!

## Summary

In this lab, you modeled a zoo and learned how to use inheritance to write nonredundant code, used subclasses and superclasses, and create a domain model using OOP.


-----File-Boundary-----
# OOP with Scikit-Learn - Lab

## Introduction

Now that you have learned some of the basics of object-oriented programming with scikit-learn, let's practice applying it!

## Objectives:

In this lesson you will practice:

* Recall the distinction between mutable and immutable types
* Define the four main inherited object types in scikit-learn
* Instantiate scikit-learn transformers and models
* Invoke scikit-learn methods
* Access scikit-learn attributes

## Mutable and Immutable Types

For each example below, think to yourself whether it is a mutable or immutable type. Then expand the details tag to reveal the answer.

<ol>
    <li>
        <details>
            <summary style="cursor: pointer">Python dictionary (click to reveal)</summary>
            <p><strong>Mutable.</strong> For example, the `update` method can be used to modify values within a dictionary.</p>
            <p></p>
        </details>
    </li>
    <li>
        <details>
            <summary style="cursor: pointer">Python tuple (click to reveal)</summary>
            <p><strong>Immutable.</strong> If you want to create a modified version of a tuple, you need to use <code>=</code> to reassign it.</p>
            <p></p>
        </details>
    </li>
    <li>
        <details>
            <summary style="cursor: pointer">pandas <code>DataFrame</code> (click to reveal)</summary>
            <p><strong>Mutable.</strong> Using the <code>inplace=True</code> argument with various different methods allows you to modify a dataframe in place.</p>
            <p></p>
        </details>
    </li>
    <li>
        <details>
            <summary style="cursor: pointer">scikit-learn <code>OneHotEncoder</code> (click to reveal)</summary>
            <p><strong>Mutable.</strong> Calling the <code>fit</code> method causes the transformer to store information about the data that is passed in, modifying its internal attributes.</p>
            <p></p>
        </details>
    </li>
</ol>

## The Data

For this lab we'll use data from the built-in iris dataset:

```python
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True, as_frame=True)
```
```python
X
```
```python
y
```
## Scikit-Learn Classes

For the following exercises, follow the documentation link to understand the class you are working with, but **do not** worry about understanding the underlying algorithm. The goal is just to get used to creating and using these types of objects.

### Estimators

For all estimators, the steps are:

1. Import the class from the `sklearn` library
2. Instantiate an object from the class
3. Pass in the appropriate data to the `fit` method

#### `MinMaxScaler` ([documentation here](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html))

Import this scaler, instantiate an object called `scaler` with default parameters, and `fit` the scaler on `X`.

```python
# Import
from sklearn.preprocessing import MinMaxScaler
# Instantiate
scaler = MinMaxScaler()
# Fit
scaler.fit(X)
```
#### `DecisionTreeClassifier` ([documentation here](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html))

Import the classifier, instantiate an object called `clf` (short for "classifier") with default parameters, and `fit` the classifier on `X` and `y`.

```python
# Import
from sklearn.tree import DecisionTreeClassifier
# Instantiate
clf = DecisionTreeClassifier()
# Fit
clf.fit(X, y)
```
### Transformers

One of the two objects instantiated above (`scaler` or `clf`) is a transformer. Which one is it? Consult the documentation.

---

<details>
    <summary style="cursor: pointer">Hint (click to reveal)</summary>
    <p>The class with a <code>transform</code> method is a transformer.</p>
</details>

---

#### Using the transformer, print out two of the fitted attributes along with descriptions from the documentation.

---

<details>
    <summary style="cursor: pointer">Hint (click to reveal)</summary>
    <p>Attributes ending with <code>_</code> are fitted attributes.</p>
</details>

```python
# (Answers will vary)
print("Minimum feature seen in the data:", scaler.data_min_)
print("Maximum feature seen in the data:", scaler.data_max_)
```
#### Now, call the `transform` method on the transformer and pass in `X`. Assign the result to `X_scaled`

```python
X_scaled = scaler.transform(X)
```
### Predictors and Models

The other of the two scikit-learn objects instantiated above (`scaler` or `clf`) is a predictor and a model. Which one is it? Consult the documentation.

---

<details>
    <summary style="cursor: pointer">Hint (click to reveal)</summary>
    <p>The class with a <code>predict</code> method and a <code>score</code> method is a predictor and a model.</p>
</details>

---

#### Using the predictor, print out two of the fitted attributes along with descriptions from the documentation.

```python
# (Answers will vary)
print("Number of classes:", clf.n_classes_)
print("Number of features seen:", clf.n_features_in_)
```
#### Now, call the `predict` method on the predictor, passing in `X`. Assign the result to `y_pred`

```python
y_pred = clf.predict(X)
```
#### Now, call the `score` method on the predictor, passing in `X` and `y`

```python
clf.score(X, y)
```
#### What does that score represent? Write your answer below

```python
"""
According to the documentation, this score represents the mean accuracy
"""
```
## Summary

In this lab, you practiced identifying mutable and immutable types as well as identifying transformers, predictors, and models using scikit-learn. You also instantiated scikit-learn objects, invoked the most common scikit-learn methods, and accessed some scikit-learn attributes.


-----File-Boundary-----
