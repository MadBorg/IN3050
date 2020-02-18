# Supervised Learning

* Learn from labeled data
* Task: Assign new items to a class

## Learning algorithms

1. Nearest neighbors
2. Perceptron
3. Linear regression
4. Simple neural networks
5. Multi-layer neural networks
  * Backpropagation
* and more

## Aspects of the algorithms

* What is the undelying idea
* How are they implementent
* What are the usecases

## Today

* Classification - The basics
* The k Nearest neighbors algorithm (kNM)
* The bain and the neuron
* The Perceptron and the Perceptron learning algorithm

## what does it mean to learn

* A pc is smart when a human does somethin and is considered smart and the pc can do the same. Not including rememberin a article.
* Learn to handle unseen data

## Supervised Learning

* Structure
  * Observations, O (possible inputs)
  * Set of target values, T(possible output values)
* Goal: Determine a mapping
* Training: To learn a mapping gamma, trained on the training set.
* Two types
  * Classification
    * Right or wrong
  * Regression

## Classification

* Looking for similarities, to figure out where the new one goes.
* WE need to
  * describe attributes of the objects
  * similarities
* For this we introduce features

## Features

* We decide some features
* Size, shape, height, with, weight

## the larger picture

* in real life, you want to apply ML to new tasks, then there is a lot of work befor you have datasets like that:
  * Data collection and Prep
  * Features Selection and extraction
* In this course we will use mainly predone datasets

## From data set to features

* On some cases we need to transformoe and scale our data

## Learing problem

* Improve over task, T
* with respect to performanc measure, P
* based on experience, E

## Training

* Msures of fit: accuresity
* Predict labels
* For repeated development we need aleast two test sets
* Thety might fit to well, so wee need to keep a testset clean of use

## k Nearset neighbors - kNM

* they who resemble you most, you will trust.

1. Calculate the distance to all the training instances
2. Pick the k nearest ones
3. Choose the majority class for this set

* No Training
* Fast
* Slow to predict, since you have to do everything all ower.
* One parameter, k - number of nearest neighbors
* Danger of overfitting with small k
* res is dependent on the scale of the diffferent parameters


## Perception

* 10^11 Neurons
* 10^14 Synapses
* 10-3 Seconds clock time.
  * Computer: 1 GHz = 10^-9 Seconds

## Neuron

* Axon
  * Transports signals
* Dendrites
  * Receive signals
* Soma (cell body)
  * "Sums" the signals from the Dendrites
  * and action potential is ent down the axon, the cell "spikes" or "fiers"

## Hebs Rule

* If there is sent a msg, then will often the connection strenghten.

## The bias term

* Theta - decied such as res is as gained from the training data
* A threshold
