---
title: "Types of Loss Function"
date: 2020-04-16
tags: [machine learning, data science, loss function]

excerpt: "Machine Learning, Data Science, Loss Function"
mathjax: "true"
---

Neural Networks learns to map a set of inputs to a set of outputs from training data. Given the training data, we usually calculate the weights for a neural network, but it is impossible to obtain the perfect weights. The approach to solve this is by transforming the problem to optimization algorithm, an algorithm that is used to navigate the space of possible sets of weights the model may use in order to make good predictions.

But in the context of an optimization algorithm, the function used to evaluate a candidate solution is referred to as the objective function. Objective function is the function we want to minimize or maximize. In the context of machine learning or deep learning, we always want to minimize the function. That is why objective function is considered as cost function or loss function.

**Machine learning and deep learning is to learn by means of a loss function.** In calculating the error of the model during optimization process, a loss function must be chosen. However, choosing loss functions is challenging because it is problem dependent. Therefore, it is important that the chosen loss function faithfully represent our design models based on the properties of the problem.

### Types of Loss Function

There are many types of loss function and there is no such one-size-fits-all loss function to algorithms in machine learning. Typically it is categorized into 3 types.

- **Regression loss functions**
  - Mean Square Error (MSE)/L2 loss/least square
  - Mean Absolute Error/L1 loss
  - Mean Square Logarithmic Error
- **Binary classification loss functions**
  - Binary Cross Entropy/Negative Log Likelihood
  - Hinge Loss/ SVM
  - Squared Hinge Loss
- **Multi-class classification loss functions**
  - Multi-class SVM
  - Multi-class Cross Entropy
  - Softmax Function/Multi-nomial logistic regression
  - Sparse Multi-class Cross Entropy Loss
  - Kullback Leibler Divergence Loss
