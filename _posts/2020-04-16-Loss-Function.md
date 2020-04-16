---
title: "Types of Loss Function"
date: 2020-04-16
tags: [machine learning, data science, loss function]

excerpt: "Machine Learning, Data Science, Loss Function"
mathjax: "true"
---
<img src="{{ site.url }}{{ site.baseurl }}/images/loss_function/header_image.jpg" alt="">

Neural Networks learns to map a set of inputs to a set of outputs from training data. Given the training data, we usually calculate the weights for a neural network, but it is impossible to obtain the perfect weights. The approach to solve this is by transforming the problem to optimization algorithm, an algorithm that is used to navigate the space of possible sets of weights the model may use in order to make good predictions.

But in the context of an optimization algorithm, the function used to evaluate a candidate solution is referred to as the objective function. Objective function is the function we want to minimize or maximize. In the context of machine learning or deep learning, we always want to minimize the function. That is why **objective function is also called as cost function or loss function**.

**Machine learning and deep learning is to learn by means of a loss function.** With the help of some optimization function, loss functions learns to reduce the error in prediction. In calculating the error of the model during optimization process, a loss function must be chosen. However, choosing loss functions is challenging because it is problem dependent. Therefore, it is important that the chosen loss function faithfully represent our design models based on the properties of the problem.

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

Take CIFAR-10 as our example, intuitively, we can determine which class a specific image is belonged to by simply visualizing the score for each class, $$f(x_i;W,b)$$. But in order to let computer make decision and justify how good or bad the score is, we need to introduce loss functions for any machine learning problem.

**To justify how good or bad the score gives us to determine the class of the image, it turns out loss function can help us accomplish this by not simply visualizing and comparing the score vectors.**

A loss function tells us how good our current classifier is. Given a dataset of examples, $${(x_i,y_i)},i=1,..,n$$, where $$x_i$$ is vector of all pixel values of a image and $$y_i$$ is its corresponding label. Generalized loss function typically is a sum of loss over examples no matter what the classification problems are, loss function is generally defined as following:

  $$
  L=\dfrac{1}{n}\sum_{i=1}^n L_i[f(x_i;\theta),y_i]
  $$

The ultimate goal of machine learning is to find the argument $$\theta^*$$ that minimizes the loss function, in the case of linear classifier, the parameter are $$W^*,b^*$$.  

### Regularization

Next thing we need address is regularization, the term that always comes along with loss function. **Regularization is a technique that helps you avoid to create a complex and flexible model, which may lead to overfit the train data.** It will be addressed in more detail in the following posts, now, we just need to realize its existence. Because of regularization, our general loss function is defined as following:
$$
L=\dfrac{1}{n}\sum_{i=1}^nL_i+\lambda R(W)^2
$$
where $$W$$ is weights matrix and $$\lambda>0$$, $$R(W)$$ is the form of regularization.

For the example of CIFAR-10, since it is a multi-class classification problem, we will focus on two important loss functions, SVM and Cross Entropy loss, in detail in the next two posts.
