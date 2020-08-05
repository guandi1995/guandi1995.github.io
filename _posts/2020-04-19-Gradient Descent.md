---
title: "Gradient Descent"
date: 2020-04-19
tags: [machine learning, gradient descent]

excerpt: "machine learning"
mathjax: "true"
---

<img src="{{ site.url }}{{ site.baseurl }}/images/loss_function/header_image.jpg" alt="">

In the previous posts, we discuss the loss function in terms of logistic regression and SVM algorithm. We know that the loss function is a function in terms of parameters $$w$$ and $$b$$. Now the question is how we find these parameters in order to achieve the minimum of loss function. It brings us to the topics of optimization, which refers to the task of minimizing the objective/loss function parameterized by $$w$$ and $$b$$.

Optimization algorithms have the following goals:

- find the global minimum of the objective function. This is feasible only if the function is a convex function since the local minimum is the global minimum in convex function.

- find the lowest possible value of the objective function within its neighbor. That is usually the case if the function is non-convex which is the most case in deep learning problems.

One of the most common optimization algorithm in deep learning is gradient descent. It takes the first derivative when performing the updates on the parameters. On each iteration, we update the parameters in the opposite direction of the gradient of the loss function $$J(w,b)$$. The size of the step we take on each iteration is determined by the learning rate $$\alpha$$. We follow the direction of the slope downhill until it reaches the minimum. The formula of the above description can be demonstrated as $$w= w-\alpha \dfrac{dJ(w,b)}{dw}$$ and $$b= b-\alpha \dfrac{dJ(w,b)}{db}$$. 
