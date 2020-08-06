---
title: "Optimization-Gradient Descent and Its Variants"
date: 2020-04-19
tags: [machine learning, optimization, gradient descent]

excerpt: "machine learning"
mathjax: "true"
---

<img src="{{ site.url }}{{ site.baseurl }}/images/loss_function/header_image.jpg" alt="">

In the previous posts, we discuss the loss function in terms of logistic regression and SVM algorithms. We know that the loss function is a function in terms of parameters such as $$w$$ and $$b$$. Now the question is how we find these parameters in order to achieve the minimum of loss function. It brings us to the topics of optimization, which refers to the task of minimizing the objective/loss function parameterized by $$w$$ and $$b$$.

Optimization algorithms have the following goals:

- find the global minimum of the objective function. This is feasible only if the function is a convex function since the local minimum is the global minimum in convex function.

- find the lowest possible value of the objective function within its neighbor. That is usually the case if the function is nonconvex which is the most case in deep learning problems since most of loss functions in deep learning are nonconvex.

In this post, we summarize three common optimization algorithms, which are gradient descent, mini-batch gradient descent and stochastic gradient descent.

### Gradient Descent

One of the most common optimization algorithm in deep learning is gradient descent. It takes the first derivative when performing the updates on the parameters. On each iteration, we update the parameters in the opposite direction of the gradient of the loss function $$J(w,b)$$. The size of the step we take on each iteration is determined by the learning rate $$\alpha$$. We follow the direction of the slope downhill until it reaches the minimum. The formula of the above description can be demonstrated as $$w= w-\alpha \dfrac{dJ(w,b)}{dw}$$ and $$b= b-\alpha \dfrac{dJ(w,b)}{db}$$.

It is quite intuitive and simple if we take an example in one-dimensional space. Consider a loss function parameterized by $$w$$ drawn below, first we start from a random position on the loss function (i.e. $$w=4$$) and then find the gradient (first derivative) with respect to that position (i.e. $$\dfrac{dJ}{dw}(w=4)=3$$). To find the minimum of this loss function where the first derivative value at the minimum is zero, we need make the value of parameter $$w$$ smaller since the derivative value at $$w=4$$ is positive. Therefore, we should go to the opposite direction to find the minimum, which is also why there is a minus sign in the above formula. Same case approves the existence of the minus sign when $$w=1$$ and the corresponding gradient is negative; thus, we need make the value $$w$$ larger in order to approach to the minimum point at $$w=2$$.

<img src="{{ site.url }}{{ site.baseurl }}/images/gradient descent/1-d_grad_example.PNG" alt="">

In summary, the general steps and notes for conducting gradient descent in deep learning are shown below:

**Step 1**: initialize weight $$w$$ and bias $$b$$ to any random numbers.

**Step 2**: select a learning rate $$\alpha$$ that determines how big the step would learn on each iteration. Note that if learning rate $$\alpha$$ is too small, it would take extremely large amount of time to converge and become computationally expensive for training while if $$\alpha$$ is too large, it may fail to converge and result in overshoot the minimum. Therefore, the choice of learning rate is quite importance and critical while training the deep neural networks. The relationship between the loss curve over epoch/iterations with different value of learning rate is illustrated below

<img src="{{ site.url }}{{ site.baseurl }}/images/gradient descent/learning_rate.PNG" alt="">

**Step 3**: normalize the data, otherwise the level curves (contours) would be narrow, which needs a longer time to converge. To illustrate it, we draw the contours for unnormalized and normalized data with respect to parameters $$w$$ and $$b$$. The contours demonstrate the value of loss when different values of $$w$$ and $$b$$ are taken. It is quite obvious that the unnormalized data is hard to converge compared to the normalized one. The way we normalize the data is quite straightforward, we could just normalize each sample data by using the formula $$\dfrac{x_i-\mu}{\sigma}$$.

<img src="{{ site.url }}{{ site.baseurl }}/images/gradient descent/normalized.PNG" alt="">

**Step 4**: update the parameters on each iteration. Using the equation described above to update parameters: $$w= w-\alpha \dfrac{dJ(w,b)}{dw}$$ and $$b= b-\alpha \dfrac{dJ(w,b)}{db}$$


For a m-sample training dataset, it might be quite challenging and time-consuming to conduct this regular type of gradient descent if $$m$$ is extremely large since for each iteration, we have to compute the gradient for all data samples just only for updating the parameters once on an iteration. Therefore, computing the gradient of the loss function over all data samples is not a wise way to reach, which leads us to another methods of gradient descent, which is mini-batch gradient descent.

### Mini-batch Gradient Descent
Instead of going over all examples, mini-batch gradient descent sums up over a smaller number of examples


### Stochastic Gradient Descent
