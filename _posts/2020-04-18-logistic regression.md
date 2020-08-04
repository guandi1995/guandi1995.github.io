---
title: "Logistic Regression"
date: 2020-04-18
tags: [machine learning]

excerpt: "machine learning, binary classification, logistic regression, logistic funtion"
mathjax: "true"
---
<img src="{{ site.url }}{{ site.baseurl }}/images/logistic regression/header_img.png" alt="">


Logistic regression is a technique in machine learning and is used to deal with the binary classification problem in supervised learning where the output of this type of problem has two-class value, i.e either 0 or 1. It is named for the function it used, which is logistic function or sigmoid function.

Consider a binary classification problem, given the input $$x$$, we want the estimated output that indicates which class the input is. Generally, we could estimate the probability of the output to between 0 or 1 and then set up a threshold that can help us decide its estimated class, which is also the estimated output. Therefore, denote the probability of the estimated output as $$\hat{y}$$ and let $$ \hat{y} = p(y~given~x) $$. Obviously, the value of $$\hat{y}$$ is supposed to be between 0 and 1, i.e. $$0 \le \hat{y} \le 1$$. On the other hand, our hypothesis model is $$z=wx+b$$ so as to estimate the output by using parameters weight $$w$$ and bias $$b$$ where $$w\in R^n, b\in R$$. Clearly the value of $$z$$ is not the probability of chance that is between 0 and 1. Therefore, in order to generate the output in the form of probability, we need transform $$z$$ to $$\hat{y}$$ by using the logistic function or sigmoid function.


### Logistic/Sigmoid Function
Instead of fitting a straight line or hyperplane, the logistic regression model uses the logistic function to squeeze the output of a linear operation equation between 0 and 1. It is defined as below:

$$
\sigma(z)=\dfrac{1}{1+e^{-z}}
$$

where $$\sigma(.)$$ is the logistic function notation.

The graph of sigmoid function is illustrated below:

<img src="{{ site.url }}{{ site.baseurl }}/images/logistic regression/sigmoid_function.jfif" alt="">

When $$z$$ is positively large, the value of $$\sigma(z)$$ converges to 1 while $$z$$ is negatively small, the value of $$\sigma(z)$$ converges to 0.

### Logistic Regression Cost Function
Unlike the cost function in linear regression, logistic regression does not apply the mean square error as its cost function since it will become a non-convex function, which is impossible to find the global minimum. In order to be a convex function and reach the global minimum when conducting the gradient descent, instead, the loss function for a single sample in the training dataset that the logistic regression applies is shown below:

$$
L(\hat{y}^{(i)},y^{(i)})=-[y^{(i)}~log(y^{(i)})+(1-y^{(i)})log(1-\hat{y}^{(i)})]
$$

If $$y=1$$, then the loss function becomes $$L(\hat{y}^{(i)},y^{(i)})=-log(\hat{y}^{(i)})$$. We want the loss function as small as possible to predict accurately. Consequently, we want the term $$-log(\hat{y}^{(i)})$$ as small as possible, but keep it in mind that the loss is always non-negative, that is, we want $$log(\hat{y}^{(i)})$$ large and $$\hat{y}$$ large as well. Another thing is that the value of $$\hat{y}$$ is between 0 and 1; therefore, we want $$\hat{y}$$ as close as to 1. In other words, the value of $$z$$ is supposed to be as large as possible due to the relationship between $$z$$ and $$\hat{y}$$ aforementioned. 
