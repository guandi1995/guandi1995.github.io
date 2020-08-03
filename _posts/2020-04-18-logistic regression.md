---
title: "Logistic Regression"
date: 2020-04-18
tags: [machine learning]

excerpt: "machine learning, binary classification, logistic regression, logistic funtion"
mathjax: "true"
---
<img src="{{ site.url }}{{ site.baseurl }}/images/logistic regression/header_img.png" alt="">


Logistic regression is a technique in machine learning and is used to deal with the binary classification problem in supervised learning where the output of this type of problem has two-class value, i.e either 0 or 1. It is named for the function it used, which is logistic function or sigmoid function.

Consider a binary classification problem, given the input $$x$$, we want the estimated output that indicates which class the input is. Generally, we would find out the probability of the output to be 0 or 1 and then set up an threshold that can help us decide its estimated class. Therefore, we could denote the probability of the estimated output as $$\hat{y}$$ and thus $$\hat{y}=P(y|X)$$. Obviously, the value of $$\hat{y}$$ will be between 0 and 1, i.e. $$0 \le \hat{y} \le 1$$.

<!-- On the other hand, our hypothesis model is $$z=wx+b$$ so as to estimate the output by using weight $$w$$ and bias $$b$$. Clearly the value of $$z$$ is not the probability of chance between 0 and 1. Therefore, in order to generate the output in the form of probability, we could transform $$z$$ to $$\hat{y}$$ by using the logistic function or sigmoid function. -->


### Sigmoid Function

So what is sigmoid function?
