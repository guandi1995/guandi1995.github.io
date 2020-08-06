---
title: "Computational Graph "
date: 2020-04-19
tags: [machine learning, optimization, gradient descent]

excerpt: "machine learning"
mathjax: "true"
---

<img src="{{ site.url }}{{ site.baseurl }}/images/loss_function/header_image.jpg" alt="">

In the previous post, we address the topics of gradient descent in order to minimize the loss function. Now we have to get familiar to computing the analytic gradient for arbitrarily complex function by using the framework of computational graphs. We can apply computational graph to represent any function with forward pass and backward pass so that we could use the computational graph to compute the gradient or derivative of the loss function with respect to the parameters.

Let's start from a really simple example. Consider a loss function $$L$$ parameterized by $$a,b,c$$ where $$L(a,b,c)=3(a+bc)$$.
