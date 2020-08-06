---
title: "Computational Graph "
date: 2020-04-19
tags: [machine learning, optimization, gradient descent]

excerpt: "machine learning"
mathjax: "true"
---

<img src="{{ site.url }}{{ site.baseurl }}/images/loss_function/header_image.jpg" alt="">

In the previous post, we address the topics of gradient descent in order to minimize the loss function. Now we have to get familiar to computing the analytic gradient for arbitrarily complex function by using the framework of computational graphs. We can apply computational graph in order to represent any function where the nodes of the graph are the steps we need to go through.

Let's start from a really simple example. 
