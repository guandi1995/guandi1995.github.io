---
title: "Computational Graph and Backpropagation"
date: 2020-04-19
tags: [machine learning, optimization, backpropagation]

excerpt: "machine learning"
mathjax: "true"
---

<img src="{{ site.url }}{{ site.baseurl }}/images/loss_function/header_image.jpg" alt="">

In the previous post, we address the topics of gradient descent in order to minimize the loss function. Now we have to get familiar to computing the analytic gradient for arbitrarily complex function by using the framework of computational graphs. We can apply computational graph to represent any function with forward pass and backward pass so that we could use the computational graph to compute the gradient or derivative of the loss function with respect to the parameters.

Let's start from a really simple example. Consider a loss function $$L$$ parameterized by $$a,b,c$$ where $$J(a,b,c)=3(a+bc)$$. First, let's start from the forward pass, which is quite simple. The forward pass refers to the regular computations that conduct the operations such as adding, subtracting, multiplying and etc in the direction from left to right. The only thing we need to notice is that we could divide the complex functions into single computation components and then combine them together to represent the whole function. Thus, denote $$u=bc$$, $$v=a+u$$ and $$J=3v$$. The figure below illustrates how forward pass works for this function:

While for the backward pass or backpropagation, its ultimate goal is just to compute the gradient of the loss function with respect to parameters. To do so, we need to compute the gradient of all components in the direction from right to left. The figure below illustrates how backward pass works. First, compute the gradient of $$\dfrac{dJ}{dv}$$ on the most right side, then compute the gradient of $$\dfrac{dJ}{da}$$ and keep in mind that $$\dfrac{dJ}{da}=\dfrac{dJ}{dv}\dfrac{dv}{da}$$ by chain rule. Since the value of $$\dfrac{dJ}{da}$$ can be determined by $$\dfrac{dJ}{dv}$$, which is the gradient on its right hand side and this is also why we need to compute the gradient from right to left. To compute the gradient of $$\dfrac{dJ}{db}, \dfrac{dJ}{dc}$$, we need to compute the gradient of $$\dfrac{dJ}{du}$$ first since $$\dfrac{dJ}{db}=\dfrac{dJ}{du}\dfrac{du}{db}$$ and $$\dfrac{dJ}{dc}=\dfrac{dJ}{du}\dfrac{du}{dc}$$. 
