---
title: "Neural Networks"
date: 2020-04-30
tags: [machine learning, neural networks]

excerpt: "machine learning"
mathjax: "true"
---

<img src="{{ site.url }}{{ site.baseurl }}/images/neural networks/header_img.svg" alt="">

### Quick Introduction to Neural Networks

In the following posts, we will discuss the neural networks in machine learning and deep learning, which is one of the most important topics in those areas. Neural networks are modeled as collections of neurons that are connected in an acyclic graph. It can be multiple layers in neural networks and the outputs of one layer will be the inputs of the following layer. For regular neural networks, the most common layer type is the fully-connected layer where neurons between two adjacent layers are fully pairwise connected, but neurons within a single layer don't share the connections. Below are two examples of fully-connected layer of neural networks:    

<img src="{{ site.url }}{{ site.baseurl }}/images/neural networks/neural_networks_example.PNG" alt="">

The left figure illustrates a two-layer fully-connected neural network with one hidden layer and one output layer while the right one shows a three-layer fully-connected neural network with two hidden layers and one output layer.

One of the most important thing in neural networks is that each layer generally has an activation function except the output layer. This is because the last output layer is usually taken as real-valued numbers to represent the class scores (e.g. classification) or as real-valued target (e.g. regression). The reason why each layer has an activation function will be discussed below in the section of Activation Function.

The figure below demonstrates an example of two-layer neural networks notation.

<img src="{{ site.url }}{{ site.baseurl }}/images/neural networks/notation.jpg" alt="">

The input layer is denotes as $$a^{[0]}$$ while the first hidden layer is denoted as $$a^{[1]}$$ and it has three hidden units, which are denoted as $$a_i^{[1]}$$.

### Computing a Neural Network's Output


### Vectorizing across Multiple Examples


### Activation Function
