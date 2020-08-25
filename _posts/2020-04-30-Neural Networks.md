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

The input layer is denotes as $$a^{[0]}$$ while the first hidden layer is denoted as $$a^{[1]}$$ and it has three hidden units total. Each hidden unit is denoted as $$a_i^{[1]}$$, represented as the $$i$$-th hidden unit in the 1st hidden layer. Additionally, we denote the vectorized  first-hidden-layer as $$a^{[1]}$$ and $$a^{[1]}=\begin{bmatrix}a_1^{[1]}&a_2^{[1]}&a_3^{[1]}\\\end{bmatrix}^T$$.

### Computing a Neural Network's Output
After being familiar with the notation of neural networks, let's discuss how to compute the final output through the hidden layers. Take the same neural network example above, we have a single sample input $$x=a^{[0]}=\begin{bmatrix}x_1&x_2\\\end{bmatrix}^T$$ with the shape of (2,1).

The value of $$a_i^{[1]}$$ is obtained by mapping the function of $$a_i^{[1]}=\sigma(z_i^{[1]})$$ where $$\sigma(.)$$ is the sigmoid activation function while the value of $$z_i^{[1]}$$ is computed by the formula $$z_i^{[1]}={w_i^{[1]}}^T x+b_i^{[1]}$$ where $$w_i^{[1]}$$ is the weight with the shape of (2,1) and $$b_i^{[1]}$$ is a real number bias term. Then, the exact value of $$a_i^{[1]}$$ and $$z_i^{[1]}$$ for the first hidden layer are computed by the following demonstration:

$$
z_1^{[1]}={w_1^{[1]}}^T x+b_1^{[1]}, ~~a_1^{[1]}=\sigma(z_1^{[1]})\\
z_2^{[1]}={w_2^{[1]}}^T x+b_2^{[1]}, ~~a_2^{[1]}=\sigma(z_2^{[1]})\\
z_3^{[1]}={w_3^{[1]}}^T x+b_3^{[1]}, ~~a_3^{[1]}=\sigma(z_3^{[1]})\\
$$

Now, to vectorize those three terms, we have

$$
z^{[1]}=\begin{bmatrix}z_1^{[1]}\\z_2^{[1]}\\z_3^{[1]}\end{bmatrix}=
\begin{bmatrix}-{w_1^{[1]}}^T-\\-{w_2^{[1]}}^T-\\-{w_3^{[1]}}^T-\end{bmatrix}
\begin{bmatrix}x_1\\x_2\end{bmatrix} + \begin{bmatrix}b_1^{[1]}\\b_2^{[1]}\\b_3^{[1]}\end{bmatrix}
$$

and

$$
a^{[1]}=\begin{bmatrix}a_1^{[1]}\\a_2^{[1]}\\a_3^{[1]}\end{bmatrix}=
\begin{bmatrix}\sigma(z_1^{[1]})\\\sigma(z_2^{[1]})\\\sigma(z_3^{[1]})\end{bmatrix}
$$

While the values for the second hidden layer, $$z_i^{[1]}$$ and $$a_i^{[1]}$$ can be computed based on the formulas below:

$$
z^{[2]} = \begin{bmatrix}z_1^{[2]}\end{bmatrix} =
\begin{bmatrix}-{w_1^{[2]}}^T-\end{bmatrix}\begin{bmatrix}x_1\\x_2\end{bmatrix} +
 \begin{bmatrix}b_1^{[2]}\end{bmatrix}
$$

and

$$
a^{[2]}=\begin{bmatrix}a_1^{[2]}\end{bmatrix}=
\begin{bmatrix}\sigma(z_1^{[2]})\end{bmatrix}
$$

With that being said, we have the vectorized version of computation for the overall neural networks, which are

$$
z^{[1]}=W^{[1]}a^{[0]}+b^{[1]}
a^{[1]}=\sigma(z^{[1]})
z^{[2]}=W^{[2]}a^{[1]}+b^{[1]}
a^{[2]}=\sigma(z^{[2]})
$$

where $$a^{[0]}=input~x$$


### Vectorizing across Multiple Examples


### Activation Function
