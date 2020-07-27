---
title: "Pooling Layers"
date: 2020-07-20
tags: [machine learning, data science, Pooling]

excerpt: "Deep Learning, Pooling Layers"
mathjax: "true"
---

<img src="{{ site.url }}{{ site.baseurl }}/images/pooling/header_img.png" alt="">

Pooling layer is another building blocks in the convolutional neural networks. Before we address the topic of the pooling layers, let's take a look at a simple example of the convolutional neural network so as to summarize what has been done.

### A Simple CovNet Example
Suppose there is a $$39\times39\times3$$ input image and then apply it to three convolution layers with following steps:

image

As shown in the figure above, the input layer $$a^{[0]}$$ is convolved with 10 filters with size of $$3\times3\times3$$, stride of 1 and without padding. The output size of the first convolution layer thus will be $$37\times37\times10$$ where $$n_H^{[1]}=n_W^{[1]}=\lfloor\dfrac{39+0-3}{1}+1\rfloor=37$$ and $$n_c^{[1]}=n_f=10$$. Then, the output of the first convolution layer, as the input of the second convolution layer, is convolved with 20 filters with the size of $$5\times5\times10$$, stride of 2 and without padding. The output size of the second convolutional layer thus will be $$17\times17\times20$$ where $$n_H^{[2]}=n_W^{[2]}=\lfloor\dfrac{37+0-5}{2}+1\rfloor=17$$ and $$n_c^{[2]}=n_f=20$$. Then, the output of the second convolution layer, as the input of the third convolution layer, is convolved with 40 filters with the size of $$5\times5\times20$$, stride of 2 and padding of 1. The output size of the third convolutional layer thus will be $$8\times8\times40$$ where $$n_H^{[3]}=n_W^{[3]}=\lfloor\dfrac{17+2\times1-5}{2}+1\rfloor=8$$ and $$n_c^{[3]}=n_f=40$$. Then flattening all the neurons in the output of the third convolution layer and vectorize those neurons in order to form a vector with the size of $$2560\times1$$ and finally apply softmax to obtain the value of $$\hat{y}$$ in order to achieve the purpose of the tasks such as classification and etc.

### Pooling Layers
In general, there are three types of layer in a convolutional neural network, which are convolution layer (CONV), pooling layer (POOL) and fully connected layer (FC). Typically, several convolution layers are followed by a pooling layer and a few fully connected layers are at the end of the convolutional network.

The function of pooling layer is to reduce the spatial size of the representation to reduce the amount of parameters and computation in the network and it operates on each feature map (channels) independently. There are two types of pooling layers, which are *max pooling* and *average pooling*. However, max pooling is the one that is commonly used while average pooling is rarely used.

The operations of the max pooling is quite simple, only two hyperparameters are used in pooling layer, which are filter size ($$f$$) and stride ($$s$$).
