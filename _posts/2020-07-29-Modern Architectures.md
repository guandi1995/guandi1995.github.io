---
title: "Modern Architectures in CNN - ResNet"
date: 2020-07-29
tags: [CNN]

excerpt: "Deep Learning, CNN Architectures, ResNet, Residual Blocks"
mathjax: "true"
---
<img src="{{ site.url }}{{ site.baseurl }}/images/classical_cnn/header_img.jpeg" alt="">

In this post, we mainly focus on one of the modern architectures, ResNet, that are commonly used nowadays and have much powerful abilities so as to achieve higher accuracy for the tasks of classification and etc. Theoretically, in convolutional neural networks, the training error or accuracy is supposed to decrease as the number of layers in CNN increases; however, in reality, as the number of convolutional layers increases, the training error does decrease but then increase again if the number of layers increases significantly. Fortunately, the innovation of ResNet helps us cope with this intractable problem. The figure below demonstrates the relationship between the training error and the number of layers in CNN in terms of "plain networks" and residual networks.



### Residual Blocks
Before diving into details, Let's take a look at how residual blocks work in residual networks. In so-called plain networks as shown below, the main path from input $$a^{[l]}$$ to $$a^{[l+2]}$$ is that  input $$a^{[l]}$$ experiences linear operations first to obtain $$z^{[l+1]}$$ and goes through the ReLU activation function afterwards to generate output $$a^{[l+1]}$$, and then applies linear operation for $$a^{[l+1]}$$ to obtain $$z^{[l+2]}$$ and goes through the ReLU activation function again to generate the final output $$a^{[l+2]}$$.

Image

The formula described above can be summarized as:
$$z^{[l+1]}=W^{[l+1]}a^{[l]}+b^{[l+1]}$$

$$a^{[l+1]}=g(z^{[l+1]})$$

$$z^{[l+2]}=W^{[l+2]}a^{[l+1]}+b^{[l+2]}$$

$$a^{[l+2]}=g(z^{[l+2]})$$

What residual block does is that instead of choosing the main path described above in plain networks, Residual networks choose to go through a short-cut path, which is also called skip-connection. 
