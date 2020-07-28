---
title: "Classical Architectures in CNN"
date: 2020-07-27
tags: [CNN]

excerpt: "Deep Learning"
mathjax: "true"
---
<img src="{{ site.url }}{{ site.baseurl }}/images/classical_cnn/header_img.jpeg" alt="">

Since we have addressed the process of main building blocks in convolutional neural networks (CNN), now let's take a look at several classical and modern CNN architectures. In this post, we will mainly focus on the classical architectures such as  LeNet-5, AlexNet and VGG16 while modern network architectures such as Inception and ResNet will be introduced in the following posts.

### LeNet-5
LeNet-5, a classical convolutional neural network that was introduced back to 1998, is aimed to recognize the digits from 0 to 10. The first figure below is the model architecture from the paper and the second figure is the model that is similar to LeNet-5. LeNet-5 is such a classical model that it consists of two convolution layers followed by average pooling layers for each and apply 3 fully connected layers in the end of the network. The second model is quite similar to LeNet-5 except using max pooling layers.

<img src="{{ site.url }}{{ site.baseurl }}/images/classical_cnn/LeNet-5.PNG" alt="">

<img src="{{ site.url }}{{ site.baseurl }}/images/classical_cnn/LeNet-5.PNG" alt="">

As shown above for the second model, the network takes a $$32\times32\times3$$ image as the input and convolve it with 6 $$5\times5\times3$$ filters and stride of 1, which results in a $$28\times28\times6$$ output and then apply the output to a max pooling layer with the filter size of $2\times2$ and stride of 2, which leads to a $$14\times14\times6$$ output. Then apply the same convolution and max pooling layer again to obtain a $$5\times5\times16$$ output. At the end of the network, the $$5\times5\times16$$ output is flattened to a vector with the size of $$400\times1$$ and then apply two fully connected layers to it and finally achieve the estimated output with the size of $$10\times1$$ by using softmax.


### AlexNet
