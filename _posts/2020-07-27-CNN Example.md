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
LeNet-5, a classical convolutional neural network that was introduced back to 1998, is aimed to recognize the digits from 0 to 9. The first figure below is the model architecture from the paper and the second one is the model that is similar to LeNet-5. LeNet-5 is such a classical model that it consists of two convolution layers followed by average pooling layers for each and apply three fully connected layers in the end of the network. The second model is quite similar to LeNet-5 except using max pooling layers.

LeNet-5
<img src="{{ site.url }}{{ site.baseurl }}/images/classical_cnn/LeNet-5.PNG" alt="">


LeNet-5 modified version
<img src="{{ site.url }}{{ site.baseurl }}/images/classical_cnn/LeNet-5_modified.PNG" alt="">

As shown above for the LeNet-5 modified model, the network takes a $$32\times32\times3$$ image as the input and convolve it with 6 $$5\times5\times3$$ filters and stride of 1, which results in a $$28\times28\times6$$ output and then apply the output to a max pooling layer with the filter size of $$2\times2$$ and stride of 2, which leads to a $$14\times14\times6$$ output. Then apply the same convolution and max pooling layer again to obtain a $$5\times5\times16$$ output. At the end of the network, the $$5\times5\times16$$ output is flattened to a vector with the size of $$400\times1$$ and then apply two fully connected layers to it and finally achieve the estimated output with the size of $$10\times1$$ by using softmax.

Take the second architecture as an example, let's practice how to compute the amounts of the parameters that are needed to learn in each layer of the network.

|                            | Activation Size | Number of Parameters                    |
| -------------------------- | --------------- | --------------------------------------- |
| Input                      | (32,32,3)       | $$0$$                                   |
| CONV1 (f=5, s=1, #6 filters)  | (28,28,6)       | $$(5\times5\times3+1^{[b]})\times6=456$$      |
| POOL1 (f=2, s=2)             | (14,14,6)       | $$0$$                                   |
| CONV2 (f=5, s=1, #16 filters) | (10,10,16)      | $$(5\times5\times6+1^{[b]})\times16=2416$$    |
| POOL2 (f=2, s=2)             | (5,5,16)        | $$0$$                                   |
| FC3 (120 neurons)           | (120,1)         | $$5\times5\times16\times120+120^{[b]}=48120$$ |
| FC4 (84 neurons)            | (84,1)          | $$120\times84+84^{[b]}=10164$$                |
| Softmax (10 neurons)        | (10,1)          | $$84\times10+10^{[b]}=850$$                   |

where b=bias and note that in the LeNet-5 network, they applied non-linearity activation function of sigmoid after each max pooling layer.
In summary, the total number of parameters that the network needs to learn is approximately 62,000. According to this classical architecture, there are actually several patterns that the modern architectures still apply, which are the general structures of the networks - CONV --> POOL --> CONV --> POOL --> FC --> FC. That is, convolution layers are followed by pooling layers and a few of fully connected layers are located in the end of the network. Additionally, the trends of nowadays networks that $$n_H$$ and $$n_W$$ decrease while $$n_C$$ increases as the networks go deeper are still applying.

### AlexNet
AlexNet, introduced in 2012, employed an 8-layer convolutional neural network where the architecture is quite similar to LeNet-5, but there are also some significant differences. First, AlexNet is much deeper  since it consists of five convolution layers, two hidden fully-connected layers and one fully-connected output layer as shown in the figure below. Aside from that, AlexNet used the ReLu activation function instead of sigmoid.

<img src="{{ site.url }}{{ site.baseurl }}/images/classical_cnn/AlexNet.PNG" alt="">

In the first layer of AlexNet, the convolution window size is $$11\times11\times3$$. This is because this model is implemented to classify the inputs from ImageNet, the dataset which has bigger size. Consequently, a larger convolution window is applied in the first layer in order to capture the object. 
