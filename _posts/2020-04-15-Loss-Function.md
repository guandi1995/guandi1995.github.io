---
title: "Linear Classifier"
date: 2020-04-15
tags: [machine learning, data science, linear classifier]

excerpt: "Machine Learning, Data Science, Linear Classifier"
mathjax: "true"
---
<img src="{{ site.url }}{{ site.baseurl }}/images/linear_classifier/header_image.png" alt="">

#### Introduction to linear classifier

In last section, we have approach to the image classification problem by using kNN classifier, which assign labels to testing images by comparing the distance to each training images. However, it has a number of disadvantages:

- classifying a test image is expansive because it requires a comparison to all training images
- it barely has training process except simply remembering the training and testing images
- it is time consuming for predicting process since you have to compute the distance for each testing image to each training image.

Overall, in practice, **you want the time for training to be long but the time for predicting to be relatively short!!** kNN performs this principle in a total opposite way, which leads this method is barely used in practice.

However, Neural Networks and Convolutional Neural Networks can take place and play a powerful role to image classification.

This approach has two major components: **score function** and **loss function**.

- score function: mapping the raw data to class scores, $R^{n}\rightarrow R^{k}$ where $n$ is the feature number, 3072, and $k$ is class number, 10.
- loss function: qualifying the agreement between the predicted scores and the ground truth labels.

**You want to cast this as an optimization problem where you minimize the loss function with respect to the parameters of the score function.**



#### Linear score function

For score function in CIFAR-10, we define it as $R^{n}\rightarrow R^{k}$ where $n=3072$ and $k=10$. We define the linear classifier as:
$$
f(x_i;W,b)=Wx_i+b
$$
where $x_i \in R^{3072},W\in R^{10*3072},b\in R^{10},f(x) \in R^{10}$. $W$ is called **weights matrix** and $b$ is called the **bias vector**.  $W$ and $b$ are all considered as **parameters**.

Notes:

- Score function $f(x_i;W,b)$ is then a score vector for each $x_i$. Intuitively, you want that the **correct class has a higher score than other classes**.
- An advantage of this approach is that the training data is used to learn the parameters $W,b$ and then predict the class scores for testing image by plugging in the testing image as $x_i$. This is also how CNN works where to map image pixels to scores exactly shown above instead it has much more complex and contain more parameters.



#### Interpreting a linear classifier

To simplify the problems, for instance, we have 3 classes: cat, car, frog.  By using the linear classifier function described above, assume our parameters: weights matrix and bias vector as below:
$$
f(x_i;W,b)=Wx_i+b\\
W= \begin{bmatrix}0.2&-0.5&0.1&2&3\\1.5&1.3&2.1&0&1\\0&0.25&0.2&-0.3&2.5\\\end{bmatrix},
x_i=\begin{bmatrix}56\\231\\24\\2\\4\\\end{bmatrix},
b=\begin{bmatrix}1.1\\3.2\\-1.2\\\end{bmatrix},
f(x_i;W,b)=\begin{bmatrix}3.2\\5.1\\-1.7\\\end{bmatrix}
$$
In this case, $x_i$ is the flatten image pixels of class cat. Repeating to plug the value of $x_i$ into $f(x_i;W,b)$ for other two classes images. We could have such a table below:

|      | image1(class=cat) | image2(class=car) | image3(class=frog) |
| ---- | ----------------- | ----------------- | ------------------ |
| cat  | **3.2**           | 1.3               | 2.2                |
| car  | 5.1               | **4.9**           | 2.5                |
| frog | -1.7              | 2.0               | **-3.1**           |

To interpret the linear classifier, you can **treat each row of $W$ corresponds to a template** and each entry of the row of $W$ is the weights for the corresponding pixels of the input image $x_i$. For instance, the first entry of the first row of $W$ is the weights for the first pixel of the input image, $x_i$.

Another way to think of it is that you are doing template matching. You only use a single image per class instead of having thousands of training images, and use the inner product as the metrics instead of L1 or L2 distance metrics in kNN.
