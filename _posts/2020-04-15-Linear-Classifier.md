---
title: "Linear Classifier"
date: 2020-04-15
tags: [machine learning, data science, linear classifier]

excerpt: "Machine Learning, Data Science, Linear Classifier"
mathjax: "true"
---
<img src="{{ site.url }}{{ site.baseurl }}/images/linear_classifier/header_image.png" alt="">

### Introduction to Linear Cassifier

In last post, we approached to the problem of image classification by using kNN classifier, aiming to assign labels to testing images by comparing the distance to each training image. However, it has a number of disadvantages in real life implementation:

- classifying a test image is expansive because it requires a comparison to all training images
- it barely has training process except simply remembering the training and testing images
- it is time consuming for predicting process since you have to compute the distance of each testing image to each training image.

In practice, **we allow the time for training to be long but the time for predicting to be relatively short!!** kNN performs this principle in a completely opposite way, which leads this method is barely used in practice.

The best replacement of it is the techniques of Neural Networks and Convolutional Neural Networks. They can take place and play a powerful role to solve the problems associated with supervised learning such as classification.

This approach has two major components: **score function** and **loss function**.

- score function: mapping the raw data to class scores, $$R^{n}\rightarrow R^{k}$$ where $$n$$ is the number of features, and $$k$$ is number of classes. Take CIFAR-10 as our example, the number of features is 32x32x3 = 3072 while the number of classes is 10.
- loss function: qualifying the agreement between the predicted scores and the ground truth labels.

**We want to cast this as an optimization problem where we minimize the loss function with respect to the parameters of the score function.**



### Linear Score Function

For score function in CIFAR-10, we define it as $$R^{n}\rightarrow R^{k}$$ where $$n=3072$$ and $$k=10$$. Our linear classifier would be:
$$
f(x_i;W,b)=Wx_i+b
$$
where $$x_i \in R^{3072},W\in R^{10*3072},b\in R^{10},f(x) \in R^{10}$$. $$W$$ is called **weights matrix** and $$b$$ is called the **bias vector**.  $$W$$ and $$b$$ are all considered as **parameters**.

Notes:

- Score function $$f(x_i;W,b)$$ is then a score vector for each training image $$x_i$$. Intuitively, we want that **the correct class has a higher score than other classes**.
- An advantage of this approach is that the training data is used to learn the parameters $$W,b$$ and then predict the class scores for testing image by plugging in the testing image, $$x_j$$. This is also how Convolutional Neural Network works where to map all image pixels $$R^n$$ to scores $$R^k$$ shown above instead CNN is much more complex and contains more parameters.



### Interpreting Linear Classifier

To simplify the problem, for instance, we have 3 classes: cat, car, frog.  By using the linear classifier function described above, assume our parameters: weights matrix $$W$$ and bias vector $$b$$ as below:

$$
f(x_i;W,b)=Wx_i+b\\
W= \begin{bmatrix}0.2&-0.5&0.1&2&3\\1.5&1.3&2.1&0&1\\0&0.25&0.2&-0.3&2.5\\\end{bmatrix},
x_i=\begin{bmatrix}56\\231\\24\\2\\4\\\end{bmatrix},
b=\begin{bmatrix}1.1\\3.2\\-1.2\\\end{bmatrix},
f(x_i;W,b)=\begin{bmatrix}3.2\\5.1\\-1.7\\\end{bmatrix}
$$

As training images, $$x_1,x_2,x_3$$ are belonged to the class of cat, car and frog, respectively. Each image is plugged into the score function $$f(x_i;W,b)$$ and has a score vector in $$R^3$$. The score vector then represents the scores for each training image in terms of all three classes. Based on the formula above, we could summarize into a table in order to have a clear visualization:

|      | image1(class=cat) | image2(class=car) | image3(class=frog) |
| ---- | ----------------- | ----------------- | ------------------ |
| cat  | **3.2**           | 1.3               | 2.2                |
| car  | 5.1               | **4.9**           | 2.5                |
| frog | -1.7              | 2.0               | **-3.1**           |



To interpret the linear classifier, we can treat **each row of $$W$$ corresponds to a template** and each entry of the row in $$W$$ is the weights for the corresponding pixels of the input training image $$x_i$$. For instance, the first entry of the first row in $$W$$ is the weight for the first pixel of the input image, $$x_i$$.

Another way to think of it is that we are doing template matching. We only use a single image per class instead of having thousands of training images, and use the inner product as the metrics instead of L1 or L2 distance metrics in kNN.

In the table above, take the first training image, $$x_1$$ as our example, the score vector for $$x_1$$ is $$[3.2,5.1,-1,7]^T$$. Since the ground truth label for $$x_1$$ is cat, intuitively, the score for class cat is supposed to be much higher than the other two classes. However, for $$x_1$$, the highest score is the class of car, 5.1 while the score for the class of cat is 3.2, which means that we did not obtain a good prediction or measurement for $$x_1$$.

As human beings, We conclude the above observation by simply visualizing the score numbers, but we need an approach to let our algorithm decide whether the scores are reasonable and qualify whether the score is good or bad. This leads us to the next topic, loss function, the topic that plays a significant impact on deep learning.

Next post, I will address the topic of loss function in more detail.
