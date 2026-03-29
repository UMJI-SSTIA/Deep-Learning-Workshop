---
discussions: true
---

# KNN and Linear Classifier

## 0\. Foreword

Computers are very good at performing a large number of explicit, repetitive operations. There are 64 squares in chess, and each side has 6 types of pieces with fixed moves. The rules of chess are also easily quantified. Therefore, IBM engineers combined brute-force search with relatively fixed human moves in a clever way, allowing them to defeat the world chess champion Kasparov using the computing power of 40 years ago.

Suppose we have 10,000 pictures, and our goal is to distinguish whether each picture is a cat or not.

Cats are cute creatures. We don't need Kasparov-level mental calculation skills to determine whether a picture is a cute kitty—it's as natural as eating or drinking. Undeniably, looking at cat pictures is the best way to kill time in the world. But if you were asked to judge 10,000 photos, over time, even the most pleasing cats would turn into mediocre four-legged animals in your eyes. Let's try handing this task over to a computer.

![type:video](../media/videos/Linear_Classifier_Animation/1080p60/LinearClassifierIntro.mp4)

During this process, the computer will encounter many problems:

1. Humans can clearly see a cat in a picture, but a computer can only see the brightness of each pixel.
2. The same photographer takes pictures of the same cat from two similar angles. In the eyes of humans, these two pictures might not be very different, but to a computer, the brightness of every pixel has drastically changed.
3. The same photographer takes pictures of two different cats from the same angle. In human eyes, both are clearly cats, but to a computer, the pixel brightness at the cat's location has drastically changed...

Sometimes, only a cat's tail appears in the photo, yet humans can say with certainty: this is a cat. Even if we successfully implement a program that judges whether there is a cat in the photo by recognizing cat eyes and bodies, it wouldn't consider a simple tail picture to be a cat.

-----

## 1\. Image Classification Problem

The issue described above is a classification problem. Suppose we have a pile of pictures, and we know beforehand what categories these pictures can be divided into. Our goal is to implement a program that, upon receiving a picture, can quickly determine which category it belongs to. Handwritten digit recognition falls under this type of problem.

### 1.1 Datasets & Models

Deep learning requires a massive amount of training data. Each data point should look like `(image, label)`, where `image` is the image itself, and `label` is the corresponding category of the image. For beginners, the biggest hurdle is acquiring this data. Fortunately, many people have already collected large amounts of categorized image data for model training in various scenarios, and we can directly use them to train our own models.

Humans can quickly determine what category a photo belongs to because we have learned what a cat is. We try to simulate this learning process on a computer. This brings us to today's topic: deep learning.

![type:video](../media/videos/Linear_Classifier_Animation/1080p60/NeuralNetworkAnimation.mp4)

Deep learning is like a water flow system, or a plumbing network. Given some input information, it will pass through a massive and complex plumbing network in the middle, which shunts and processes the information, and finally converges it into an output valve. In the image classification problem, the input information is the picture, the output information is the category of this input picture, and the plumbing network in the middle is the **model**.

However, a plumbing network that can achieve our desired functionality is incredibly complex. The time and effort required to manually set the behavior of each pipe is unacceptable. We want to design an automated algorithm that allows these pipes to adjust themselves automatically based on a large amount of data. This adjustment process is called **deep learning**, or **training the model**. The behavioral patterns obtained by the intermediate plumbing network after training, such as the "flow rate" of each pipe, are called **parameters**.

### 1.2 k-Nearest Neighbor Algorithm

Let's take CIFAR-10 as an example and consider how to implement the simplest image classification algorithm.

* 50,000 training images + 10,000 test images;
* Each picture is a 32x32 RGB three-channel image featuring ten common classes of animals or vehicles;
* The corresponding label for each picture indicates which class of animal or vehicle the main object in the image belongs to.

#### 1.2.1 Problem Simplification

Let's first consider a simpler problem. Imagine a 2D plane painted entirely with red and blue colors, but we don't know where the boundary between red and blue is. However, we do know there are many points on this plane, and we know the color of the plane at the location of each point. We want to guess as accurately as possible what color corresponds to any given point on the plane (represented by the point with the green dashed outline in the figure).

A very intuitive idea is to determine the color of this point based on the colors of its $k$ nearest neighbors. When $k=5$, we find that among the 5 points closest to the unknown point, there are 3 red points and 2 blue points. Therefore, we predict that this point is red. This is the k-Nearest Neighbor (kNN) algorithm.

![type:video](../media/videos/Linear_Classifier_Animation/1080p60/KNN2DScene.mp4)

#### 1.2.2 kNN

Returning to the CIFAR-10 dataset. Each image in CIFAR-10 is composed of 32x32 RGB three-channel pixels, and has 10 possible labels. If we treat the brightness of each channel on each pixel of the image as the coordinate of this image's data point in a certain dimension, we can map each image to a point in a high-dimensional space. So let's adapt the above problem:

There is a $32\times32\times3=3072$-dimensional high-dimensional space painted with 10 colors, but we don't know where the boundaries of these 10 colors are. However, we know there are 50,000 points in this space, and we know the color of the space at the location of each point. We want to guess as accurately as possible what color corresponds to any given point in this high-dimensional space.

Before using the kNN algorithm, we need to solve two problems first:

First, we need to define the distance between two points in the 3072-dimensional space. If $I_i$ represents the data point of the $i$-th image, and $I_{i,p}$ represents the distance of this data point in the $p$-th dimension, expanding the definition of Euclidean distance in 2D space, we get:

$$d(I_1, I_2) = \sqrt{\sum_{p=1}^{3072}(I_{1, p}-I_{2,p})^2}$$

Second, considering that taking the mode of the $k$ nearest neighbors might result in a tie, we need a rule to break it. When a tie occurs, we can select the label of the closest point among the labels with the highest number of votes.

![type:video](../media/videos/Linear_Classifier_Animation/1080p60/KNNImageSpace.mp4)

#### 1.2.3 Hyperparameters

Earlier we mentioned parameters, which are obtained after the model is trained. However, to regulate the model's performance during training, we might need to artificially set some parameters before the model starts training; these are hyperparameters. In the kNN algorithm, $k$ is a hyperparameter. Let's consider how to choose the optimal $k$.

First, let's define what makes a model optimal. Given the data, a model with a higher probability of producing accurate outputs is superior. But exactly which data should we use to evaluate the model, and how do we obtain a model's accuracy?

1. Adopt the $k$ that performs best on the training data.
      * This approach is equivalent to directly choosing $k=1$, because when $k=1$, the model will always select the label of the closest data point in the test data. And in the training data, the point closest to a certain point is always itself, with a distance of 0. This means that if we use accuracy on the training data as the criterion for evaluating the model, a model with $k=1$ will always yield a perfect score.
2. Split the existing data into two parts: 90% as training data, and the remaining 10% as test data; we choose the $k$ that performs best on the test data.
      * Compared to the previous approach, doing this prevents our model from degrading to the $k=1$ situation.
      * However, the accuracy calculated this way only represents how well this model performs on this 10% of test data. We cannot confirm how the $k$ selected in this way will perform on brand new data.

We must be aware of the issue of data contamination when tuning hyperparameters: after adjusting hyperparameters based on the results from the test data, we cannot use the model's results on this same set of test data as its definitive accuracy.

Split the existing data into three parts: for example, use 80% of the data as training data, a small portion of 10% as validation data, and the other 10% for testing. We select the $k$ that performs best based on the accuracy on the validation set, and then use the test dataset as input to calculate the true approximate accuracy of this algorithm.

#### 1.2.4 Reflection

This method is quite foolish.

The kNN model requires no training, but assuming $N$ is the size of the training dataset and $d$ is the dimensionality of each data point, the time complexity of using kNN to predict the label of a test image is $O(dN)$. When $N=50000$ and $d=3072$, testing a single photo is very slow. In reality, we can tolerate high time complexity when training a model, but we demand that using this model to predict labels for unknown data be as efficient as possible.

Secondly, calculating the Euclidean distance for each channel of every pixel doesn't make much intuitive sense.

For example, suppose all images in the training dataset belonging to the "dog" label feature small white dogs, while all images belonging to the "horse" label feature large black horses. Now we use a picture of a small black dog as test data. After pixel-by-pixel color comparison, our model finds that the small black dog is closer in distance to the large black horse, and ultimately decides the label for this small black dog is "horse".

Our kNN algorithm does not actually learn what features belong to each label.

-----

## 2\. Linear Classifier

We can also flatten each image into a $32\times32\times3=3072$-dimensional vector, and consider using matrix multiplication to apply a linear transformation to this vector.

Let $f(x; W)=W\cdot x$. Here, $x$ is the 3072-dimensional image vector, and $W$ is the $10\times3072$ parameter matrix. The result of $W\cdot x$ is a 10-dimensional vector, where the value of the $i$-th dimension represents $W$'s score for $x$ belonging to the $i$-th label. We want the correct label to have the highest score.

Think about the meaning of $W$ here. The $i$-th row vector of $W$ seems to correspond to the $i$-th label, ensuring that images of the corresponding label yield a maximum value when multiplied with it. At the same time, the size of each row of $W$ is identical to the image vector, so we can imagine each row of $W$ as a template image.

But there is also a problem with doing this: one label can only correspond to one template. The CIFAR-10 dataset has many images labeled `horse`, and the horses in these images face both left and right. The resulting horse template in $W$ after training is, as shown in the figure, a blurry two-headed horse. Intuitively speaking, the linear classifier hasn't truly learned what a horse's features are; instead, it classifies by comparing the positions of large pixel blobs.

![type:video](../media/videos/Linear_Classifier_Animation/1080p60/LinearClassifier.mp4)

### 2.1 Loss Function

Earlier we roughly described how to evaluate the superiority or inferiority of a model; now let's formally define it as a loss function.

Assume our dataset has a total of $N$ data points, shaped like $\{(x_i, y_i)\}_{i=1}^N$. We define the loss function of each data point with respect to $W$ as $L_i(f(x_i; W), y_i)$, and the loss function of the entire dataset with respect to $W$ is the average of the loss functions of each data point, namely:

$$L=\frac{1}{N}\sum_{i=1}^NL_i(f(x_i; W), y_i)$$

For a certain image vector $x_i$, let $s_i=f(x_i,W)$ be its score vector. We take the label with the highest score as the prediction for this image. Since a higher score indicates a greater likelihood that the image belongs to that label, we might as well process $s_i$ into a set of conditional probabilities $p_i$. Let the number of labels be $M$; we need $p_i$ to satisfy two properties:

1. The sum of the components across all dimensions of $p_i$ must be 1, i.e.,

    $$\sum_{j=1}^Mp_{i,j}=1;$$

2. The probability corresponding to the correct label in $p_i$ needs to be significantly larger than the probabilities of other incorrect labels.

We can quickly think of a solution: directly set the dimension in $p_i$ with the largest $s_i$ to 1, and the other dimensions to 0. However, after doing this, we'll find that for the same image $x$, $p_i$ is not differentiable when using $W$ as the independent variable. Differentiability is crucial for us to find the most accurate $W$, so we need to slightly alter our strategy:

#### 2.1.1 Softmax Function & Cross-Entropy Loss

$$p_{i,j} = \frac{\exp(s_{i,j})}{\sum_{k=1}^{M} \exp(s_{i,k})}$$

Doing this is equivalent to using a differentiable function to imitate the operation of directly taking the maximum value and setting it to 1, hence we call it the Softmax function.

Let's further consider how to calculate $L_i$ using $p_i$. In the best-case scenario, a $W$ can perfectly fit all the data in the dataset; let $L_i=0$ at this time. As the fitting ability of $W$ worsens, $L_i$ should also become larger.

$$L_i(f(x_i, W), y_i)=-\log p_{i, y_i}$$

This is the loss function that fits the above characteristics.

#### 2.1.2 Regularization

When designing the loss function, we also need to consider how to avoid the occurrence of overfitting.

First, consider a relatively simplified situation: as shown in the figure, given many points on a 2D coordinate system, we are asked to fit a polynomial curve that can relatively accurately summarize the pattern of these points.

Experience tells us that simply using a linear equation as shown in the left figure can summarize the general trend of this data. However, if we insist on having the computer choose a curve that completely fits all data points, we might get a super convoluted polynomial curve. But the computer doesn't know this, so we need to introduce another hyperparameter to express our aversion to overfitting to the computer.

To review, hyperparameters refer to parameters that have already been artificially determined before model training begins, just like $k$ in the k-NN algorithm is a hyperparameter. Here, we want to add a regularization term to the loss function to increase the numerical value of the loss function for curves where overfitting occurs.

In the example of the above figure, the more non-zero terms in the polynomial, the more complex the polynomial curve becomes. Similarly, we empirically hope that this model is simpler; that is, we hope our model can use fewer features, meaning we want more terms in the matrix to be smaller and closer to 0. We can communicate this preference of ours to the computer through a previously mentioned hyperparameter—a parameter determined before training. We can add a regularization term after the existing loss function:

$$L=\frac{1}{N}\sum_{i=1}^NL_i+\lambda\cdot R(W)$$

$$R(W)=\sum_j\sum_k(W_{j,k})^2$$

Here, $\lambda$ is a real number used to adjust the magnitude of the impact the regularization term has on the loss function. The size of $\lambda$ and the definition method of $R(W)$ can both be counted as hyperparameters, and we similarly need to avoid data contamination when determining them.

This standardization method of taking out each term of $W$, squaring it, and summing them up is called L2 Regularization, and facts have proven that this method works very well. As for why doing this yields great results, interested listeners can further consult relevant literature to explore.

### 2.2 Optimizing Weight Matrix

Now we know we need to use a matrix to perform a linear transformation on the original image, and we also know we can use a loss function to evaluate whether a matrix can faithfully reflect the patterns in the dataset. Next, our goal is to find a $W$ with a very small loss function $L$. We are finally going to start training our model.

Discussing matrices directly is a bit complex; we might as well simplify the problem again. For a differentiable function $f:\mathbb{R}\rightarrow\mathbb{R}$, deriving it at a certain point gives us the slope of the function's graph at that point:

$$\frac{d f(x)}{dx}=\lim_{h\rightarrow 0}\frac{f(x+h)-f(x)}{h}$$

Suppose we want to find a relatively low point on this function. We can start from any point and descend in the direction of the slope. Although doing this won't guarantee that we stably land on the global minimum point, it can at least ensure a local optimal solution.

#### 2.2.1 Gradient

Multivariate functions also have similar gradient vectors, where each dimension of the vector represents the derivative of the multivariate function with respect to each variable. The slope of a multivariate function in a certain direction can be obtained by taking the dot product of this direction vector and the gradient vector. Through mathematical operations, we know that the direction in which a multivariate function descends fastest is actually the exact opposite direction of this gradient vector.

#### 2.2.2 Gradient Descent

We know that the loss function $L=\frac{1}{N}\sum_{i=1}^NL_i(f(x_i; W), y_i)+\lambda\cdot R(W)$ is a function of $W$, and also a multivariate function of each term in $W$. Therefore, we can calculate the gradient of $L$ with respect to $W$, $\nabla_WL$. Consequently, we can know how to change $W$ when $W=W_0$ so that the loss function $L$ descends the fastest.

Consider how to obtain $\nabla_WL$.

1. We can obtain the numerical solution of $\nabla_WL$. by slightly adding an extremely small value to each term in $W$ and observing the change in $L$. However, when the matrix size of $W$ is very large, this method is highly inefficient.
2. We could also manually derive the partial derivative of $L$ with respect to each term in $W$, but we would need to re-derive the formulas every time we wanted to modify our algorithm, which is very troublesome.
3. We can design a modular algorithm capable of automatically obtaining the analytical solution of the gradient through formulas. This is the well-known **Back Propagation** algorithm. Interested listeners can look up more information on their own.

We descend from a random starting point $W_0$ to a locally optimal $W$ through multiple iterations.

$$
\begin{aligned}
&\textbf{procedure } \text{GradientDescent}(rate, cnt)\\
&\quad\quad\text{initialize } W\\
&\quad\quad\textbf{for } t=1 \textbf{ to } cnt:\\
&\quad\quad\quad\quad d:=\nabla_WL\\
&\quad\quad\quad\quad W:=W-d\cdot rate\\
&\quad\quad\textbf{end}\\
&\textbf{return } W
\end{aligned}
$$

In this process, the initialization method of $W$, the learning $rate$ (step size), and the step count $cnt$ all belong to hyperparameters that need to be set in advance.

![type:video](../media/videos/Linear_Classifier_Animation/1080p60/RosenbrockFullProcess.mp4)

#### 2.2.3 Stochastic Gradient Descent

Reviewing the calculation process of the loss function $L$: we need to enumerate the loss function $L_i$ of every data point in the dataset and sum them up. And every $L_i$ requires performing a matrix multiplication. When the width and height of the matrix are very large, or when we design more complex models later on that use more matrix multiplications, the time cost of executing each descent step is extremely high, which is unacceptable to us.

We consider that every time we calculate the gradient, we only randomly sample a small batch from the massive dataset to calculate the loss function $L'$ of $W$ on this small batch of data to approximate $L$, obtaining an approximate solution with fewer calculations. This is Stochastic Gradient Descent (SGD for short).

-----

## 3\. Neural Networks

Since we found that matrix multiplication can fit the given dataset, why not try adding a few more layers? Let $f(x;W_1, W_2)=W_2\cdot W_1\cdot x$. At this point, we need to simultaneously optimize $W_1$ and $W_2$ during gradient optimization.

But doing this actually has a huge problem: both $W_1$ and $W_2$ are matrices, and matrix multiplication is associative. If we let $W=W_1\cdot W_2$, then the fitting capability of our model hasn't actually changed at all, yet we consumed more resources during optimization.

### 3.1 Activation Function

It's all linearity's fault\! We must introduce non-linear factors to break the associativity of matrix multiplication and enhance our model's fitting ability\!

Let $f(x;W_1, W_2)=W_2\cdot \max(0, W_1\cdot x)$. The $\max(0, \cdot)$ operation here means taking the maximum value between each term in the matrix and 0. Similarly, we can introduce even more layers of matrices: $f(x;W_1, W_2, W_3)=W_3\cdot\max(0, W_2\cdot \max(0, W_1\cdot x))$.

Because the width and height of $W_2$ are not rigidly restricted by the dimensions of the input and output vectors, we can scale it up. By doing this, we effectively expand the previous limitation of each row of $W$ acting as a single template: now, one label can correspond to many templates, and our model thus gains the ability to "learn" multiple features from images of a single label.

Functions like $\max(0,\cdot)$ that can introduce non-linear factors to the model are called activation functions. $\max(0, \cdot)$ is frequently used due to its fast calculation speed and its advantage of not triggering **vanishing gradients** after multiple nested layers. It is granted the name ReLU (Rectified Linear Unit).

![type:video](../media/videos/Linear_Classifier_Animation/1080p60/ActivationFunctions.mp4)

### 3.2 Perceptron

We don't just have to repeatedly apply this operation to a single matrix: we can also combine the input vector linearly after processing it through multiple pathways of linear transformations + non-linear activation functions.

Up to this point, we have built the basic unit of deep learning from scratch—the Perceptron.

Have you noticed that it looks very much like a biological neuron?

![type:video](../media/videos/Linear_Classifier_Animation/1080p60/FNNMath.mp4)

-----

## References

Tendo, L. SSTIA Deep Learning Workshop 2025. Source: <https://github.com/UMJI-SSTIA/Deeplearning-wksp-2025/blob/main/Worksheet/Part1.html>
