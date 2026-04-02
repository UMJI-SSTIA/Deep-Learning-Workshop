---
discussions: true
---

# Convolutional Neural Network

## 4\. Why Perceptrons Aren't Enough

We have learned about Linear Classifier in the previous study, but in fact, it is still usually not enough for identify cat in real world.

When we treat an image as a 1D list of 3072 numbers ($32 \times 32 \times 3$), we are effectively telling the computer that the "top-left pixel" has no special relationship with the pixel right next to it.

* Spatial Structure: A cat is defined by the geometry of its features. A pointy ear isn't just a collection of pixels; it's a specific arrangement of "light" and "dark" values in a 2D space. By flattening the image, we force the model to work ten times harder to "re-learn" that pixels $(0,0)$ and $(0,1)$ are neighbors.
* Translation Invariance: If our model learns that a "cat ear" always appears in the top-left (neurons 0–100), and then we show it a photo where the cat is in the bottom-right, the model will fail. It lacks "translation invariance"—the ability to recognize a pattern regardless of where it appears in the frame.

---

## 5\. Convolutional Neutral Networks

### 5.1 How the Computer "Sees" the Cat

Unlike grayscale digits, our cat images are RGB (Red, Green, Blue). This means our input isn't just a flat grid; it's a "sandwich" of three $32 \times 32$ layers.

### 5.2 The Filter

Our "flashlight" (the **Filter** or **Kernel**) is also 3D. If we use a $5 \times 5$ filter, it actually has a depth of 3 to match the RGB channels.

* The Dot Product: As the $5 \times 5 \times 3$ filter slides across the $32 \times 32 \times 3$ image, it calculates a single number at every stop.
* The Activation Map: This sliding process results in a 2D "heatmap." If the filter is looking for "orange fur texture," the heatmap will glow brightly in the areas where the cat's orange coat is present and stay dark on the blue sky background.

To make it more easy to understand, let's simlify it to use a single layer of the image and 2D filter.

![type:video](../media/videos/CNN_Animation/1080p60/Convolution2D.mp4)

### 5.3 Symmetry Breaking

If you started every filter with the same weights (e.g., all 0.5), every single filter in your first layer would detect the exact same thing. By randomly initializing the weights, we ensure:

1. Filter Diversity: Filter A might start leaning toward vertical edges, while Filter B starts leaning toward green grass colors.
2. Efficient Learning: During training, the "scouts" (filters) explore different "mathematical paths" to minimize the error, ensuring we don't miss any critical cat features like whiskers or pupils.

---

## 6\. ReLU

Just as we already studied, we recall that we can use nonlinear activation like **ReLU** or **Sigmoid** for dealing the output of each layer.

![type:video](../media/videos/Linear_Classifier_Animation/1080p60/ActivationFunctions.mp4)

A cat's edge isn't a simple mathematical line; it's a complex transition of light. After every convolution, we apply **ReLU** ($f(x) = \max(0, x)$). Generally **ReLU** is the more common choice than **Sigmoid** nowadays.

* Why zero out negatives? By removing negative signals, we create sparsity. It effectively tells the network: "If this area doesn't look like an ear at all, stop sending information about it." This helps the model focus only on the relevant parts of the image.

---

## 7\. Pooling

Once we have our feature maps (e.g., a "Whisker Heatmap"), we use **Max Pooling**.

* Spatial Reduction: We slide a $2 \times 2$ window over the $32 \times 32$ feature map and only keep the largest value. This shrinks the map to $16 \times 16$.
* Flexibility: Max Pooling gives the model "wiggle room." As long as the cat's whisker is somewhere in that $2 \times 2$ area, the strongest signal survives. This is how the CNN learns to recognize the cat even if it moves slightly or tilts its head.

![type:video](../media/videos/CNN_Animation/1080p60/WhiskerMaxPooling.mp4)

---

## 8\. The Grand Architecture

The magic happens in the sequence:

1. **Low-Level Features:** The first layers detect simple things like "diagonal edges" or "patches of white fur."
2. **High-Level Features:** Deep layers combine those edges into shapes (circles, triangles) and those shapes into objects (eyes, ears).
3. **The Flattening:** After several rounds of CONV and POOL, our $32 \times 32 \times 3$ image might be reduced to a small but very deep volume (e.g., $4 \times 4 \times 64$).
4. **The Fully Connected (FC) Layer:** We flatten this into a 1024-dimensional vector. This vector doesn't represent pixels anymore; it represents concepts like "has fur," "has four legs," and "pointed ears."
5. **The Classifier:** The final layer looks at these concepts and outputs a probability: 98% Cat, 2% Dog.

![type:video](../media/videos/CNN_Animation/1080p60/CNNArchitecture.mp4)

---

## 9\. The Training Loop

1. **Forward Pass:** The image goes in; the model guesses "Dog."
2. **The Loss:** The **Loss Function** measures the gap between the guess ("Dog") and the truth ("Cat").
3. **Backpropagation:** The error travels backward. It tells the filters: "You were looking for floppy ears, but you should have been looking for pointed ones!"
4. **Optimizer (Adam):** The optimizer tweaks the weights of the filters just a tiny bit ($\text{learning rate} = 0.001$) so the model does better on the next cat it sees.

---

## 10\. Coding with Pytorch

Pytorch is a common library for deep learning, so let's use Pytorch to really make a CNN model that can identify hand-written numbers from 0 to 9 (MNIST)
> (Actually, I find that identifying cat in 32x32 pixel is not pratical and hand-written number can make a interactive window, which makes more fun. So, that's it)

Use Google Colab to run or edit the code. (Click the icon below)

<!-- Google Colab link -->
[![Run in Google Colab](https://i.ibb.co/2P3SLwK/colab.png)](https://colab.research.google.com/github/UMJI-SSTIA/Deep-Learning-Workshop/blob/2026/model/MNIST.ipynb)
{: .center }

### 10.1 Dependencies

```py
from torchvision import datasets
from torchvision.transforms import ToTensor

from torch.utils.data import DataLoader

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
```

### 10.2 Data Preparation

```py
train_data = datasets.MNIST(
    root="data", train=True, transform=ToTensor(), download=True
)

test_data = datasets.MNIST(
    root="data", train=False, transform=ToTensor(), download=True
)

loaders = {
    "train": DataLoader(train_data, batch_size=100, shuffle=True, num_workers=1),
    "test": DataLoader(test_data, batch_size=100, shuffle=True, num_workers=1),
}

```

### 10.3 CNN Model in Pytorch

```py
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return F.softmax(x, dim=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN().to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)

loss_fn = nn.CrossEntropyLoss()
```

### 10.4 Training and Test Loop

```py
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(loaders["train"]):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx*len(data)}/{len(loaders['train'].dataset)} ({100.0 * batch_idx/ len(loaders['train']):.0f}%)]\t{loss.item():.6f}"
            )


def test():
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in loaders["test"]:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(loaders["test"].dataset)
    print(
        f"\nTest set: Average loss: {test_loss:.4f}, Accuracy {correct}/{len(loaders['test'].dataset)} ({100.0 * correct / len(loaders['test'].dataset):.0f}%)\n"
    )


for epoch in range(1, 6):
    train(epoch)
    test()

model.eval()
```

---

## References

NeuralNine. "PyTorch Project: Handwritten Digit Recognition." *YouTube*, 22, Aug. 2023, <https://www.youtube.com/watch?v=vBlO87ZAiiw>.

SSTIA. "SSTIA Deep Learning Workshop 2025." Source: <https://github.com/UMJI-SSTIA/Deeplearning-wksp-2025/blob/main/Worksheet/Worksheet_Part2.md>
