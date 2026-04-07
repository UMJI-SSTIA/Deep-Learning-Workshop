# PyTorch vs TensorFlow: Core Syntax Comparison (Model Definition, Training Loop, Data Loading)

## Table of Contents

1. [Model Definition](#1-model-definition)
2. [Data Loading](#2-data-loading)
3. [Training Loop](#3-training-loop)
4. [Mixed Precision Training](#4-mixed-precision-training)
5. [Model Persistence](#5-model-persistence)

---

## 1. Model Definition

### PyTorch (Imperative/Object-Oriented)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForwardNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(FeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x

# Initialize model
model = FeedForwardNet(input_dim=784, hidden_dim=256, output_dim=10)
```

### TensorFlow (Declarative/Functional + Subclassing)

```python
import tensorflow as tf
from tensorflow.keras import layers, Model

class FeedForwardNet(Model):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(FeedForwardNet, self).__init__()
        self.fc1 = layers.Dense(hidden_dim, activation=None)
        self.bn1 = layers.BatchNormalization()
        self.dropout = layers.Dropout(rate=0.5)
        self.fc2 = layers.Dense(output_dim, activation=None)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = tf.nn.relu(self.bn1(self.fc1(x), training=training))
        x = self.dropout(x, training=training)
        x = tf.nn.sigmoid(self.fc2(x))
        return x

# Initialize model
model = FeedForwardNet(input_dim=784, hidden_dim=256, output_dim=10)
# Build model (required if subclassing)
model.build(input_shape=(None, 784))
```

### Key Differences:

- PyTorch uses `nn.Module` base class with explicit `forward()` method; TensorFlow uses `Model` base class with `call()` method that accepts `training` flag
- PyTorch relies on `nn.functional` for stateless operations; TensorFlow uses `tf.nn` functional API or Keras layers
- PyTorch initializes model weights on first forward pass; TensorFlow requires explicit `build()` call for subclassed models

---

## 2. Data Loading

### PyTorch (torch.utils.data)

```python
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

# Define custom transform pipeline
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load MNIST dataset
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# Create DataLoader for batching/shuffling
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=2
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=2
)

# Example iteration
for inputs, labels in train_loader:
    print(f"Input batch shape: {inputs.shape}")  # [64, 1, 28, 28]
    print(f"Label batch shape: {labels.shape}")  # [64]
    break
```

### TensorFlow (tf.data.Dataset)

```python
import tensorflow as tf
from tensorflow.keras.datasets import mnist

# Load and preprocess MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

def preprocess(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    x = tf.reshape(x, (-1, 28*28))  # Flatten to [784]
    y = tf.cast(y, tf.int64)
    return x, y

# Create Dataset pipeline
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=10000) \
                             .map(preprocess) \
                             .batch(batch_size=64) \
                             .prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.map(preprocess) \
                           .batch(batch_size=64) \
                           .prefetch(tf.data.AUTOTUNE)

# Example iteration
for inputs, labels in train_dataset.take(1):
    print(f"Input batch shape: {inputs.shape}")  # [64, 784]
    print(f"Label batch shape: {labels.shape}")  # [64]
```

### Key Differences:

- PyTorch uses `Dataset` + `DataLoader` combination; TensorFlow uses `tf.data.Dataset` pipeline API
- PyTorch handles multi-threading via `num_workers` parameter; TensorFlow uses `prefetch(tf.data.AUTOTUNE)` for automatic optimization
- PyTorch supports custom `Dataset` subclasses; TensorFlow supports `from_generator()` for dynamic data sources

---

## 3. Training Loop

### PyTorch (Imperative/Manual)

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Initialize training components
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FeedForwardNet(784, 256, 10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 5

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.view(inputs.size(0), -1)  # Flatten to [64, 784]

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track stats
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss:.3f} | Accuracy: {epoch_acc:.2f}%")

# Evaluation
model.eval()
test_correct = 0
test_total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.view(inputs.size(0), -1)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        test_total += labels.size(0)
        test_correct += predicted.eq(labels).sum().item()

test_acc = 100 * test_correct / test_total
print(f"Test Accuracy: {test_acc:.2f}%")
```

### TensorFlow (Keras API + Custom Loop)

```python
import tensorflow as tf
from tensorflow.keras import optimizers, losses

# Initialize training components
device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
model = FeedForwardNet(784, 256, 10)
model.build(input_shape=(None, 784))
loss_fn = losses.SparseCategoricalCrossentropy()
optimizer = optimizers.Adam(learning_rate=1e-3)
num_epochs = 5

# Training loop
for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_dataset:
        with tf.GradientTape() as tape:
            # Forward pass
            outputs = model(inputs, training=True)
            loss = loss_fn(labels, outputs)

        # Backward and optimize
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Track stats
        running_loss += loss * tf.cast(tf.shape(inputs)[0], tf.float32)
        _, predicted = tf.math.top_k(outputs, k=1)
        predicted = tf.squeeze(predicted, axis=1)
        correct += tf.math.reduce_sum(tf.cast(tf.equal(predicted, labels), tf.float32))
        total += tf.cast(tf.shape(inputs)[0], tf.float32)

    epoch_loss = running_loss / len(x_train)
    epoch_acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss:.3f} | Accuracy: {epoch_acc:.2f}%")

# Evaluation
model.eval()
test_correct = 0
test_total = 0
for inputs, labels in test_dataset:
    outputs = model(inputs, training=False)
    _, predicted = tf.math.top_k(outputs, k=1)
    predicted = tf.squeeze(predicted, axis=1)
    test_correct += tf.math.reduce_sum(tf.cast(tf.equal(predicted, labels), tf.float32))
    test_total += tf.cast(tf.shape(inputs)[0], tf.float32)

test_acc = 100 * test_correct / test_total
print(f"Test Accuracy: {test_acc:.2f}%")
```

### Key Differences:

- PyTorch uses explicit `model.train()`/`model.eval()` and `torch.no_grad()` context manager; TensorFlow uses `training` flag in `call()`
- PyTorch accumulates gradients in `backward()`; TensorFlow uses `tf.GradientTape()` context manager to record operations
- PyTorch optimizer uses `step()` after zeroing gradients; TensorFlow optimizer uses `apply_gradients()` with gradient list
- PyTorch stats tracking uses Python variables; TensorFlow uses tf.Tensor operations for GPU acceleration

---

## 4. Mixed Precision Training

### PyTorch Implementation

PyTorch uses the `torch.cuda.amp` (Automatic Mixed Precision) module to prevent gradient underflow through scaling loss.

```python
from torch.cuda.amp import autocast, GradScaler

# 1. Initialize the GradScaler
scaler = GradScaler()

for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # 2. Use autocast context manager for the forward pass
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        # 3. Scale the loss and backpropagate
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        
        # 4. Update the scale factor for the next iteration
        scaler.update()
```

### TensorFlow Implementation

The mixed-precision configuration of TensorFlow is usually set globally **before** the model is built.

```python
# Configure the policy before building the model
# 1. Configure the global policy before model instantiation
# This tells Keras to use mixed precision (FP16 storage with FP32 compute)
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# --- Model Definition (Unchanged) ---
# The model definition remains exactly the same
# Keras automatically casts layers to float16 where appropriate
model = FeedForwardNet(784, 256, 10)
model.build(input_shape=(None, 784))

# --- Training Loop (Unchanged) ---
# The training loop logic remains the same
# However, under the hood, it's now using mixed precision
# This provides a speedup without requiring code changes in the loop
optimizer = tf.keras.optimizers.Adam()
```

## 5. Model Persistence

PyTorch tends to save the state dictionary (with high flexibility), while TensorFlow tends to save the entire computational graph (deploymental-friendly)

### PyTorch:

```Python
import torch
import torch.nn as nn

# --- Saving ---
# Option 1: Recommended (Saves only weights)
torch.save(model.state_dict(), 'model_weights.pth')

# Option 2: Saves entire model (Less flexible)
torch.save(model, 'full_model.pth')

# --- Loading ---
# When loading weights, you must first instantiate the model class
model = FeedForwardNet(784, 256, 10) 

# Option 1: Load weights
model.load_state_dict(torch.load('model_weights.pth'))
model.eval() # Always set to eval mode after loading

# Option 2: Load entire model
# loaded_model = torch.load('full_model.pth')
```

### TensorFlow:

```Python
import tensorflow as tf

# --- Saving ---
# Option 1: SavedModel format (Default, recommended for TF serving)
model.save('saved_model_directory/')

# Option 2: HDF5 format (Single file, similar to PyTorch)
model.save('model_weights.h5', save_format='h5')

# --- Loading ---
# For SavedModel
loaded_model = tf.keras.models.load_model('saved_model_directory/')

# For HDF5
# loaded_model = tf.keras.models.load_model('model_weights.h5')

# Evaluate or predict directly
loaded_model.predict(inputs)
```

Both frameworks offer excellent capabilities, with PyTorch being preferred for research due to its flexibility and TensorFlow for production due to its scalability and deployment tools (TensorFlow Lite, TensorRT, etc.). You could construct high-performance deep learning models with both!！
