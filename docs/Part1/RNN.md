---
discussions: true
---

# Recurrent Neural Network

While FNNs are excellent for mapping fixed-size inputs to fixed-size outputs and CNNs excel at processing spatial data (like images), they both share a limitation: they have no memory of previous inputs. They assume all inputs (and outputs) are independent of each other.

RNNs are designed to solve this by handling **sequential data** (e.g., time series, natural language, audio). They do this by introducing a "hidden state" that acts as an internal memory, allowing information to persist from one step of the sequence to the next.

So, in this section, we are going to learn how to make a RNN to write like **Shakespeare**.

---

## 11\. The Core Principle: Unrolling Through Time

An RNN processes a sequence of inputs one element at a time. At each time step $t$, the network takes two inputs:

1. The current input in the sequence, $x_t$
2. The hidden state from the previous time step, $h_{t-1}$

By combining these, the RNN generates a new hidden state $h_t$ and an output $y_t$.

To understand how an RNN is trained, we conceptually "unroll" it. Unrolling simply means writing out the network for the complete sequence. For a sequence of 3 words, the network would be unrolled into a 3-layer neural network, with one layer for each word.

---

## 12\. The Mathematics of the Forward Pass

Let's formalize the forward propagation of an RNN. Assume our input sequence is $x = (x_1, x_2, \dots, x_T)$.

The network uses three sets of weight matrices:

* $W_{hx}$: Weights for the current input $x_t$
* $W_{hh}$: Weights for the previous hidden state $h_{t-1}$ (the recurrent weights)
* $W_{yh}$: Weights mapping the hidden state to the output $y_t$

At any given time step $t$, the forward pass equations are:

### Step 1: Updating the Hidden State

The new hidden state $h_t$ is a function of the previous hidden state and the current input:

$$h_t = \sigma_h(W_{hh} h_{t-1} + W_{hx} x_t + b_h)$$

* $\sigma_h$: The hidden layer activation function (typically $\tanh$ or ReLU). We use $\tanh$ because it keeps the values normalized between $[-1, 1]$, which prevents the hidden state from growing too large over many time steps.
* $b_h$: The bias vector for the hidden state.
* $h_0$: The initial hidden state (usually initialized to a vector of zeros).

### Step 2: Calculating the Output

The output at time step $t$ is calculated solely from the current hidden state $h_t$:

$$y_t = \sigma_y(W_{yh} h_t + b_y)$$

* $\sigma_y$: The output activation function (e.g., Softmax for classification, linear for regression).
* $b_y$: The bias vector for the output.

> **Key Takeaway:** Notice that the weight matrices ($W_{hh}$, $W_{hx}$, $W_{yh}$) are shared across all time steps. This weight sharing greatly reduces the number of parameters the network needs to learn and allows it to generalize across sequences of varying lengths.

![type:video](../media/videos/RNN_Animation/1080p60/RNNVisualization.mp4)

---

## 13\. Backpropagation Through Time (BPTT)

Because an RNN is essentially a deep neural network unrolled over time, we use an extension of standard backpropagation called **Backpropagation Through Time (BPTT)** to train it.

To find the gradients, we need to calculate how the total loss $L$ changes with respect to our weight matrices. The total loss for a sequence is the sum of the losses at each time step:

$$L = \sum_{t=1}^T L_t$$

Let's look at the gradient of the loss with respect to the recurrent weight matrix $W_{hh}$. By the chain rule, the gradient at a specific time step $t$ depends on the current hidden state $h_t$. However, $h_t$ depends on $h_{t-1}$, which depends on $h_{t-2}$, and so on.

We must sum up the contributions of $W_{hh}$ from the current time step $t$ all the way back to $k=1$:

$$\frac{\partial L_t}{\partial W_{hh}} = \sum_{k=1}^t \frac{\partial L_t}{\partial y_t} \frac{\partial y_t}{\partial h_t} \frac{\partial h_t}{\partial h_k} \frac{\partial h_k}{\partial W_{hh}}$$

### 13.1 The Vanishing Gradient Problem

The critical term in BPTT is $\frac{\partial h_t}{\partial h_k}$, which measures how the hidden state at step $t$ is affected by the hidden state at an earlier step $k$. This term is actually a chain of partial derivatives itself:

$$\frac{\partial h_t}{\partial h_k} = \prod_{i=k+1}^t \frac{\partial h_i}{\partial h_{i-1}}$$

Because $h_i = \tanh(W_{hh} h_{i-1} + \dots)$, the derivative $\frac{\partial h_i}{\partial h_{i-1}}$ involves the weight matrix $W_{hh}$ and the derivative of the $\tanh$ function.

If the largest eigenvalue of the weight matrix $W_{hh}$ is less than 1, multiplying this matrix by itself many times (over long sequences) causes the gradients to shrink exponentially fast toward zero. This is the vanishing gradient problem.

When gradients vanish, the network cannot learn long-range dependencies because the updates to the weights become microscopic. (Conversely, if the eigenvalues are greater than 1, you face the exploding gradient problem, where gradients become massive and cause the model to diverge).

I'll take that "yes" as a green light to dive into the natural next step! Let's continue the tutorial by exploring how Long Short-Term Memory (LSTM) networks mathematically solve the vanishing gradient problem we just identified.

![type:video](../media/videos/RNN_Animation/1080p60/VanishingGradientBPTT.mp4)

### 13.2 Long Short-Term Memory

Standard RNNs struggle to learn long-range dependencies because of the vanishing gradient problem caused by repeated matrix multiplications during BPTT.

LSTMs, introduced by Hochreiter and Schmidhuber in 1997, solve this by introducing a new internal structure: a **Cell State** and a set of **Gates**.

In a standard RNN, the hidden state $h_t$ is the only form of memory. In an LSTM, we add a second line of memory called the **Cell State**, denoted as $C_t$.

Think of the cell state as a conveyor belt or highway that runs straight down the entire chain of unrolled sequence layers. It has only minor linear interactions, making it very easy for information—and gradients—to flow along it unchanged.

The LSTM uses "gates" to carefully control what information is added to or removed from this cell state.

### 13.3 The Mathematics of LSTM Gates

An LSTM cell takes three inputs: the current input $x_t$, the previous hidden state $h_{t-1}$, and the previous cell state $C_{t-1}$. Inside the cell, data passes through four distinct mathematical operations.

*(Note: In the equations below, $[h_{t-1}, x_t]$ denotes the concatenation of the previous hidden state and the current input vectors. The $\odot$ symbol represents the Hadamard product, which is element-wise multiplication.)*

#### Step 1: The Forget Gate

First, the LSTM decides what information it should throw away from the previous cell state. This decision is made by a sigmoid layer called the "forget gate layer."

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

The sigmoid function ($\sigma$) outputs a vector with numbers between $0$ and $1$. A $1$ means "keep this completely," while a $0$ means "get rid of this completely."

#### Step 2: The Input Gate and Candidate Values

Next, the LSTM decides what new information to store in the cell state. This has two parts:

1. **The Input Gate Layer:** A sigmoid layer decides which specific values in the cell state we will update.

    $$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

2. **The Candidate Layer:** A $\tanh$ layer creates a vector of new candidate values, $\tilde{C}_t$, that could be added to the state.

    $$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

#### Step 3: Updating the Cell State

Now, we mathematically update the old cell state $C_{t-1}$ into the new cell state $C_t$. We multiply the old state by $f_t$ (forgetting the things we decided to forget), and then add $i_t \odot \tilde{C}_t$ (the new candidate values, scaled by how much we decided to update each state value).

$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

#### Step 4: The Output Gate (Updating the Hidden State)

Finally, we decide what we are going to output. This output (the new hidden state $h_t$) will be based on our newly updated cell state, but filtered.

1. **The Output Gate:** We run a sigmoid layer to decide what parts of the cell state we are going to output.

    $$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

2. **The Hidden State Update:** We push the cell state through $\tanh$ (to push the values to be between $-1$ and $1$) and multiply it by the output gate.

    $$h_t = o_t \odot \tanh(C_t)$$

![type:video](../media/videos/RNN_Animation/1080p60/LSTMMathGates.mp4)

### 13.4 How LSTMs Mathematically Solve Vanishing Gradients

To understand why LSTMs solve the vanishing gradient problem, we need to look back at the troublesome derivative from BPTT: $\frac{\partial C_t}{\partial C_{t-1}}$. (In an LSTM, the primary gradient highway is the cell state $C$, rather than the hidden state $h$).

If we take the derivative of the cell state update equation with respect to $C_{t-1}$, we get:

$$\frac{\partial C_t}{\partial C_{t-1}} = f_t + \dots$$

*(There are other terms because $f_t$, $i_t$, and $\tilde{C}_t$ also indirectly depend on $C_{t-1}$ via $h_{t-1}$, but $f_t$ is the dominant term here).*

This is the magic of LSTMs: In a standard RNN, the hidden state update relies heavily on matrix multiplication ($W_{hh}$). Repeated multiplication during backpropagation causes gradients to vanish.

In an LSTM, the cell state update relies on addition. The gradient $\frac{\partial C_t}{\partial C_{t-1}}$ contains the forget gate $f_t$ as an isolated term. If the network learns to set the forget gate $f_t \approx 1$ (meaning "remember this"), the gradient can pass backward through the cell state unhindered, flowing across hundreds of time steps without vanishing. The addition operation creates a mathematical "shortcut" for gradients to flow backward through time.

![type:video](../media/videos/RNN_Animation/1080p60/LSTMvsRNNGradientSolve.mp4)

---

## 14\. Make Shakespearean Text With RNN

So, let's use LSTM to make a model that can generate shakespearean text with Pytorch.

Use Google Colab to run or edit the code. (Click the icon below)

<!-- Google Colab link -->
[![Run in Google Colab](https://i.ibb.co/2P3SLwK/colab.png)](https://colab.research.google.com/github/UMJI-SSTIA/Deep-Learning-Workshop/blob/2026/model/NanoShakespeare.ipynb)
{: .center }

### 14.1 Dependencies

```bash
! wget https://raw.githubusercontent.com/UMJI-SSTIA/Deep-Learning-Workshop/2026/docs/assets/ShakespeareData.txt
```

```py
import torch
import torch.nn as nn
from torch.nn import functional as F
import time
```

### 14.2 Hyperparameters

```py
# --- HYPERPARAMETERS ---

batch_size = 32
block_size = 256
max_iters = 2000
eval_interval = 500
learning_rate = 6e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 100
n_embd = 384
n_layer = 3
dropout = 0.5

# ----------------------------------------------------

torch.manual_seed(1337)
```

### 14.3 Data Preparation

```py
# Load data

with open("ShakespeareData.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

# Move data to device

data = torch.tensor(encode(text), dtype=torch.long).to(device)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data_source = train_data if split == "train" else val_data
    ix = torch.randint(len(data_source) - block_size, (batch_size,))
    x = torch.stack([data_source[i : i + block_size] for i in ix])
    y = torch.stack([data_source[i + 1 : i + block_size + 1] for i in ix])
    return x, y
```

### 14.4 LSTM Model

```py
class LSTMLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        self.lstm = nn.LSTM(
            input_size=n_embd,
            hidden_size=n_embd,
            num_layers=n_layer,
            dropout=dropout,
            batch_first=True,
        )
        self.ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None, hidden=None):
        x = self.token_embedding_table(idx)
        x, hidden = self.lstm(x, hidden)
        x = self.ln(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss, hidden

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        # Initial context processing
        logits, _, hidden = self(idx)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        next_input = torch.multinomial(probs, num_samples=1)
        generated = torch.cat((idx, next_input), dim=1)

        # Stateful loop
        for _ in range(max_new_tokens - 1):
            logits, _, hidden = self(next_input, hidden=hidden)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_input = torch.multinomial(probs, num_samples=1)
            generated = torch.cat((generated, next_input), dim=1)
        return generated

model = LSTMLanguageModel().to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)

```

### 14.5 Training Loop

```py
def estimate_loss(model):
    """Helps us see how the model is doing on both train and val data"""
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss, _ = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

print(f"Training on {device}...")
start_time = time.time()

for iter in range(max_iters):
    # --- VALIDATION STEP ---
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(model)
        elapsed = time.time() - start_time
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f} | time: {elapsed:.2f}s"
        )

    xb, yb = get_batch("train")
    logits, loss, _ = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

```

### 14.6 Generation

```py
model.eval()
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print("\n--- GENERATED TEXT ---\n")
print(decode(model.generate(context, max_new_tokens=1500)(0).tolist()))
```

---

## References

Andrej Karpathy. "Let's build GPT: from scratch, in code, spelled out." *YouTube*, 18, Jan. 2023, <https://www.youtube.com/watch?v=kCc8FmEb1nY>.
