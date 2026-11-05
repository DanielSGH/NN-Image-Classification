#!/usr/bin/env python
# coding: utf-8

# # IDL Assignment 1 - Task 2

# In[ ]:


from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
import torchvision
#import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ## Prototyping Phase

# First, we load the data, and take a look at its dimensions:

# In[ ]:


sample_images = np.load("A1 Data/A1_data_75/images.npy")
sample_labels = np.load("A1 Data\\A1_data_75\\labels.npy")

all_images = np.load("A1 Data/A1_data_150/images.npy")
all_labels = np.load("A1 Data/A1_data_150/labels.npy")

print(sample_images)
plt.figure(figsize=(4, 4))
plt.imshow(sample_images[12], cmap="gray")
plt.title(f"Hour: {sample_labels[12]}")
plt.axis("off")
plt.show()

# In[3]:


sample_labels

# ## Regressor Model

# Here, we needed a way to transform our current categorical labels, we could get a continuous version of it, in order to build the regressor model for this part of the task.

# In[4]:


def transform_labels(sample):
    val = float(sample[0])
    if not 0 <= sample[1] <= 59:
        raise ValueError("minutes placement must be between 0 and 59")
    
    floating = sample[1] / 60

    return val + floating

# In[ ]:


float_labels = np.array([transform_labels(x) for x in all_labels])
float_labels[20:60]

# We need to transform our current data into pytorch tensors, so it can be processed by our NN (such is a requirement for PyTorch NN Models):

# In[ ]:


# Transform data, so it can be processed by the PyTorch NN
X = torch.from_numpy(all_images)
y = torch.from_numpy(float_labels)

dataset = TensorDataset(X, y)

# In[7]:


class NNRegressor(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(NNRegressor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=5)
        self.conv2 = nn.Conv2d(256, 64, kernel_size=5)
        self.conv3 = nn.Conv2d(64, 16, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=0.1)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.max_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.nn = nn.Sequential(
            nn.Linear(16*5*5, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(16, out_features=num_classes),
            #nn.Softmax()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        x = self.conv2_drop(x)
        x = self.conv3(x)
        x = self.max_pool3(x)
        
        x = torch.flatten(x, start_dim=1)
        
        x = self.nn(x)
        return x

# This is our first prototype to be trained. We made it "wide and tall" so we could push its capacity to the extreme and see how it behaves with our data. 

# Training Loop

# In[ ]:


learning_rate = 0.001
batch_size = 64
epochs = 50

generator = torch.Generator().manual_seed(42) # for reproducibility purposes

train_set, val_set, test_set = random_split(
    dataset, [0.8, 0.1, 0.1], generator=generator
)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size)
test_loader = DataLoader(test_set, batch_size=batch_size)


model = NNRegressor(in_channels=1, num_classes=1).to(device)
loss_function = nn.SmoothL1Loss(beta=0.25)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.015)

# In[19]:


# auxiliary function to smooth out the graph plots
def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    values = [value.detach().cpu().float().mean().item() if torch.is_tensor(value) else float(value) for value in y]
    values = np.asarray(values, dtype=float)
    weights = np.ones(box_pts) / box_pts
    smoothed = np.convolve(values, weights, mode="same")
    counts = np.convolve(np.ones_like(values), weights, mode="same")
    return smoothed / counts

# In[ ]:


def common_sense_error_minutes(pred, target):
    def to_clock_minutes(x):
        x = torch.remainder(x.squeeze(-1), 12)
        hour = torch.floor(x)
        minute = (x - hour) * (59 / 0.99)  # invert your label encoding
        return torch.remainder(hour * 60 + minute, 720)

    pred_minutes = to_clock_minutes(pred)
    target_minutes = to_clock_minutes(target)

    difference = torch.abs(pred_minutes - target_minutes)
    return torch.minimum(difference, 720 - difference)

# In[ ]:


# training process
train_loss_history = []
val_loss_history = []
best_val_sc = 0.0
best_model = None
for n in range(epochs):
    model.train()
    cumulated_loss = torch.zeros((), device=device)

    for X, y in train_loader:
        X = X.float().to(device).unsqueeze(1)
        y = y.float().to(device).unsqueeze(1)

        optimizer.zero_grad()
        pred = model(X)
        loss = loss_function(pred, y)
        loss.backward()
        optimizer.step()

        cumulated_loss += loss.detach()

    # validation
    with torch.no_grad():
        model.eval()
        val_losses = []
        for X_val, y_val in val_loader:
            X_val = X_val.float().to(device).unsqueeze(1)
            y_val = y_val.float().to(device).unsqueeze(1)

            pred_val = model(X_val)
            val_loss = common_sense_error_minutes(pred_val, y_val)
            val_losses.append(val_loss.item())
            print(f"Validation loss: {val_loss.item():.4f}")

    print(f"End of epoch. Cumulated loss: {cumulated_loss.item():.4f}")
    print(f"Validation loss: {torch.tensor(val_losses).mean().item():.4f}")
    val_loss_history.append(torch.tensor(val_losses).mean().item())
    train_loss_history.append(cumulated_loss.item() / len(train_loader))

    if torch.tensor(val_losses).mean().item() < best_val_sc or best_model is None:
        best_val_sc = torch.tensor(val_losses).mean().item()
        best_model = model.state_dict()
        print(f"New best model found with validation loss: {best_val_sc:.4f}")

epochs = range(1, len(train_loss_history) + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs, smooth(train_loss_history, 19))
plt.plot(epochs, smooth(val_loss_history, 19), color="orange")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss over Iterations")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Even though we got an error for this first try, we can see that by merely running this model's training process on a cpu is not feasible.

# 
# ##### Evaluation Phase

# In[ ]:


model.eval()  # disables dropout
val_error_sum = 0.0
num_examples = 0

with torch.inference_mode():  # no gradients needed for validation
    for X, y in val_loader:
        X = X.float().to(device).unsqueeze(1)
        y = y.float().to(device).unsqueeze(1)

        pred = model(X)
        batch_error = common_sense_error_minutes(pred, y)

        # Count predictions no more than the chosen circular-time error tolerance.
        val_error_sum += batch_error.sum().item()
        num_examples += X.size(0)

mean_error_minutes = val_error_sum / num_examples
print(f"Mean circular error: {mean_error_minutes:.2f} minutes")

# It took too long...let's try a different, smaller architecure instead!

# In[21]:


class SmallerNNRegressor(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SmallerNNRegressor, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, dilation=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 32, kernel_size=5, dilation=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten()

        )
        self.fn = nn.Sequential(
            nn.Linear(32*4*4, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(16, out_features=num_classes),
            #nn.Softmax()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.fn(x)
        return x

# In[ ]:


model = SmallerNNRegressor(in_channels=1, num_classes=1).to(device)
loss_function = nn.SmoothL1Loss(beta=4.0)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.015)

# training process
train_loss_history = []
val_loss_history = []
best_val_sc = 0.0
best_model = None
for n in range(epochs):
    model.train()
    cumulated_loss = torch.zeros((), device=device)

    for X, y in train_loader:
        X = X.float().to(device).unsqueeze(1)
        y = y.float().to(device).unsqueeze(1)

        optimizer.zero_grad()
        pred = model(X)
        loss = loss_function(pred, y)
        loss.backward()
        optimizer.step()

        cumulated_loss += loss.detach()

    # validation
    with torch.no_grad():
        model.eval()
        val_losses = []
        for X_val, y_val in val_loader:
            X_val = X_val.float().to(device).unsqueeze(1)
            y_val = y_val.float().to(device).unsqueeze(1)

            pred_val = model(X_val)
            val_loss = common_sense_error_minutes(pred_val, y_val)
            val_losses.append(val_loss.item())
            print(f"Validation loss: {val_loss.item():.4f}")

    print(f"End of epoch. Cumulated loss: {cumulated_loss.item():.4f}")
    print(f"Validation loss: {torch.tensor(val_losses).mean().item():.4f}")
    val_loss_history.append(torch.tensor(val_losses).mean().item())
    train_loss_history.append(cumulated_loss.item() / len(train_loader))

    if torch.tensor(val_losses).mean().item() < best_val_sc or best_model is None:
        best_val_sc = torch.tensor(val_losses).mean().item()
        best_model = model.state_dict()
        print(f"New best model found with validation loss: {best_val_sc:.4f}")

# In[ ]:


epochs = range(1, len(train_loss_history) + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs, smooth(train_loss_history, 19))
plt.plot(epochs, smooth(val_loss_history, 19), color="orange")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss over Iterations for Smaller Model")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# #### Evaluation Stage

# In[20]:


model.eval()  # disables dropout
val_error_sum = 0.0
num_examples = 0

error_history = []

with torch.inference_mode():  # no gradients needed for validation
    for X, y in val_loader:
        X = X.float().to(device).unsqueeze(1)
        y = y.float().to(device).unsqueeze(1)

        pred = model(X)
        batch_error = common_sense_error_minutes(pred, y)
        error_history.append(batch_error)

        # Count predictions no more than the chosen circular-time error tolerance.
        val_error_sum += batch_error.sum().item()
        num_examples += X.size(0)

mean_error_minutes = val_error_sum / num_examples
print(f"Mean circular error: {mean_error_minutes:.2f} minutes")

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(error_history) + 1), smooth(error_history, 3))
plt.xlabel("Data Iteration")
plt.ylabel("Common Sense Error Value")
plt.title("Evaluation Metric Over Common Sense Error")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Very good already! But can we try to smooth even more the loss landscape?

# In[ ]:


class Block(nn.Module):
    def __init__(self, ci,co,pool=True):
        super().__init__()
        layers= [ nn.Conv2d(ci,co,3,padding=1),
        nn.BatchNorm2d(co),
        nn.ReLU(inplace=True),
        nn.Conv2d(co,co,3,padding=1),
        nn.BatchNorm2d(co),
        nn.ReLU(inplace=True),]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class BatchNormalizedNN(nn.Module):
    def __init__(self, in_channels, num_classes):
            super(BatchNormalizedNN, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=5, dilation=2),
                nn.BatchNorm2d(num_features=32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
    
                nn.Conv2d(32, 64, kernel_size=5, padding=1),
                nn.BatchNorm2d(num_features=64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
    
                nn.Conv2d(64, 32, kernel_size=5, dilation=2),
                nn.BatchNorm2d(num_features=32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
    
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten()
    
            )
            self.fn = nn.Sequential(
                nn.Linear(32*4*4, 64),
                nn.ReLU(),
                nn.Dropout(p=0.2),
    
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(p=0.2),
    
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Dropout(p=0.2),
    
                nn.Linear(16, out_features=num_classes),
                #nn.Softmax()
            )
    
    def forward(self, x):
        x = self.features(x)
        x = self.fn(x)
        return x

# Here, on this approach we implemented batch normalization to re-center and re-scale the activation inputs of each hidden layer of our convolutional phase, thus minimizing big input values differences.

# In[ ]:


model = BatchNormalizedNN(in_channels=1, num_classes=1).to(device)
loss_function = nn.SmoothL1Loss(beta=4.0)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.015)
scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

# training process
train_loss_history = []
val_loss_history = []
best_val_sc = 0.0
best_model = None
for n in range(epochs):
    model.train()
    cumulated_loss = torch.zeros((), device=device)

    for X, y in train_loader:
        X = X.float().to(device).unsqueeze(1)
        y = y.float().to(device).unsqueeze(1)

        optimizer.zero_grad()
        pred = model(X)
        loss = loss_function(pred, y)
        loss.backward()
        optimizer.step()

        cumulated_loss += loss.detach()

    # validation
    with torch.no_grad():
        model.eval()
        val_losses = []
        for X_val, y_val in val_loader:
            X_val = X_val.float().to(device).unsqueeze(1)
            y_val = y_val.float().to(device).unsqueeze(1)

            pred_val = model(X_val)
            val_loss = common_sense_error_minutes(pred_val, y_val)
            val_losses.append(val_loss.item())
            print(f"Validation loss: {val_loss.item():.4f}")

    print(f"End of epoch. Cumulated loss: {cumulated_loss.item():.4f}")
    print(f"Validation loss: {torch.tensor(val_losses).mean().item():.4f}")
    val_loss_history.append(torch.tensor(val_losses).mean().item())
    train_loss_history.append(cumulated_loss.item() / len(train_loader))

    scheduler.step()

    if torch.tensor(val_losses).mean().item() < best_val_sc or best_model is None:
        best_val_sc = torch.tensor(val_losses).mean().item()
        best_model = model.state_dict()
        print(f"New best model found with validation loss: {best_val_sc:.4f}")

epochs = range(1, len(train_loss_history) + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs, smooth(train_loss_history, 19))
plt.plot(epochs, smooth(val_loss_history, 19), color="orange")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss over Iterations for Batch Normalized Model")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# #### Evaluation Stage

# In[26]:


model.eval()  # once more, to disable dropout
val_loss_sum = 0.0
num_examples = 0

error_history = []

with torch.inference_mode():
    for X, y in val_loader:
        X = X.float().to(device).unsqueeze(1)
        y = y.float().to(device).unsqueeze(1)

        pred = model(X)
        batch_error = common_sense_error_minutes(pred, y)
        error_history.append(batch_error)

        # Count predictions no more than the chosen circular-time error tolerance.
        val_error_sum += batch_error.sum().item()
        num_examples += X.size(0)

mean_error_minutes = val_error_sum / num_examples
print(f"Mean circular error: {mean_error_minutes:.2f} minutes")

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(error_history) + 1), smooth(error_history, 3))
plt.xlabel("Data Iteration")
plt.ylabel("Common Sense Error Value")
plt.title("Evaluation Metric Over Common Sense Error (BatchNormalized Version)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# A slightly better loss.

# Let's test it!

# In[27]:


#testing phase
test_loss_sum = 0.0
num_examples = 0

test_history = []

model.eval()
with torch.inference_mode():
    for X, y in test_loader:
        X = X.float().to(device).unsqueeze(1)
        y = y.float().to(device).unsqueeze(1)

        pred = model(X)

        test_error = common_sense_error_minutes(pred, y)
        test_history.append(test_error)

        # Weight each batch by its size, including a possibly smaller last batch
        val_loss_sum += test_error.mean().item() * X.size(0)
        num_examples += X.size(0)

avg_val_loss = val_loss_sum / num_examples
print(f"Test loss: {avg_val_loss:.4f}")
        

# Still, the Test Loss is too big. What if the problem may reside in the mismatch between the training loss and the defined metric?

# Lets explore that!

# In[28]:


# new (circular-) loss
def circular_smoothl1(pred, target, beta=0.05):
    difference = torch.remainder(pred - target, 12.0)

    d = torch.minimum(difference, 12.0 - difference)
    loss = nn.SmoothL1Loss(beta=beta)
    return loss(d, torch.zeros_like(d))

# ##### Training Phase (again)

# In[ ]:


model = BatchNormalizedNN(in_channels=1, num_classes=1).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.015)
scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

# training process
train_loss_history = []
val_loss_history = []
best_val_sc = 0.0
best_model = None
for n in range(epochs):
    model.train()
    cumulated_loss = torch.zeros((), device=device)

    for X, y in train_loader:
        X = X.float().to(device).unsqueeze(1)
        y = y.float().to(device).unsqueeze(1)

        optimizer.zero_grad()
        pred = model(X)
        loss = circular_smoothl1(pred, y, beta=0.25)
        loss.backward()
        optimizer.step()

        cumulated_loss += loss.detach()

    # validation
    with torch.no_grad():
        model.eval()
        val_losses = []
        for X_val, y_val in val_loader:
            X_val = X_val.float().to(device).unsqueeze(1)
            y_val = y_val.float().to(device).unsqueeze(1)

            pred_val = model(X_val)
            val_loss = common_sense_error_minutes(pred_val, y_val)
            val_losses.append(val_loss.item())
            print(f"Validation loss: {val_loss.item():.4f}")

    print(f"End of epoch. Cumulated loss: {cumulated_loss.item():.4f}")
    print(f"Validation loss: {torch.tensor(val_losses).mean().item():.4f}")
    val_loss_history.append(torch.tensor(val_losses).mean().item())
    train_loss_history.append(cumulated_loss.item() / len(train_loader))

    scheduler.step()

    if torch.tensor(val_losses).mean().item() < best_val_sc or best_model is None:
        best_val_sc = torch.tensor(val_losses).mean().item()
        best_model = model.state_dict()
        print(f"New best model found with validation loss: {best_val_sc:.4f}")

epochs = range(1, len(train_loss_history) + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs, smooth(train_loss_history, 19))
plt.plot(epochs, smooth(val_loss_history, 19), color="orange")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss over Iterations for Batch Normalized Model")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# #### Evaluation Phase

# In[34]:


eval_loss = 0.0
num_examples = 0

error_history = []

model.eval()
with torch.inference_mode():
    for X, y in val_loader:
        X = X.float().to(device).unsqueeze(1)
        y = y.float().to(device).unsqueeze(1)

        pred = model(X)

        batch_error = common_sense_error_minutes(pred, y)
        error_history.append(batch_error)
        
        # Count predictions no more than the chosen circular-time error tolerance.
        val_error_sum += batch_error.sum().item()
        num_examples += X.size(0)
        
mean_error_minutes = val_error_sum / num_examples
print(f"Mean circular error: {mean_error_minutes:.2f} minutes")
        
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(error_history) + 1), smooth(error_history, 3))
plt.xlabel("Data Iteration")
plt.ylabel("Common Sense Error Value")
plt.title("Evaluation Metric Over Common Sense Error (BatchNormalized Version)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# #### Test Phase:

# In[35]:


#testing phase
test_loss_sum = 0.0
num_examples = 0

test_history = []

model.eval()
with torch.inference_mode():
    for X, y in test_loader:
        X = X.float().to(device).unsqueeze(1)
        y = y.float().to(device).unsqueeze(1)

        pred = model(X)

        test_error = common_sense_error_minutes(pred, y)
        test_history.append(test_error)

        # Weight each batch by its size, including a possibly smaller last batch
        val_loss_sum += test_error.mean().item() * X.size(0)
        num_examples += X.size(0)

avg_val_loss = val_loss_sum / num_examples
print(f"Test loss: {avg_val_loss:.4f}")

# ## Trigonometric Encoding Model

# In[37]:


def trigonometric_encoding(sample):
    hour_angle = (2*np.pi * (sample[0] + sample[1] / 60)) / 12
    min_angle = (2* np.pi * sample[1]) / 60

    return [hour_angle, min_angle]

# In[ ]:


test = trigonometric_encoding(all_labels[2])
print("Time: ", all_labels[312])
print("Angle: ", test)

# In[39]:


class NNTrigonometricEnconder(nn.Module):
    def __init__(self, in_channels, num_classes=4, **kwargs: Any):
        super(NNTrigonometricEnconder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 32, kernel_size=6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten()
        )

        self.nn = nn.Sequential(
            nn.Linear(32*4*4, 132),
            nn.ReLU(),

            nn.Linear(132, 64),
            nn.ReLU(),
            nn.Dropout(p=0.4),

            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(p=0.4),

            nn.Linear(64, num_classes),
            nn.Tanh() # here, in the output layer, it is pertinent to advocate for the use of a tangent function
                      # for we want to output the sine and cosine values of our angles, 
                      # thus limiting our possible values to (-1, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.nn(x)
        return x

# In[ ]:


# Transform data, so it can be processed by the PyTorch NN
angle_labels = np.array([trigonometric_encoding(sample) for sample in all_labels])

sin_cos_labels = np.array([[np.cos(x[0]), np.sin(x[0]), np.cos(x[1]), np.sin(x[1])] for x in angle_labels])
X = torch.from_numpy(all_images)
y = torch.from_numpy(sin_cos_labels)

print("X: ", X.size())
print("y: ", y.size())

dataset = TensorDataset(X, y)

# Some Hyperparameters
learning_rate = 1e-4
batch_size = 64
epochs = 50

#generator = torch.Generator().manual_seed(42) # for reproducibility purposes

train_set, val_set, test_set = random_split(
    dataset, [0.8, 0.1, 0.1], generator=generator
)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size)
test_loader = DataLoader(test_set, batch_size=batch_size)


# In[ ]:


# Error handler for our trigonometric model's evaluation
def common_sense_error_trigonometric(pred, target):
    error = 0

    # The hour-hand angle represents a 12-hour (720-minute) cycle.
    pred_angle = torch.atan2(pred[..., 1], pred[..., 0])
    target_angle = torch.atan2(target[..., 1], target[..., 0])

    angle_difference = torch.atan2(
        torch.sin(pred_angle - target_angle), # get the sine value from the difference between pred and target angles
        torch.cos(pred_angle)) # same goes for the cossine value

    error = angle_difference.abs() * (720 / (2.0 * torch.pi))
    return error

# In[ ]:


model = NNTrigonometricEnconder(in_channels=1, num_classes=4).to(device)
loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.015)
scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

train_loss_history = []
eval_loss_history = []
best_val_sc = 0.0
best_model = None

for n in range(epochs):
    model.train()
    cumulated_loss = torch.zeros((), device=device)

    for X, y in train_loader:
        X = X.float().to(device).unsqueeze(1)
        y = y.float().to(device)

        optimizer.zero_grad()
        pred = model(X)
        loss = loss_function(pred, y)
        loss.backward()
        optimizer.step()

        cumulated_loss += loss.detach()

    with torch.no_grad():
        model.eval()
        val_losses = []
        for X_val, y_val in val_loader:
            X_val = X_val.float().to(device).unsqueeze(1)
            y_val = y_val.float().to(device)

            pred_val = model(X_val)
            val_loss = common_sense_error_trigonometric(pred_val, y_val)
            val_losses.append(val_loss.item())
            print(f"Validation loss: {val_loss.item():.4f}")
         
    print(f"End of epoch. Cumulated loss: {cumulated_loss.item():.4f}")
    print(f"Validation loss: {torch.tensor(val_losses).mean().item():.4f}")
    val_loss_history.append(torch.tensor(val_losses).mean().item())
    train_loss_history.append(cumulated_loss.item() / len(train_loader))

    scheduler.step()

    if torch.tensor(val_losses).mean().item() < best_val_sc or best_model is None:
        best_val_sc = torch.tensor(val_losses).mean().item()
        best_model = model.state_dict()
        print(f"New best model found with validation loss: {best_val_sc:.4f}")

epochs = range(1, len(train_loss_history) + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs, smooth(train_loss_history, 19))
plt.plot(epochs, smooth(val_loss_history, 19), color="orange")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss over Iterations for Trigonometric Encoder Model")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ### Evaluation Stage

# This was just a mere test to see if I could output different metrics for this model's evaluation:

# In[ ]:


model.eval()  # once more, to disable dropout
val_error_sum = 0.0
correct_within_tolerance = 0
num_examples = 0
tolerance_minutes = 5
all_targets = []
all_predictions = []

model = best_model["model"] # get the best model from our training section
with torch.inference_mode():
    for X, y in val_loader:
        X = X.float().to(device).unsqueeze(1)
        y = y.float().to(device)

        pred = model(X)
        batch_error = common_sense_error_trigonometric(pred, y)

        # Count predictions no more than the chosen circular-time error tolerance.
        val_error_sum += batch_error.sum().item()
        correct_within_tolerance += (batch_error <= tolerance_minutes).sum().item()
        num_examples += X.size(0)
        all_targets.append(y.squeeze(-1).cpu())
        all_predictions.append(pred.squeeze(-1).cpu())

mean_error_minutes = val_error_sum / num_examples
accuracy_within_tolerance = correct_within_tolerance / num_examples
y_true = torch.cat(all_targets).numpy()
y_pred = torch.cat(all_predictions).numpy()

# Score clock times in circular sin/cos space so midnight wraparound is respected.
true_angle = np.remainder(y_true, 12) * (2 * np.pi / 12)
pred_angle = np.remainder(y_pred, 12) * (2 * np.pi / 12)
true_components = np.column_stack((np.sin(true_angle), np.cos(true_angle)))
pred_components = np.column_stack((np.sin(pred_angle), np.cos(pred_angle)))

component_r2 = []
for i in range(2):
    residual_sum = np.sum((true_components[:, i] - pred_components[:, i]) ** 2)
    total_sum = np.sum((true_components[:, i] - np.mean(true_components[:, i])) ** 2)
    component_r2.append(1 - residual_sum / total_sum if total_sum > 0 else np.nan)
r2 = np.nanmean(component_r2)

print(f"Mean circular error: {mean_error_minutes:.2f} minutes")
print(f"Accuracy (within +/-{tolerance_minutes} minutes): {accuracy_within_tolerance:.2%}")
print(f"Circular-embedding R²: {r2:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
component_names = ["sin(time)", "cos(time)"]
for i, ax in enumerate(axes):
    ax.scatter(true_components[:, i], pred_components[:, i], alpha=0.35, s=12)
    ax.plot([-1, 1], [-1, 1], "r--", linewidth=1)
    ax.set_title(f"{component_names[i]} (R²={component_r2[i]:.3f})")
    ax.set_xlabel("Actual component")
    ax.grid(True, alpha=0.3)
axes[0].set_ylabel("Predicted component")
fig.suptitle(f"Validation circular-embedding R² = {r2:.3f}")
fig.tight_layout()
plt.show()

# if the code above does not work, it doesn't matter that much.

# ##### True Evaluation Stage

# In[ ]:


model.eval()  # once more, to disable dropout
val_loss_sum = 0.0
num_examples = 0

error_history = []

model = best_model["model"] # get the best model from our training section

with torch.inference_mode():
    for X, y in val_loader:
        X = X.float().to(device).unsqueeze(1)
        y = y.float().to(device)

        pred = model(X)

        batch_loss = common_sense_error_trigonometric(pred, y)
        error_history.append(batch_loss.mean().item())

        val_loss_sum += batch_loss.mean().item() * X.size(0)
        num_examples += X.size(0)

avg_val_loss = val_loss_sum / num_examples
print(f"Validation loss: {avg_val_loss:.4f}")
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(error_history) + 1), smooth(error_history, 3))
plt.xlabel("Data Iteration")
plt.ylabel("Common Sense Error Value")
plt.title("Evaluation Metric Over Common Sense Error (Trigonometric Version)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Testing Phase:

# In[ ]:


test_loss_sum = 0.0
hour_error_sum = 0.0
minute_error_sum = 0.0 # we thought as important here to make a distinction between 
                       # the error related with the prediction of the angle that the hour pointer
                       # does and the one related with the angle prediction of the minute pointer

num_examples = 0

model.eval() #very similar with the evaluation stage
with torch.inference_mode():
    for X, y in test_loader:
        X = X.float().to(device).unsqueeze(1)
        y = y.float().to(device)

        pred = model(X)
        batch_size = X.size(0)

        test_loss_sum += common_sense_error_trigonometric(pred, y).mean().item() * batch_size
        num_examples += batch_size

avg_test_loss = test_loss_sum / num_examples
print(f"Test loss: {avg_test_loss:.4f}")
