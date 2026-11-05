#!/usr/bin/env python
# coding: utf-8

# # Task 1 - Learn the Basics of PyTorch

# ## Multi Layer Perceptron

# In[18]:


import torch
from torch import nn
from torchvision.datasets import FashionMNIST, CIFAR10
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split
import json

# In[19]:


print("---------------------------------------------------------------------")
print(f"PYTORCH Version: {torch.__version__}")
# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'DEVICE: {device}')

#Additional Info when using cuda
if device.type == 'cuda':
  print(f"GPU NAME: {torch.cuda.get_device_name(0)}")
  print('Memory Usage:')
  print('  - Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
  print('  - Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
  print(f"---------------------------------------------------------------------\n")

# ### Global Variables

# In[20]:


NUM_CLASSES   = 10      # 10
NUM_FEATURES  = 28*28   # 28x28 image
RANDOM_SEED   = 42
BATCH_SIZE    = 64

# In[21]:


# Loading Data - Taken from https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
train_data = FashionMNIST(
  root="data",
  train=True,
  download=True,
  transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
)

test_data = FashionMNIST(
  root="data",
  train=False,
  download=True,
  transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
)

train_data, val_data = random_split(
  train_data, 
  [0.9, 0.1], 
  generator=torch.Generator().manual_seed(RANDOM_SEED)
)

# In[22]:


def move_to_gpu(data, device, shuffle=False):
  X, y = next(iter(DataLoader(data,batch_size=len(data))))
  return DataLoader(
    TensorDataset(X.to(device), y.to(device)),
    batch_size=BATCH_SIZE,
    shuffle=shuffle
  )

# In[23]:



train_dataloader  = move_to_gpu(train_data, device, shuffle=True)
test_dataloader   = move_to_gpu(test_data, device, shuffle=False)
val_dataloader    = move_to_gpu(val_data, device, shuffle=False)

# ### Helper Functions

# In[24]:


def move_to_gpu(data, device, shuffle=False):
  X, y = next(iter(DataLoader(data,batch_size=len(data))))
  return DataLoader(
    TensorDataset(X.to(device), y.to(device)),
    batch_size=BATCH_SIZE,
    shuffle=shuffle
  )

# In[25]:


# Should be Train: 54000 Validation: 6000 Test: 10000
print(f"Train: {len(train_data)}  Validation: {len(val_data)}  Test: {len(test_data)}") 

# In[26]:


torch.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ## Model

# ### Architecture

# In[27]:


class FashionModel(nn.Module):
  def __init__(self, input_features, output_features, hidden_units=8, activation_fn=nn.ReLU,
               num_hidden_layers=2, dropout=0.0):
    super().__init__()
    layers = [nn.Flatten()]
    in_features = input_features

    for _ in range(num_hidden_layers):
      layers.append(nn.Linear(in_features, hidden_units))
      layers.append(activation_fn())
      if dropout > 0.0:
        layers.append(nn.Dropout(dropout))
      in_features = hidden_units
      
    layers.append(nn.Linear(in_features, output_features))
    self.layer_stack = nn.Sequential(*layers)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.layer_stack(x)

# ### Initialisation

# In[28]:


def apply_init(model: nn.Module, method: str):
  if method == "default":
    return
  
  for layer in model.layer_stack:
    if isinstance(layer, nn.Linear):
      if method == "kaiming_normal":
        nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
      elif method == "normal_0.01":
        nn.init.normal_(layer.weight, mean=0.0, std=0.01)
      elif method == "zeros":
        nn.init.zeros_(layer.weight)
      nn.init.zeros_(layer.bias)

# ### Loss

# In[29]:


loss_fn = nn.CrossEntropyLoss()

# ## Training and Validation Function

# In[30]:


def train_and_validate(model: nn.Module, optimizer, epochs=100) -> dict[str,list]:
  history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
  print(f"{'Epoch':<8}{'Train Loss':<12}{'Train Acc':<12}{'Val Loss':<12}{'Val Acc'}")

  for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    train_correct = 0
    for X, y in train_dataloader:
      y_pred = model(X)
      loss = loss_fn(y_pred, y)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      epoch_loss += loss.item()
      train_correct += (y_pred.argmax(dim=1) == y).sum().item()

    val_loss = 0.0
    val_correct = 0
    model.eval()
    with torch.inference_mode():
      for X, y in val_dataloader:
        val_pred = model(X)
        val_loss += loss_fn(val_pred, y).item()
        val_correct += (val_pred.argmax(dim=1) == y).sum().item()

    train_loss = epoch_loss / len(train_dataloader)
    train_acc = 100 * train_correct / len(train_data)
    val_loss = val_loss / len(val_dataloader)
    val_acc = 100 * val_correct / len(val_data)

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(f"{epoch:<8}{train_loss:<12.5f}{train_acc:<12.2f}{val_loss:<12.5f}{val_acc:.2f}")
  return history

# ## Testing
# This is only run once after the configurations have been set.
# In short, the top 3 performers on the validation set are tested

# In[31]:


def test(model: nn.Module):
  test_loss = 0.0
  test_correct = 0
  model.eval()
  with torch.inference_mode():
    for X, y in test_dataloader:
      test_pred = model(X)
      test_loss += loss_fn(test_pred, y).item()
      test_correct += (test_pred.argmax(dim=1) == y).sum().item()

  # print(f"\nTest loss: {test_loss / len(test_dataloader):.5f}  "
  #       f"Test acc: {100 * test_correct / len(test_data):.2f}%")

  return test_loss / len(test_dataloader), 100 * test_correct / len(test_data)

# ## Run Script

# In[32]:


# configs = [
#   # {"name": "default",   "lr": 0.1,  "hidden_units": 16, "activation_fn": nn.ReLU, "optimizer": torch.optim.SGD},
#   # {"name": "lr_0.01",   "lr": 0.01, "hidden_units": 16, "activation_fn": nn.ReLU, "optimizer": torch.optim.SGD},      
#   # {"name": "hidden_64", "lr": 0.1,  "hidden_units": 64, "activation_fn": nn.ReLU, "optimizer": torch.optim.SGD}     Note: this is old experimentation, I found that adding more hidden units very slightly improves test accuracy
# ]

activation_functions = {
  "relu":         nn.ReLU,
  "leaky_relu":   nn.LeakyReLU,
  "gelu":         nn.GELU
}

optimizers_and_lrs = [
  {"name": "sgd_momentum", "fn": torch.optim.SGD,   "kwargs": {"lr": 0.01, "momentum": 0.9}},
  {"name": "adam",         "fn": torch.optim.Adam,  "kwargs": {"lr": 0.001}},
  {"name": "adamw",        "fn": torch.optim.AdamW, "kwargs": {"lr": 0.001}},
]

BASE = {
  "activation_fn":     nn.GELU,
  "optimizer":         torch.optim.AdamW,
  "optimizer_kwargs":  {"lr": 1e-3, "weight_decay": 0.01},
  "hidden_units":      64,
  "num_hidden_layers": 2,
  "dropout":           0.0,
  "init":              "default",
}

configs = []

for activation_name, activation_fn in activation_functions.items():
  for opt in optimizers_and_lrs:
    configs.append({
      **BASE,
      "name":             f"{activation_name}_{opt['name']}",
      "activation_fn":    activation_fn,
      "optimizer":        opt["fn"],
      "optimizer_kwargs": opt["kwargs"],
    })

configs += [
  {**BASE, "name": "lr3e-3",         "optimizer_kwargs": {"lr": 3e-3, "weight_decay": 0.01}},
  {**BASE, "name": "wd0.1",          "optimizer_kwargs": {"lr": 1e-3, "weight_decay": 0.1}},
  {**BASE, "name": "dropout0.3",     "dropout": 0.3},
  {**BASE, "name": "init_normal",    "init": "normal_0.01"},
  {**BASE, "name": "init_kaiming",   "init": "kaiming_normal"},
  {**BASE, "name": "hidden256",       "hidden_units": 256},
  {**BASE, "name": "depth4",         "num_hidden_layers": 4},
]

results = {}
models = {}

for cfg in configs:
  torch.manual_seed(RANDOM_SEED)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  model = FashionModel(
    input_features=NUM_FEATURES,
    hidden_units=cfg["hidden_units"],
    output_features=NUM_CLASSES,
    activation_fn=cfg["activation_fn"],
    num_hidden_layers=cfg["num_hidden_layers"],
    dropout=cfg["dropout"]
  ).to(device)
  apply_init(model, cfg["init"])

  optimizer = cfg["optimizer"](params=model.parameters(), **cfg["optimizer_kwargs"])

  print(f"\nSELECTED CONFIG: {cfg['name']}")
  results[cfg["name"]] = train_and_validate(model=model, optimizer=optimizer, epochs=30)
  models[cfg["name"]] = model

  h = results[cfg["name"]]
  print(f"{cfg['name']:<28}train acc {h['train_acc'][-1]:.2f}%   val acc {h['val_acc'][-1]:.2f}%")

with open("results_mlp.json", "w") as f:
  json.dump(results, f)

ranking = sorted(results.items(), key=lambda item: item[1]["val_acc"][-1], reverse=True)

# In[33]:


print("\nLeaderboard (Val Acc%)")
for name, history in ranking:
  print(f"{name:<24}{history['val_acc'][-1]:.2f}%")

# In[34]:


print("\nFinalists on Test Set")
test_results = {}
for name, history in ranking[:3]:
  test_loss, test_acc = test(models[name])
  test_results[name] = {"val_acc": history["val_acc"][-1], "test_acc": test_acc, "test_loss": test_loss}
  print(f"{name:<24}val acc {history['val_acc'][-1]:.2f}%   test acc {test_acc:.2f}%   test loss {test_loss:.5f}")
with open("test_mlp.json", "w") as f:
  json.dump(test_results, f)

# # Part (b)

# ## CNN model

# ### Architecture

# In[35]:


class CNNModel(nn.Module):
  def __init__(self, in_channels, image_size, output_features,
               num_conv_blocks=2, filters=16, kernel_size=3,
               hidden_units=64, activation_fn=nn.ReLU, dropout=0.0):
    super().__init__()

    blocks = []
    channels = in_channels
    for _ in range(num_conv_blocks):
      blocks.append(nn.Conv2d(channels, filters, kernel_size=kernel_size, padding=kernel_size // 2))
      blocks.append(activation_fn())
      blocks.append(nn.MaxPool2d(2))
      channels = filters
    self.features = nn.Sequential(*blocks)

    with torch.no_grad():
      flat_size = self.features(torch.zeros(1, in_channels, image_size, image_size)).numel()

    head = [nn.Flatten(), nn.Linear(flat_size, hidden_units), activation_fn()]
    if dropout > 0.0:
      head.append(nn.Dropout(dropout))
    head.append(nn.Linear(hidden_units, output_features))
    self.classifier = nn.Sequential(*head)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.classifier(self.features(x))

# ## Global Variables

# In[36]:


IN_CHANNELS = 1
IMAGE_SIZE  = 28

# ## Init

# In[37]:


def apply_init_cnn(model: nn.Module, method: str):
  if method == "default":
    return

  for layer in model.modules():
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
      if method == "kaiming_normal":
        nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
      elif method == "normal_0.01":
        nn.init.normal_(layer.weight, mean=0.0, std=0.01)
      elif method == "zeros":
        nn.init.zeros_(layer.weight)
      nn.init.zeros_(layer.bias)

# ## Run Script

# In[38]:


activation_functions = {
  "relu":         nn.ReLU,
  "leaky_relu":   nn.LeakyReLU,
  "gelu":         nn.GELU
}

optimizers_and_lrs = [
  {"name": "sgd_momentum", "fn": torch.optim.SGD,   "kwargs": {"lr": 0.01, "momentum": 0.9}},
  {"name": "adam",         "fn": torch.optim.Adam,  "kwargs": {"lr": 0.001}},
  {"name": "adamw",        "fn": torch.optim.AdamW, "kwargs": {"lr": 0.001}},
]

BASE = {
  "activation_fn":     nn.GELU,
  "optimizer":         torch.optim.AdamW,
  "optimizer_kwargs":  {"lr": 1e-3, "weight_decay": 0.01},
  "num_conv_blocks":   2,
  "filters":           16,
  "kernel_size":       3,
  "hidden_units":      64,
  "dropout":           0.0,
  "init":              "default",
}

configs = []

for activation_name, activation_fn in activation_functions.items():
  for opt in optimizers_and_lrs:
    configs.append({
      **BASE,
      "name":             f"cnn_{activation_name}_{opt['name']}",
      "activation_fn":    activation_fn,
      "optimizer":        opt["fn"],
      "optimizer_kwargs": opt["kwargs"],
    })

configs += [
  {**BASE, "name": "cnn_lr3e-3",       "optimizer_kwargs": {"lr": 3e-3, "weight_decay": 0.01}},
  {**BASE, "name": "cnn_lr3e-4",       "optimizer_kwargs": {"lr": 3e-4, "weight_decay": 0.01}},
  {**BASE, "name": "cnn_wd0.0",        "optimizer_kwargs": {"lr": 1e-3, "weight_decay": 0.0}},
  {**BASE, "name": "cnn_wd0.1",        "optimizer_kwargs": {"lr": 1e-3, "weight_decay": 0.1}},
  {**BASE, "name": "cnn_dropout0.3",   "dropout": 0.3},
  {**BASE, "name": "cnn_init_normal",  "init": "normal_0.01"},
  {**BASE, "name": "cnn_init_kaiming", "init": "kaiming_normal"},
  {**BASE, "name": "cnn_filters8",     "filters": 8},
  {**BASE, "name": "cnn_filters32",    "filters": 32},
  {**BASE, "name": "cnn_kernel5",      "kernel_size": 5},
  {**BASE, "name": "cnn_blocks1",      "num_conv_blocks": 1},
  {**BASE, "name": "cnn_blocks3",      "num_conv_blocks": 3},
]

results = {}
models = {}

for cfg in configs:
  torch.manual_seed(RANDOM_SEED)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  model = CNNModel(
    in_channels=IN_CHANNELS,
    image_size=IMAGE_SIZE,
    output_features=NUM_CLASSES,
    num_conv_blocks=cfg["num_conv_blocks"],
    filters=cfg["filters"],
    kernel_size=cfg["kernel_size"],
    hidden_units=cfg["hidden_units"],
    activation_fn=cfg["activation_fn"],
    dropout=cfg["dropout"]
  ).to(device)
  apply_init_cnn(model, cfg["init"])

  optimizer = cfg["optimizer"](params=model.parameters(), **cfg["optimizer_kwargs"])

  print(f"\nSELECTED CONFIG: {cfg['name']}")
  results[cfg["name"]] = train_and_validate(model=model, optimizer=optimizer, epochs=30)
  models[cfg["name"]] = model

  h = results[cfg["name"]]
  print(f"{cfg['name']:<28}train acc {h['train_acc'][-1]:.2f}%   val acc {h['val_acc'][-1]:.2f}%")

with open("results_cnn.json", "w") as f:
  json.dump(results, f)

ranking = sorted(results.items(), key=lambda item: item[1]["val_acc"][-1], reverse=True)

# In[39]:


print("\nLeaderboard (Val Acc%)")
for name, history in ranking:
  print(f"{name:<28}{history['val_acc'][-1]:.2f}%")

# In[40]:


print("\nFinalists on Test Set")
test_results = {}
for name, history in ranking[:3]:
  test_loss, test_acc = test(models[name])
  test_results[name] = {"val_acc": history["val_acc"][-1], "test_acc": test_acc, "test_loss": test_loss}
  print(f"{name:<24}val acc {history['val_acc'][-1]:.2f}%   test acc {test_acc:.2f}%   test loss {test_loss:.5f}")
with open("test_cnn.json", "w") as f:
  json.dump(test_results, f)

# # Applied to CIFAR10

# In[41]:


from torchvision.transforms import v2

# ## Global Variables

# In[42]:


EPOCHS = 30

IN_CHANNELS  = 3
IMAGE_SIZE   = 32
NUM_FEATURES = IN_CHANNELS * IMAGE_SIZE * IMAGE_SIZE   # 3072, for the MLP

# ## Configurations

# In[43]:


CONFIGS = [
  {
    "name": "hidden256", "network": "mlp",
    "activation_fn": nn.GELU, "optimizer": torch.optim.AdamW,
    "optimizer_kwargs": {"lr": 1e-3, "weight_decay": 0.01},
    "hidden_units": 256, "num_hidden_layers": 2, "dropout": 0.0, "init": "default",
  },
  {
    "name": "cnn_dropout0.3", "network": "cnn",
    "activation_fn": nn.GELU, "optimizer": torch.optim.AdamW,
    "optimizer_kwargs": {"lr": 1e-3, "weight_decay": 0.01},
    "num_conv_blocks": 2, "filters": 16, "kernel_size": 3, "hidden_units": 64, "dropout": 0.3, "init": "default",
  },
  {
    "name": "cnn_leaky_relu_adam", "network": "cnn",
    "activation_fn": nn.LeakyReLU, "optimizer": torch.optim.Adam,
    "optimizer_kwargs": {"lr": 1e-3},
    "num_conv_blocks": 2, "filters": 16, "kernel_size": 3, "hidden_units": 64, "dropout": 0.0, "init": "default",
  },
]

# ## Run Script

# In[44]:


transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

full_train = CIFAR10(root="data", train=True,  download=True, transform=transform)
test_data  = CIFAR10(root="data", train=False, download=True, transform=transform)

# Same 90/10 split as Fashion-MNIST: 45,000 train / 5,000 validation
train_data, val_data = random_split(
  full_train,
  [0.9, 0.1],
  generator=torch.Generator().manual_seed(RANDOM_SEED)
)
print(f"Train: {len(train_data)}  Validation: {len(val_data)}  Test: {len(test_data)}")

train_data, val_data, test_data = train_data, val_data, test_data
train_dataloader = move_to_gpu(train_data, device, shuffle=True)
val_dataloader   = move_to_gpu(val_data, device, shuffle=False)
test_dataloader  = move_to_gpu(test_data, device, shuffle=False)

results = {}
models = {}

for cfg in CONFIGS:
  torch.manual_seed(RANDOM_SEED)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  if cfg["network"] == "mlp":
    model = FashionModel(
      input_features=NUM_FEATURES,
      hidden_units=cfg["hidden_units"],
      output_features=NUM_CLASSES,
      activation_fn=cfg["activation_fn"],
      num_hidden_layers=cfg["num_hidden_layers"],
      dropout=cfg["dropout"]
    ).to(device)
    apply_init(model, cfg["init"])
  else:
    model = CNNModel(
      in_channels=IN_CHANNELS,
      image_size=IMAGE_SIZE,
      output_features=NUM_CLASSES,
      num_conv_blocks=cfg["num_conv_blocks"],
      filters=cfg["filters"],
      kernel_size=cfg["kernel_size"],
      hidden_units=cfg["hidden_units"],
      activation_fn=cfg["activation_fn"],
      dropout=cfg["dropout"]
    ).to(device)
    apply_init_cnn(model, cfg["init"])

  optimizer = cfg["optimizer"](params=model.parameters(), **cfg["optimizer_kwargs"])

  print(f"\nSELECTED CONFIG: {cfg['name']}")
  results[cfg["name"]] = train_and_validate(model=model, optimizer=optimizer, epochs=EPOCHS)
  models[cfg["name"]] = model

  h = results[cfg["name"]]
  print(f"{cfg['name']:<28}train acc {h['train_acc'][-1]:.2f}%   val acc {h['val_acc'][-1]:.2f}%")

with open("results_cifar.json", "w") as f:
  json.dump(results, f)

# In[45]:


print("\nCIFAR-10 test set")
test_results = {}
for name, history in results.items():
  test_loss, test_acc = test(models[name])
  test_results[name] = {"val_acc": history["val_acc"][-1], "test_acc": test_acc, "test_loss": test_loss}
  print(f"{name:<28}val acc {history['val_acc'][-1]:.2f}%   test acc {test_acc:.2f}%   test loss {test_loss:.5f}")

with open("test_cifar.json", "w") as f:
  json.dump(test_results, f)

# # Plotting Script

# ## Imports

# In[46]:


import os

# ## Global Variables

# In[47]:


OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ## Helper Functions

# In[48]:


def load(path):
  if os.path.exists(path):
    with open(path) as f:
      return json.load(f)
  return None


def save(fig, fname):
  path = os.path.join(OUT_DIR, fname)
  fig.savefig(path, dpi=150, bbox_inches="tight")
  plt.close(fig)
  plt.show(fig)
  print(f"saved {path}")


def val_acc(res, name):
  return res[name]["val_acc"][-1]

def print_tables(nets):
  for label, res in nets.items():
    print(f"\n{label}: top 5 by final validation accuracy")
    for n in sorted(res, key=lambda n: val_acc(res, n), reverse=True)[:5]:
      print(f"  {n:<26}train {res[n]['train_acc'][-1]:.2f}%   val {val_acc(res, n):.2f}%")

  for label, path in [("MLP", "test_mlp.json"), ("CNN", "test_cnn.json"), ("CIFAR-10", "test_cifar.json")]:
    test = load(path)
    if test:
      print(f"\n{label}: finalists on the test set")
      for n, r in test.items():
        print(f"  {n:<26}val {r['val_acc']:.2f}%   test {r['test_acc']:.2f}%   loss {r['test_loss']:.4f}")

# ## Learning Curves

# In[49]:


def plot_curves(nets):
  fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(13, 5))
  for k, (label, res) in enumerate(nets.items()):
    best = max(res, key=lambda n: val_acc(res, n))
    h = res[best]
    epochs = range(1, len(h["val_acc"]) + 1)
    ax_loss.plot(epochs, h["val_loss"], color=f"C{k}", label=f"{label} val ({best})")
    ax_loss.plot(epochs, h["train_loss"], color=f"C{k}", linestyle="--", alpha=0.6, label=f"{label} train")
    ax_acc.plot(epochs, h["val_acc"], color=f"C{k}", label=f"{label} val ({best})")
    ax_acc.plot(epochs, h["train_acc"], color=f"C{k}", linestyle="--", alpha=0.6, label=f"{label} train")
  ax_loss.set_title("Loss per epoch (best config per network)")
  ax_acc.set_title("Accuracy (%) per epoch (best config per network)")
  for ax in (ax_loss, ax_acc):
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
  save(fig, "curves.png")

# ## Final Validation Accuracy

# In[50]:


def plot_configs(nets):
  fig, axes = plt.subplots(1, len(nets), figsize=(7 * len(nets), 7), squeeze=False, gridspec_kw={"wspace": 0.6})
  for ax, (label, res) in zip(axes[0], nets.items()):
    names = sorted(res, key=lambda n: val_acc(res, n))
    accs = [val_acc(res, n) for n in names]
    ax.barh(names, accs)
    ax.set_xlim(min(accs) - 1, max(accs) + 0.5)
    ax.set_xlabel("final validation accuracy (%)")
    ax.set_title(label)
    ax.grid(alpha=0.3, axis="x")
  save(fig, "configs.png")

# ## Carry-over / Transfer Experiment

# In[ ]:


def plot_transfer(fashion_test, cifar_test):
  names = [n for n in cifar_test if n in fashion_test]
  missing = [n for n in cifar_test if n not in fashion_test]
  if missing:
    print(f"Warning: no Fashion-MNIST test result for {missing} (not a finalist); left out of transfer.png")
  if not names:
    print("No shared configs between CIFAR-10 and Fashion-MNIST test results; skipping transfer.png")
    return

  fig, ax = plt.subplots(figsize=(1.8 * len(names) + 3, 5))
  xs = list(range(len(names)))
  ax.bar([x - 0.2 for x in xs], [fashion_test[n]["test_acc"] for n in names], width=0.4, label="Fashion-MNIST")
  ax.bar([x + 0.2 for x in xs], [cifar_test[n]["test_acc"] for n in names], width=0.4, label="CIFAR-10")
  ax.set_xticks(xs)
  ax.set_xticklabels(names, rotation=20, ha="right")
  ax.set_ylabel("Test Accuracy (%)")
  ax.set_title("Transfer of the top configurations: Fashion-MNIST vs CIFAR-10")
  ax.legend()
  ax.grid(alpha=0.3, axis="y")
  save(fig, "transfer.png")

# ## Main Script

# In[59]:


mlp, cnn, cifar = load("results_mlp.json"), load("results_cnn.json"), load("results_cifar.json")
test_mlp, test_cnn, test_cifar = load("test_mlp.json"), load("test_cnn.json"), load("test_cifar.json")
fashion_test = {**(test_mlp or {}), **(test_cnn or {})}
nets = {label: res for label, res in [("MLP", mlp), ("CNN", cnn)] if res}
if not nets:
	raise SystemExit("No results_mlp.json / results_cnn.json found. Save them from the experiment scripts first.")

plot_curves(nets)
plot_configs(nets)

if test_cifar and fashion_test:
  plot_transfer(fashion_test, test_cifar)
else:
  print("test_*.json not found; skipping transfer.png")
print_tables(nets)
