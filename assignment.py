import torch
from torch import nn
from torchvision.datasets import FashionMNIST, CIFAR10
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split

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

# Hyperparameters
NUM_CLASSES   = 10      # 10
NUM_FEATURES  = 28*28   # 28x28 image
RANDOM_SEED   = 42
LEARNING_RATE = 0.1
BATCH_SIZE    = 64

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

def move_to_gpu(data, device, shuffle=False):
  X, y = next(iter(DataLoader(data,batch_size=len(data))))
  return DataLoader(
    TensorDataset(X.to(device), y.to(device)),
    batch_size=BATCH_SIZE,
    shuffle=shuffle
  )

train_dataloader  = move_to_gpu(train_data, device, shuffle=True)
test_dataloader   = move_to_gpu(test_data, device, shuffle=False)
val_dataloader    = move_to_gpu(val_data, device, shuffle=False)

# print(len(training_data), len(test_data)) confirming we should have 60,000 train and 10,000 test
print(f"Train: {len(train_data)}  Validation: {len(val_data)}  Test: {len(test_data)}")  # 54000 / 6000 / 10000

# debugging
# train_features, train_labels = next(iter(train_dataloader))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")
# print(train_data[0][0].shape, train_data[0][1], train_data.classes)

torch.manual_seed(RANDOM_SEED)

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

loss_fn = nn.CrossEntropyLoss()

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

def test(model: nn.Module):
  # Final test evaluation: run once, only for your final chosen configuration
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

if __name__ == "__main__":
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

  # Baseline = best run of the grid (gelu + AdamW). weight_decay=0.01 is AdamW's default, made explicit.
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

  ranking = sorted(results.items(), key=lambda item: item[1]["val_acc"][-1], reverse=True)

  print("\nLeaderboard (Val Acc%)")
  for name, history in ranking:
    print(f"{name:<24}{history['val_acc'][-1]:.2f}%")

  # uncomment below lines when configs are finalised
  # print("\nFinalists on Test Set")
  # for name, history in ranking[:3]:
  #   test_loss, test_acc = test(models[name])
  #   print(f"{name:<24}val acc {history['val_acc'][-1]:.2f}%   test acc {test_acc:.2f}%   test loss {test_loss:.5f}")