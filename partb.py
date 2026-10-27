import torch
from torch import nn

from assignment import device, NUM_CLASSES, RANDOM_SEED, train_and_validate, test

IN_CHANNELS = 1
IMAGE_SIZE  = 28


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


if __name__ == "__main__":
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

  ranking = sorted(results.items(), key=lambda item: item[1]["val_acc"][-1], reverse=True)

  print("\nLeaderboard (Val Acc%)")
  for name, history in ranking:
    print(f"{name:<28}{history['val_acc'][-1]:.2f}%")

  # Keep this commented out until the CNN config list is final, then uncomment for ONE last run.
  # print("\nFinalists on Test Set")
  # for name, history in ranking[:3]:
  #   test_loss, test_acc = test(models[name])
  #   print(f"{name:<28}val acc {history['val_acc'][-1]:.2f}%   test acc {test_acc:.2f}%   test loss {test_loss:.5f}")