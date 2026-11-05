#!/usr/bin/env python
# coding: utf-8

# classification problem

# In[13]:


from typing import Any

#imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
import copy
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from jinja2 import optimizer
from sklearn.model_selection import train_test_split

# In[23]:


# function for time error

def time_error_minutes(pred_h, pred_m, true_h, true_m):
    pred_t = pred_h * 60 + pred_m
    true_t = true_h * 60 + true_m
    diff = np.abs(pred_t - true_t)
    return np.minimum(diff, 720 - diff)

# In[4]:


# data loading

# EDIT THIS TO PROJECT DIRECTORY  (sorry)
DATA_ROOT = Path("/home/s5235766/idl2026/assignment1")

CKPT = DATA_ROOT / "checkpoints"
CKPT.mkdir(parents=True, exist_ok=True)
# prototype data
images = np.load(DATA_ROOT / "data/A1_data_75/images.npy")
labels = np.load(DATA_ROOT / "data/A1_data_75/labels.npy")

# report training data
f_images=  np.load(DATA_ROOT / "data/A1_data_150/images.npy")
f_labels = np.load(DATA_ROOT / "data/A1_data_150/labels.npy")

device = "cuda" if torch.cuda.is_available() else "cpu"

print(images.shape,images.dtype)
print(labels.shape,labels.dtype)


# In[5]:


#split

idx = np.arange(len(images))
tr, tmp = train_test_split(idx, test_size=0.2, random_state=42)
va, te = train_test_split(tmp, test_size=0.5, random_state=42)

# In[6]:


bin30 = (labels[:,0] * 2 + labels[:,1] // 30).astype(np.int64)
print(bin30.shape, bin30.min(), bin30.max(), len(np.unique(bin30)))

# In[7]:


# load dataset as tensors

X_all = torch.from_numpy(images).float().unsqueeze(1) /255.0
y_all = torch.from_numpy(bin30).long()

tr_t = torch.from_numpy(tr).long()
va_t = torch.from_numpy(va).long()
te_t = torch.from_numpy(te).long()

X_tr = X_all[tr_t]
y_tr = y_all[tr_t]
X_va = X_all[va]
y_va = y_all[va]
X_te = X_all[te]
y_te = y_all[te]
print(X_tr.shape, y_tr.shape, X_va.shape, y_va.shape, X_te.shape, y_te.shape)

# In[58]:


# basic CNN architecture
n_classes =24

class FirstNet(nn.Module):
    def __init__(self, n_classes=24):
        super(FirstNet, self).__init__()
        self.conv1 = nn.Conv2d(1,16,3, padding=1)
        self.conv2 = nn.Conv2d(16,32,3, padding=1)
        self.conv3 = nn.Conv2d(32,64,3, padding=1)
        self.pool = nn.MaxPool2d(2,)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64,n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.gap(x)
        x = x.flatten(1)
        return self.fc(x)


# In[8]:


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

class Net(nn.Module):
    def __init__(self, n_classes=24):
        super().__init__()
        self.features = nn.Sequential(
            Block(1, 32),
            Block(32, 64),
            Block(64, 128),
            Block(128, 256),
        )
        self.gap= nn.AdaptiveAvgPool2d(1)
        self.drop =nn.Dropout(p=0.5)
        self.fc = nn.Linear(256,n_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = x.flatten(1)
        x = self.drop(x)
        return self.fc(x)



# In[61]:


batch_size = 64
NUM_EPOCHS = 25


model = Net(n_classes=24).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

best_val_acc = 0.0
best_state = None
best_epoch = 0
# training loop

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss =0.0
    perm = torch.randperm(len(X_tr))
    for start in range(0, len(X_tr), batch_size):
        bidx = perm[start:start+batch_size]                      # ← renamed
        inputs, labels = X_tr[bidx].to(device), y_tr[bidx].to(device)

        optimizer.zero_grad()
        outputs= model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (start // batch_size) % 50 == 49:
            print(f"epoch {epoch:02d} batch {start//batch_size+1:4d} loss {running_loss/50:.3f}")
            running_loss = 0.0
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for start in range(0, len(X_va), batch_size):
            inputs = X_va[start:start+batch_size].to(device)
            labels = y_va[start:start+batch_size]
            pred = model(inputs).argmax(dim=1)
            correct += (pred.cpu() == labels).sum().item()
            total += len(labels)
    val_acc = correct / total

    marker =""
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = epoch
        marker= "   -best model"

    print(f"epoch {epoch:02d}  lr {current_lr:.2e}  val_acc {val_acc:.4f}{marker}")

model.load_state_dict(best_state)
print(f"\nBest val_acc: {best_val_acc:.4f}  (epoch {best_epoch})")
torch.save(best_state, CKPT / "cls24_best.pt")
print('Finished Training')


# now 720

# In[9]:


cls720 = (labels[:,0] * 60 + labels[:,1]).astype(np.int64)
y_all720 = torch.from_numpy(cls720).long()
y_tr720 = y_all720[tr_t]
y_va720 = y_all720[va_t]
y_te720 = y_all720[te_t]

print(y_tr720.shape, y_tr720.min().item(), y_tr720.max().item())   # (14400,) 0 719

# In[74]:


batch_size = 64
NUM_EPOCHS = 100


model720 = Net(n_classes=720).to(device)
optimizer = torch.optim.AdamW(model720.parameters(), lr=1e-3, weight_decay=1e-3)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

best_val_acc = 0.0
best_state = None
best_epoch = 0
# training loop

for epoch in range(NUM_EPOCHS):
    model720.train()
    running_loss =0.0
    perm = torch.randperm(len(X_tr))
    for start in range(0, len(X_tr), batch_size):
        bidx = perm[start:start+batch_size]
        inputs= X_tr[bidx].to(device)
        lb = y_tr720[bidx].to(device)

        optimizer.zero_grad()
        outputs= model720(inputs)
        loss = criterion(outputs, lb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (start // batch_size) % 50 == 49:
            print(f"epoch {epoch:02d} batch {start//batch_size+1:4d} loss {running_loss/50:.3f}")
            running_loss = 0.0
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]

    model720.eval()
    correct = 0
    total = 0
    with (torch.no_grad()):
        for start in range(0, len(X_va), batch_size):
            inputs = X_va[start:start+batch_size].to(device)
            lb = y_va720[start:start+batch_size]
            pred = model720(inputs).argmax(dim=1)
            correct += (pred.cpu() == lb).sum().item()
            total += len(lb)
    val_acc = correct / total

    marker =""
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = copy.deepcopy(model720.state_dict())
        best_epoch = epoch
        marker= "   -best 720 model"

    print(f"epoch {epoch:02d}  lr {current_lr:.2e}  val_acc {val_acc:.4f}{marker}")

model720.load_state_dict(best_state)
print(f"\nBest val_acc: {best_val_acc:.4f}  (epoch {best_epoch})")
torch.save(best_state, CKPT / "cls720_best.pt")
print('Finished Training')


# ## 150x150 on 24 and 720 classes

# In[10]:


# 150x150 tensors, SAME split as 75x75 (fair comparison)
X_all150 = torch.from_numpy(f_images).float().unsqueeze(1) / 255.0    # (18000, 1, 150, 150)

tr_t = torch.from_numpy(tr).long()
va_t = torch.from_numpy(va).long()
te_t = torch.from_numpy(te).long()

X_tr150 = X_all150[tr_t]
X_va150 = X_all150[va_t]
X_te150 = X_all150[te_t]

print("X_tr150:", X_tr150.shape, " X_va150:", X_va150.shape, " X_te150:", X_te150.shape)

# # 150x150 n_classes= 24 training loop

# In[11]:


batch_size = 64
NUM_EPOCHS = 60


model = Net(n_classes=24).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

#for plots
train_losses = []
val_losses   = []
val_accs     = []
lrs          = []

best_val_acc = 0.0
best_state = None
best_epoch = 0
# training loop

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss =0.0
    epoch_loss= 0.0
    n_batches = 0
    perm = torch.randperm(len(X_tr150))
    for start in range(0, len(X_tr150), batch_size):
        bidx = perm[start:start+batch_size]                      # ← renamed
        inputs, lb = X_tr150[bidx].to(device), y_tr[bidx].to(device)

        optimizer.zero_grad()
        outputs= model(inputs)
        loss = criterion(outputs, lb)
        loss.backward()
        optimizer.step()
        loss_val = loss.item()
        running_loss += loss_val
        epoch_loss += loss_val
        n_batches += 1
        if (start // batch_size) % 50 == 49:
            print(f"epoch {epoch:02d} batch {start//batch_size+1:4d} loss {running_loss/50:.3f}")
            running_loss = 0.0
    train_losses.append(epoch_loss/n_batches)
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    lrs.append(current_lr)

    model.eval()
    correct = 0
    total = 0
    val_loss_sum =0.0
    with torch.no_grad():
        for start in range(0, len(X_va150), batch_size):
            inputs = X_va150[start:start+batch_size].to(device)
            lb = y_va[start:start+batch_size].to(device)
            out = model(inputs)
            val_loss_sum += criterion(out, lb).item() * len(lb)
            pred =out.argmax(dim=1)
            correct += (pred.cpu() == lb.cpu()).sum().item()
            total += len(lb)
    val_acc = correct / total
    val_loss = val_loss_sum / total
    val_accs.append(val_acc)
    val_losses.append(val_loss)
    marker =""
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = epoch
        marker= "   -best model"

    print(f"epoch {epoch:02d}  lr {current_lr:.2e} " f"train{train_losses[-1]:.4f}" f"val_loss {val_loss:.4f} " f" val_acc {val_acc:.4f}{marker}" )

model.load_state_dict(best_state)
print(f"\nBest val_acc: {best_val_acc:.4f}  (epoch {best_epoch})")
torch.save(best_state, CKPT / "cls24_150_best.pt")

torch.save({
    "train_losses": train_losses,
    "val_losses":   val_losses,
    "val_accs":     val_accs,
    "lrs":          lrs,
    "best_val_acc": best_val_acc,
    "best_epoch":   best_epoch,
}, CKPT / "history_cls24_150.pt")

print('Finished Training')


# 150x150 720 classes training+val loop

# In[14]:


batch_size = 64
NUM_EPOCHS = 200


model720 = Net(n_classes=720).to(device)
optimizer = torch.optim.AdamW(model720.parameters(), lr=1e-3, weight_decay=3e-3)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

#for plots
train_losses_720 = []
val_losses_720   = []
val_accs_720     = []
lrs_720          = []

best_val_acc = 0.0
best_state = None
best_epoch = 0
# training loop

for epoch in range(NUM_EPOCHS):
    model720.train()
    running_loss =0.0
    epoch_loss= 0.0
    n_batches = 0
    perm = torch.randperm(len(X_tr150))
    for start in range(0, len(X_tr150), batch_size):
        bidx = perm[start:start+batch_size]                      # ← renamed
        inputs= X_tr150[bidx].to(device)
        lb = y_tr720[bidx].to(device)

        optimizer.zero_grad()
        outputs= model720(inputs)
        loss = criterion(outputs, lb)
        loss.backward()
        optimizer.step()
        loss_val = loss.item()
        running_loss += loss_val
        epoch_loss += loss_val
        n_batches += 1
        if (start // batch_size) % 50 == 49:
            print(f"epoch {epoch:02d} batch {start//batch_size+1:4d} loss {running_loss/50:.3f}")
            running_loss = 0.0
    train_losses_720.append(epoch_loss/n_batches)
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    lrs_720.append(current_lr)

    model720.eval()
    correct = 0
    total = 0
    val_loss_sum =0.0
    with (torch.no_grad()):
        for start in range(0, len(X_va150), batch_size):
            inputs = X_va150[start:start+batch_size].to(device)
            lb = y_va720[start:start+batch_size].to(device)
            out = model720(inputs)
            val_loss_sum += criterion(out, lb).item() * len(lb)
            pred =out.argmax(dim=1)
            correct += (pred.cpu() == lb.cpu()).sum().item()
            total += len(lb)
    val_acc = correct / total
    val_loss = val_loss_sum / total
    val_accs_720.append(val_acc)
    val_losses_720.append(val_loss)
    marker =""
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = copy.deepcopy(model720.state_dict())
        best_epoch = epoch
        marker= "   -best 720 model"

    print(f"epoch {epoch:02d}  lr {current_lr:.2e} " f"train{train_losses_720[-1]:.4f}" f"val_loss {val_loss:.4f} " f" val_acc {val_acc:.4f}{marker}" )

model720.load_state_dict(best_state)
print(f"\nBest val_acc: {best_val_acc:.4f}  (epoch {best_epoch})")
torch.save(best_state, CKPT / "cls720_150_best.pt")

torch.save({
    "train_losses": train_losses_720,
    "val_losses":   val_losses_720,
    "val_accs":     val_accs_720,
    "lrs":          lrs_720,
    "best_val_acc": best_val_acc,
    "best_epoch":   best_epoch,
}, CKPT / "history_cls720_150.pt")

print('Finished Training')

# In[17]:


# load
h24  = torch.load(CKPT / "history_cls24_150.pt",  map_location="cpu")
h720 = torch.load(CKPT / "history_cls720_150.pt", map_location="cpu")

# loss curves
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# 24-class
axes[0].plot(h24["train_losses"], label="train")
axes[0].plot(h24["val_losses"],   label="val")
axes[0].axvline(h24["best_epoch"], color="red", ls="--", alpha=0.4, label=f"best epoch {h24['best_epoch']}")
axes[0].set_xlabel("epoch"); axes[0].set_ylabel("cross-entropy loss")
axes[0].set_title(f"24-class @ 150 (best val_acc = {h24['best_val_acc']:.4f})")
axes[0].legend(); axes[0].grid(alpha=0.3)

# 720-class
axes[1].plot(h720["train_losses"], label="train")
axes[1].plot(h720["val_losses"],   label="val")
axes[1].axvline(h720["best_epoch"], color="red", ls="--", alpha=0.4, label=f"best epoch {h720['best_epoch']}")
axes[1].set_xlabel("epoch"); axes[1].set_ylabel("cross-entropy loss")
axes[1].set_title(f"720-class @ 150 (best val_acc = {h720['best_val_acc']:.4f})")
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(CKPT / "fig_loss_curves.png", dpi=150)
plt.show()

# fig 2 val acc
plt.figure(figsize=(10, 5))
plt.plot(h24["val_accs"],  label=f"24-class  (best {h24['best_val_acc']:.4f})",  color="tab:blue")
plt.plot(h720["val_accs"], label=f"720-class (best {h720['best_val_acc']:.4f})", color="tab:orange")
plt.axhline(h24["best_val_acc"],  color="tab:blue",   ls=":", alpha=0.4)
plt.axhline(h720["best_val_acc"], color="tab:orange", ls=":", alpha=0.4)
plt.xlabel("epoch")
plt.ylabel("validation exact-match accuracy")
plt.title("Validation accuracy @ 150×150")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(CKPT / "fig_val_acc_overlay.png", dpi=150)
plt.show()

# final eval

# In[11]:


# load
model24 = Net(n_classes=24).to(device)
model24.load_state_dict(torch.load(CKPT / "cls24_150_best.pt", map_location=device))
model24.eval()

correct = total = 0
with torch.no_grad():
    for start in range(0, len(X_te150), 128):
        inputs = X_te150[start:start+128].to(device)
        lb = y_te[start:start+128]
        pred = model24(inputs).argmax(dim=1)
        correct += (pred.cpu() == lb).sum().item()
        total += len(lb)

test_acc = correct / total
print(f"24-class @ 150 test accuracy: {test_acc:.4f}")

# In[12]:


#load
model720 = Net(n_classes=720).to(device)
model720.load_state_dict(torch.load(CKPT / "cls720_150_best.pt", map_location=device))
model720.eval()

correct = total = 0
with torch.no_grad():
    for start in range(0, len(X_te150), 128):
        inputs = X_te150[start:start+128].to(device)
        lb = y_te720[start:start+128]                       #
        pred = model720(inputs).argmax(dim=1)
        correct += (pred.cpu() == lb).sum().item()
        total += len(lb)

test_acc = correct / total
print(f"720-class @ 150 test accuracy: {test_acc:.4f}")

# In[24]:


import numpy as np
import torch

def time_error_minutes(pred_h, pred_m, true_h, true_m):
    pred_t = pred_h * 60 + pred_m
    true_t = true_h * 60 + true_m
    diff = np.abs(pred_t - true_t)
    return np.minimum(diff, 720 - diff)

true_h = labels[te][:,0]
true_m = labels[te][:,1]

# 24-class
m24 = Net(n_classes=24).to(device)
m24.load_state_dict(torch.load(CKPT / "cls24_150_best.pt", map_location=device))
m24.eval()
preds = []
with torch.no_grad():
    for s in range(0, len(X_te150), 128):
        preds.append(m24(X_te150[s:s+128].to(device)).argmax(1).cpu())
pred_cls_24 = torch.cat(preds).numpy()
pred_h_24 = pred_cls_24 // 2
pred_m_24 = (pred_cls_24 % 2) * 30 + 15
err_24 = time_error_minutes(pred_h_24, pred_m_24, true_h, true_m)

# 720-class
m720 = Net(n_classes=720).to(device)
m720.load_state_dict(torch.load(CKPT / "cls720_150_best.pt", map_location=device))
m720.eval()
preds = []
with torch.no_grad():
    for s in range(0, len(X_te150), 128):
        preds.append(m720(X_te150[s:s+128].to(device)).argmax(1).cpu())
pred_cls_720 = torch.cat(preds).numpy()
pred_h_720 = pred_cls_720 // 60
pred_m_720 = pred_cls_720 % 60
err_720 = time_error_minutes(pred_h_720, pred_m_720, true_h, true_m)

# summary
print(f"{'model':<12} {'MAE':>8} {'median':>8} {'max':>6} {'<=5min':>8} {'exact':>8}")
print("-" * 58)
for name, err in [("24-class", err_24), ("720-class", err_720)]:
    print(f"{name:<12} {err.mean():>8.2f} {np.median(err):>8.1f} {err.max():>6.0f} "
          f"{(err <= 5).mean()*100:>7.1f}% {(err == 0).mean():>8.4f}")

# # multi-head approach
# 

# In[14]:


# split labels into separate hour and minute tensors
hour_all   = labels[:,0].astype(np.int64)
minute_all = labels[:,1].astype(np.int64)

y_tr_h = torch.from_numpy(hour_all[tr]).long()
y_va_h = torch.from_numpy(hour_all[va]).long()
y_tr_m = torch.from_numpy(minute_all[tr]).long()
y_va_m = torch.from_numpy(minute_all[va]).long()
y_te_h = torch.from_numpy(hour_all[te]).long()
y_te_m = torch.from_numpy(minute_all[te]).long()

print(y_tr_h.shape, y_tr_h.min().item(), y_tr_h.max().item())
print(y_tr_m.shape, y_tr_m.min().item(), y_tr_m.max().item())

# In[16]:


# shared backbone plus two heads
class NetMultiHead(nn.Module):
    def __init__(self, n_hours=12, n_minutes=60):
        super().__init__()
        # same backbone as single head net
        self.features = nn.Sequential(
            Block(1, 32),
            Block(32, 64),
            Block(64, 128),
            Block(128, 256),
        )
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.5)

        self.head_hour   = nn.Linear(256, n_hours)
        self.head_minute = nn.Linear(256, n_minutes)

    def forward(self, x):
        x = self.features(x)       # shared features
        x = self.gap(x)
        x = x.flatten(1)
        x = self.drop(x)
        out_hour   = self.head_hour(x)
        out_minute = self.head_minute(x)
        return out_hour, out_minute

# In[17]:


mh_test = NetMultiHead().to(device)
xb_test = torch.randn(4, 1, 75, 75).to(device)
oh, om = mh_test(xb_test)
print(oh.shape, om.shape)

# In[20]:


# multi head training loop with history tracking
batch_size = 64
NUM_EPOCHS = 100
alpha      = 1.0

model_mh = NetMultiHead().to(device)
optimizer = torch.optim.AdamW(model_mh.parameters(), lr=1e-3, weight_decay=1e-3)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

# history buffers
train_losses_mh = []
val_losses_mh   = []
exact_accs_mh   = []
hour_accs_mh    = []
minute_accs_mh  = []
lrs_mh          = []

best_val_acc = 0.0
best_state = None
best_epoch = 0

for epoch in range(NUM_EPOCHS):
    # train
    model_mh.train()
    running_loss = 0.0
    epoch_loss = 0.0
    n_batches = 0
    perm = torch.randperm(len(X_tr150))
    for start in range(0, len(X_tr150), batch_size):
        bidx = perm[start:start+batch_size]
        inputs = X_tr150[bidx].to(device)
        lb_h   = y_tr_h[bidx].to(device)
        lb_m   = y_tr_m[bidx].to(device)

        optimizer.zero_grad()
        out_h, out_m = model_mh(inputs)
        loss = criterion(out_h, lb_h) + alpha * criterion(out_m, lb_m)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        running_loss += loss_val
        epoch_loss   += loss_val
        n_batches    += 1
        if (start // batch_size) % 50 == 49:
            print(f"epoch {epoch:02d} batch {start//batch_size+1:4d} loss {running_loss/50:.3f}")
            running_loss = 0.0

    train_losses_mh.append(epoch_loss / n_batches)

    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    lrs_mh.append(current_lr)

    # validate
    model_mh.eval()
    correct_exact = correct_h = correct_m = total = 0
    val_loss_sum = 0.0
    with torch.no_grad():
        for start in range(0, len(X_va150), batch_size):
            inputs = X_va150[start:start+batch_size].to(device)
            lb_h   = y_va_h[start:start+batch_size].to(device)
            lb_m   = y_va_m[start:start+batch_size].to(device)
            out_h, out_m = model_mh(inputs)
            val_loss_sum += (criterion(out_h, lb_h) + alpha * criterion(out_m, lb_m)).item() * len(lb_h)
            ph = out_h.argmax(dim=1).cpu()
            pm = out_m.argmax(dim=1).cpu()
            correct_h     += (ph == lb_h.cpu()).sum().item()
            correct_m     += (pm == lb_m.cpu()).sum().item()
            correct_exact += ((ph == lb_h.cpu()) & (pm == lb_m.cpu())).sum().item()
            total         += len(lb_h)

    val_exact = correct_exact / total
    val_loss  = val_loss_sum / total
    exact_accs_mh.append(val_exact)
    hour_accs_mh.append(correct_h / total)
    minute_accs_mh.append(correct_m / total)
    val_losses_mh.append(val_loss)

    marker = ""
    if val_exact > best_val_acc:
        best_val_acc = val_exact
        best_state = copy.deepcopy(model_mh.state_dict())
        best_epoch = epoch
        marker = "   -best multi head model"

    print(f"epoch {epoch:02d} lr {current_lr:.2e} "
          f"train{train_losses_mh[-1]:.4f}"
          f"val_loss {val_loss:.4f} "
          f"exact {val_exact:.4f} h {hour_accs_mh[-1]:.4f} m {minute_accs_mh[-1]:.4f}{marker}")

model_mh.load_state_dict(best_state)
print(f"\nbest val_acc: {best_val_acc:.4f}  (epoch {best_epoch})")
torch.save(best_state, CKPT / "multihead_150_best.pt")

torch.save({
    "train_losses": train_losses_mh,
    "val_losses":   val_losses_mh,
    "exact_accs":   exact_accs_mh,
    "hour_accs":    hour_accs_mh,
    "minute_accs":  minute_accs_mh,
    "lrs":          lrs_mh,
    "best_val_acc": best_val_acc,
    "best_epoch":   best_epoch,
}, CKPT / "history_multihead_150.pt")

print('finished training')

# In[28]:


import matplotlib.pyplot as plt
import torch

# load multi head history
hmh = torch.load(CKPT / "history_multihead_150.pt", map_location="cpu")

# loss curve
plt.figure(figsize=(10, 4))
plt.plot(hmh["train_losses"], label="train")
plt.plot(hmh["val_losses"],   label="val")
plt.axvline(hmh["best_epoch"], color="red", ls="--", alpha=0.4, label=f"best epoch {hmh['best_epoch']}")
plt.xlabel("epoch")
plt.ylabel("joint loss (CE_hour + CE_minute)")
plt.title(f"multi-head - loss (best val exact = {hmh['best_val_acc']:.4f})")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(CKPT / "fig_multihead_loss.png", dpi=150, facecolor='white')
plt.show()

# val acc
plt.figure(figsize=(10, 5))
plt.plot(hmh["exact_accs"],  label=f"exact-match (best {hmh['best_val_acc']:.4f})", color="tab:green")
plt.axvline(hmh["best_epoch"], color="red", ls=":", alpha=0.4)
plt.xlabel("epoch")
plt.ylabel("validation accuracy")
plt.title("multi-head - validation accuracy")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(CKPT / "fig_multihead_valacc.png", dpi=150, facecolor='white')
plt.show()


# In[22]:


model_mh = NetMultiHead().to(device)
model_mh.load_state_dict(torch.load(CKPT / "multihead_150_best.pt", map_location=device))
model_mh.eval()

correct_exact = correct_h = correct_m = total = 0
with torch.no_grad():
    for start in range(0, len(X_te), 128):
        inputs = X_te150[start:start+128].to(device)
        lb_h   = y_te_h[start:start+128]
        lb_m   = y_te_m[start:start+128]
        out_h, out_m = model_mh(inputs)
        ph = out_h.argmax(1).cpu()
        pm = out_m.argmax(1).cpu()
        correct_h     += (ph == lb_h).sum().item()
        correct_m     += (pm == lb_m).sum().item()
        correct_exact += ((ph == lb_h) & (pm == lb_m)).sum().item()
        total         += len(lb_h)

print(f"multi-head test exact-match  : {correct_exact/total:.4f}")
print(f"multi-head test hour-only    : {correct_h/total:.4f}")
print(f"multi-head test minute-only  : {correct_m/total:.4f}")

# In[25]:


# multi-head
mmh = NetMultiHead().to(device)
mmh.load_state_dict(torch.load(CKPT / "multihead_150_best.pt", map_location=device))
mmh.eval()
ph_list, pm_list = [], []
with torch.no_grad():
    for s in range(0, len(X_te150), 128):
        oh, om = mmh(X_te150[s:s+128].to(device))
        ph_list.append(oh.argmax(1).cpu())
        pm_list.append(om.argmax(1).cpu())
pred_h_mh = torch.cat(ph_list).numpy()
pred_m_mh = torch.cat(pm_list).numpy()
err_mh = time_error_minutes(pred_h_mh, pred_m_mh, true_h, true_m)

# summary
print(f"{'model':<12} {'MAE':>8} {'median':>8} {'max':>6} {'<=5min':>8} {'exact':>8}")
print("-" * 58)
print(f"{'multi-head':<12} {err_mh.mean():>8.2f} {np.median(err_mh):>8.1f} {err_mh.max():>6.0f} "
      f"{(err_mh <= 5).mean()*100:>7.1f}% {(err_mh == 0).mean():>8.4f}")
