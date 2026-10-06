import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, models, transforms

# ---- 1. Dataset dhundho (./dataset ke andar) --------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(HERE, "dataset")

DATA_DIR = None
for root, dirs, _ in os.walk(DATASET_ROOT):
    if "cardboard" in dirs and "plastic" in dirs:
        DATA_DIR = root
        break

if DATA_DIR is None:
    raise SystemExit("Class folders dataset folder mein nahi mile.")
print("Dataset:", DATA_DIR)

MODEL_DIR = os.path.join(HERE, "model")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "trash_model.pt")

# ---- 2. Data ----------------------------------------------------------
norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    norm,
])
val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    norm,
])

full_train = datasets.ImageFolder(DATA_DIR, transform=train_tf)
full_val = datasets.ImageFolder(DATA_DIR, transform=val_tf)
classes = full_train.classes
print("Classes:", classes, "| images:", len(full_train))

n_val = int(0.2 * len(full_train))
gen = torch.Generator().manual_seed(42)
train_idx, val_idx = random_split(
    range(len(full_train)), [len(full_train) - n_val, n_val], generator=gen
)
train_ds = Subset(full_train, list(train_idx))
val_ds = Subset(full_val, list(val_idx))

train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
val_dl = DataLoader(val_ds, batch_size=32, num_workers=0)

# ---- 3. Model ---------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(classes))
model = model.to(device)

loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
EPOCHS = 8

# ---- 4. Train ---------------------------------------------------------
best = 0.0
for epoch in range(EPOCHS):
    model.train()
    for x, y in train_dl:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in val_dl:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    acc = correct / total
    print(f"Epoch {epoch + 1}/{EPOCHS}  val accuracy: {acc:.3f}")
    if acc > best:
        best = acc
        torch.save({"model": model.state_dict(), "classes": classes}, MODEL_PATH)

print(f"Best val accuracy: {best:.3f}  (saved: {MODEL_PATH})")

# ---- 5. Bin mapping (predict.py mein kaam aayega) ---------------------
BIN_MAP = {
    "cardboard": "Dry / Recyclable",
    "paper": "Dry / Recyclable",
    "plastic": "Dry / Recyclable",
    "metal": "Dry / Recyclable",
    "glass": "Dry / Recyclable (glass)",
    "trash": "General / Reject",
}
print("Bin mapping:", BIN_MAP)