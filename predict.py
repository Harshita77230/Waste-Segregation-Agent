import os
import sys
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "model", "trash_model.pt")

BIN_MAP = {
    "cardboard": ("Dry / Recyclable (Blue bin)", "Flatten it and keep it dry."),
    "paper": ("Dry / Recyclable (Blue bin)", "Keep it clean and dry. Greasy paper goes to wet waste."),
    "plastic": ("Dry / Recyclable (Blue bin)", "Rinse it and remove the cap if possible."),
    "metal": ("Dry / Recyclable (Blue bin)", "Rinse cans and crush them to save space."),
    "glass": ("Dry / Recyclable - Glass (Blue bin)", "Handle carefully. Wrap broken glass in paper first."),
    "trash": ("General / Reject (Black bin)", "Not recyclable. Dispose of it with general waste."),
}

CONF_THRESHOLD = 0.60

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_model():
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    classes = ckpt["classes"]
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(classes))
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, classes


def predict(image_path):
    model, classes = load_model()
    img = Image.open(image_path).convert("RGB")
    x = tf(img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    top = torch.topk(probs, 3)
    results = [(classes[i], p.item()) for p, i in zip(top.values, top.indices)]
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Use: python predict.py <image_path>")

    path = sys.argv[1]
    results = predict(path)
    label, conf = results[0]

    print(f"\nImage: {path}")
    print(f"Detected: {label}  (confidence {conf:.0%})")

    if conf < CONF_THRESHOLD:
        print("Not sure. Try another angle or better light.")
        print("Other guesses:", ", ".join(f"{l} ({p:.0%})" for l, p in results[1:]))
    else:
        bin_name, tip = BIN_MAP[label]
        print(f"Bin: {bin_name}")
        print(f"Tip: {tip}")
        print("Also possible:", ", ".join(f"{l} ({p:.0%})" for l, p in results[1:]))