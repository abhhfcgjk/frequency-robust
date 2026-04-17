import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from ufbrp.datasets import get_data_loader
from ufbrp.models import DBFTT, create_model

DATASET = "CIFAR100"
MODEL_NAME = "DBFTT"
num_classes = 100

NORMALIZATION = {
    "CIFAR10": {"mean": (0.4914, 0.4822, 0.4465), "std": (0.2023, 0.1994, 0.2010)},
    "CIFAR100": {"mean": (0.5071, 0.4867, 0.4408), "std": (0.2675, 0.2565, 0.2761)},
    "ImageNet": {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
}
device = "cuda:0"

path = f"ufbrp/logs/{MODEL_NAME.lower()}-cifar100/best_model.pt"

model = create_model(
    DBFTT(wavelet_level=4, num_classes=num_classes),
    mean=NORMALIZATION[DATASET]["mean"],
    std=NORMALIZATION[DATASET]["std"],
).to(device)
# model = create_model(ResNet50(num_classes=num_classes),
#                      mean=NORMALIZATION[DATASET]["mean"],
#                      std=NORMALIZATION[DATASET]["std"]).to(device)
# model = create_model(LipReg(wavelet_level=4, num_classes=num_classes),
#                      mean=NORMALIZATION[DATASET]["mean"],
#                      std=NORMALIZATION[DATASET]["std"]).to(device)

ckpt = torch.load(path, weights_only=True)
model.load_state_dict(ckpt["model"])
model.eval()
_, _, dataloader = get_data_loader(DATASET, f"data/{DATASET}", 2, 128, val_split_ratio=0.9)

margins = []

with torch.no_grad():
    for data, label in tqdm(dataloader, total=len(dataloader)):
        data = data.to(device)
        logits = model(data)
        probs = torch.softmax(logits, dim=1)

        # Sort probabilities in descending order
        sorted_probs, _ = torch.sort(probs, dim=1, descending=True)

        # Margin = top1 - top2
        batch_margins = sorted_probs[:, 0] - sorted_probs[:, 1]

        # Store results
        margins.append(batch_margins.cpu())

# Concatenate all batch results
margins = torch.cat(margins)

print(f"Mean margin: {margins.mean().item():.4f}")
print(f"Std of margins: {margins.std().item():.4f}")

# Optionally visualize the margin distribution
plt.hist(margins.numpy(), bins=50, alpha=0.7)
plt.xlabel("Margin (Top1 - Top2)")
plt.ylabel("Frequency")
plt.ylim(top=6000)
plt.title(f"{MODEL_NAME} {DATASET}")
plt.savefig(f"margins_{MODEL_NAME.lower()}-{DATASET.lower()}.png")
