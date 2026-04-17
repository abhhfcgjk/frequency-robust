import numpy as np
import scipy as sp
import torch
import torchvision.transforms.functional as TF
from matplotlib import pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from tqdm import tqdm

from ufbrp.datasets import get_data_loader
from ufbrp.models import DBFTT, create_model


def grad_to_pca_image(grad: torch.Tensor, save_path, title: str):
    """
    Convert 3-channel gradient to grayscale via PCA, normalize, and save with colorbar.
    """
    grad_np = grad.detach().cpu().numpy()  # (3,H,W)
    C, H, W = grad_np.shape
    grad_flat = grad_np.reshape(C, -1).T  # (H*W, C)

    # PCA reduce to 1D
    pca = PCA(n_components=1)
    pca_grad = pca.fit_transform(grad_flat).reshape(H, W)

    # Normalize to [0,1]
    # pca_grad -= pca_grad.min()
    # pca_grad /= (pca_grad.max() + 1e-8)

    # Plot with colorbar
    plt.figure(figsize=(3, 3))
    im = plt.imshow(pca_grad, cmap="viridis")
    plt.axis("off")
    plt.title(title, fontsize=8)
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Gradient intensity")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def tensor_to_image(t: torch.Tensor) -> Image.Image:
    """
    Convert tensor [C,H,W] -> PIL.Image after normalization 0–255
    """
    t = t.detach().cpu()
    t = t - t.min()
    t = t / (t.max() + 1e-8)
    t = (t * 255).clamp(0, 255).byte()
    return TF.to_pil_image(t)


def grad_to_image(grad: torch.Tensor) -> Image.Image:
    """
    Normalize gradient tensor [B, C, H, W] to 0-255 uint8 and return PIL image.
    Operates on a single sample (C,H,W).
    """
    grad = grad.detach().cpu()
    grad = grad - grad.min()
    grad = grad / (grad.max() + 1e-8)
    grad = (grad * 255).clamp(0, 255).byte()
    return TF.to_pil_image(grad)


import torch


def grad_to_pca_image_pair(g_high: torch.Tensor, g_low: torch.Tensor, path: str, title="Gradient comparison"):
    """
    Visualize and save high and low branch Jacobians on the same figure with shared normalization and colorbar.
    """

    # Ensure tensors are detached and moved to CPU
    g_high = g_high.detach().cpu().numpy()
    g_low = g_low.detach().cpu().numpy()

    # Both expected shape [C, H, W]
    C, H, W = g_high.shape

    # Flatten spatial dimensions for PCA
    X_high = g_high.reshape(C, -1).T
    X_low = g_low.reshape(C, -1).T

    # Fit PCA jointly across both gradients
    X_all = np.concatenate([X_high, X_low], axis=0)
    pca = PCA(n_components=1)
    pca.fit(X_all)

    img_high = pca.transform(X_high).reshape(H, W)
    img_low = pca.transform(X_low).reshape(H, W)

    # Shared normalization
    vmin = min(img_high.min(), img_low.min())
    vmax = max(img_high.max(), img_low.max())

    # Plot side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axes[0].imshow(img_high, cmap="viridis")
    axes[0].set_title("High branch")
    axes[0].axis("off")

    im1 = axes[1].imshow(img_low, cmap="viridis")
    axes[1].set_title("Low branch")
    axes[1].axis("off")

    print(
        f"[Low mean & hmean & median]: {img_low.mean()} & {sp.stats.hmean(np.abs(img_low).flatten())} & {sp.ndimage.median(img_low)}; [High mean & hmean & median]: {img_high.mean()} & {sp.stats.hmean(np.abs(img_high).flatten())} & {sp.ndimage.median(img_high)}"
    )
    # Shared colorbar (single bar for both images)
    cbar = fig.colorbar(im1, ax=axes, orientation="vertical", fraction=0.035, pad=0.04)
    # cbar.set_label("PCA projection value", rotation=270, labelpad=15)

    fig.suptitle(title)
    # plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison figure to {path}")


NORMALIZATION = {
    "CIFAR10": {"mean": (0.4914, 0.4822, 0.4465), "std": (0.2023, 0.1994, 0.2010)},
    "CIFAR100": {"mean": (0.5071, 0.4867, 0.4408), "std": (0.2675, 0.2565, 0.2761)},
    "ImageNet": {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
}
device = "cuda:0"

path = "ufbrp/logs/dbftt-coif/best_model.pt"
num_classes = 10
model = create_model(
    DBFTT(wavelet_level=2, num_classes=num_classes),
    mean=NORMALIZATION["CIFAR10"]["mean"],
    std=NORMALIZATION["CIFAR10"]["std"],
).to(device)

ckpt = torch.load(path, weights_only=True)
model.load_state_dict(ckpt["model"])

criterion = torch.nn.CrossEntropyLoss()

_, _, dataloader = get_data_loader("CIFAR10", "data/CIFAR10", 2, 128, val_split_ratio=0.9)

grads = torch.tensor([], device=device)
for data, label in tqdm(dataloader, total=len(dataloader)):
    data, label = data.to(device), label.to(device)
    data.requires_grad_(True)
    output = model(data)
    loss = criterion(output, label)
    grad = torch.autograd.grad(loss, [data])[0].detach()
    grads = torch.concat([grads, grad], dim=0)
    # print(grad.shape)

# g = grad_to_image(grads.mean(dim=0))
# g.save("grad.png")
grad_to_pca_image(grads.mean(dim=0), "main_grad.png", title="Jacobian")


NORMALIZATION = {
    "CIFAR10": {"mean": (0.4914, 0.4822, 0.4465), "std": (0.2023, 0.1994, 0.2010)},
}
device = "cuda:0"

# --- Load model ---
path = "ufbrp/logs/dbftt-coif/best_model.pt"
num_classes = 10
base_model = DBFTT(wavelet_level=2, num_classes=num_classes).to(device)
model = create_model(base_model, mean=NORMALIZATION["CIFAR10"]["mean"], std=NORMALIZATION["CIFAR10"]["std"]).to(device)

ckpt = torch.load(path, weights_only=True)
model.load_state_dict(ckpt["model"])
model.eval()

criterion = torch.nn.CrossEntropyLoss()

_, _, dataloader = get_data_loader("CIFAR10", "data/CIFAR10", 2, 128, val_split_ratio=0.9)

grads_high = torch.tensor([], device=device)
grads_low = torch.tensor([], device=device)

for data, label in tqdm(dataloader, total=len(dataloader)):
    data, label = data.to(device), label.to(device)
    tensor_to_image(data[0]).save("image.png")
    data.requires_grad_(True)

    # 1️⃣ Wavelet decomposition
    x_high, x_low = base_model._DBFTT__decompose_high_low_domains(data)

    tensor_to_image(x_high[0]).save("high-dec.png")
    tensor_to_image(x_low[0]).save("low-dec.png")

    # 2️⃣ Forward through each branch separately
    out_high = base_model.high_model(x_high)
    out_low = base_model.low_conv(base_model.low_model(x_low))
    pooled_high = base_model.avgpool(out_high).view(out_high.size(0), -1)
    pooled_low = base_model.avgpool(out_low).view(out_low.size(0), -1)

    logits_high = base_model.fc(pooled_high)
    logits_low = base_model.fc(pooled_low)

    # 3️⃣ Compute losses per branch
    loss_high = criterion(logits_high, label)
    loss_low = criterion(logits_low, label)

    # 4️⃣ Compute Jacobians separately
    grad_low = torch.autograd.grad(loss_low, [data], retain_graph=True)[0].detach()
    grad_high = torch.autograd.grad(loss_high, [data], retain_graph=True)[0].detach()

    grads_high = torch.concat([grads_high, grad_high], dim=0)
    grads_low = torch.concat([grads_low, grad_low], dim=0)
    # print("Grad high:", grads_high.shape)
    # print("Grad low:", grads_low.shape)
    # break

# g_high = grad_to_image(grads_high.mean(dim=0))
# g_low = grad_to_image(grads_low.mean(dim=0))
g_high = grads_high.mean(dim=0)
g_low = grads_low.mean(dim=0)

print(g_high.shape, g_low.shape)
grad_to_pca_image(g_high, "h.png", title="High branch")
grad_to_pca_image(g_low, "l.png", title="Low branch")
grad_to_pca_image_pair(g_high, g_low, "gradients.png", "High/Low gradients")
# g_high.save("im_h.png")
# g_low.save("im_l.png")
