"""Konfigurasi & preprocessing bersama. Dipakai train.py, latency.py, dan predict.py
supaya preprocessing pelatihan == deployment (slide 13)."""
import torch.nn as nn
from torchvision import models, transforms

IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def build_model(num_classes, mode="feature"):
    """MobileNetV3-Large. mode: feature | partial | scratch"""
    if mode == "scratch":
        m = models.mobilenet_v3_large(weights=None)
    else:
        m = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)

    # head baru: classifier[3] adalah Linear(1280, 1000) bawaan ImageNet
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)

    if mode == "feature":          # backbone beku, hanya head dilatih
        for p in m.features.parameters():
            p.requires_grad = False
    elif mode == "partial":        # beku blok awal, buka blok akhir (features[13:])
        for p in m.features[:13].parameters():
            p.requires_grad = False
    return m


def set_train_mode(m, mode):
    """Lapisan beku tetap eval() agar statistik BatchNorm ImageNet tidak berubah."""
    m.train()
    if mode == "feature":
        m.features.eval()
    elif mode == "partial":
        m.features[:13].eval()


def build_optimizer(m, mode):
    import torch
    if mode == "feature":
        return torch.optim.Adam(m.classifier.parameters(), lr=1e-3)
    if mode == "partial":
        return torch.optim.Adam([
            {"params": m.features[13:].parameters(), "lr": 1e-4},
            {"params": m.classifier.parameters(), "lr": 1e-3},
        ])
    return torch.optim.Adam(m.parameters(), lr=1e-3)   # scratch


def embed(m, x):
    """Vektor fitur 960-d dari MobileNetV3 (sebelum classifier)."""
    import torch
    return torch.flatten(m.avgpool(m.features(x)), 1)