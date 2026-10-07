"""Ukur latensi inferensi (batch 1) beberapa model pada perangkat ini.
    python latency.py
"""
import time
import numpy as np
import torch
from torchvision import models

dev = "cuda" if torch.cuda.is_available() else "cpu"
cands = {
    "MobileNetV3-Small": models.mobilenet_v3_small,
    "MobileNetV3-Large": models.mobilenet_v3_large,
    "ResNet-18": models.resnet18,
}
x = torch.randn(1, 3, 224, 224).to(dev)
print("device:", dev)
print(f"{'model':20s} {'mean ms':>8s} {'p95 ms':>8s} {'FPS':>7s}")
for name, fn in cands.items():
    m = fn(weights=None).to(dev).eval()
    with torch.no_grad():
        for _ in range(10):
            m(x)
        ts = []
        for _ in range(100):
            if dev == "cuda": torch.cuda.synchronize()
            t = time.perf_counter(); m(x)
            if dev == "cuda": torch.cuda.synchronize()
            ts.append((time.perf_counter() - t) * 1000)
    print(f"{name:20s} {np.mean(ts):8.1f} {np.percentile(ts, 95):8.1f} {1000/np.mean(ts):7.1f}")
print("\nIngat: anggaran 15 FPS = ~67 ms per frame untuk SELURUH pipeline.")
