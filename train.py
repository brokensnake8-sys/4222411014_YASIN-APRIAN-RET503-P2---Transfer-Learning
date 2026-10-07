"""Latih MobileNetV3-Large dengan 3 pendekatan: feature | partial | scratch.

    python train.py                          # jalankan ketiganya berurutan
    python train.py --mode feature --epochs 5

Keluaran: runs/<mode>/best.pt, history.csv, ringkasan runs/summary.csv, runs/akurasi.png
"""
import argparse, json, os, time
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from common import build_model, set_train_mode, build_optimizer, train_tf, eval_tf


def run(mode, a, dev, tr, tl, vl):
    torch.manual_seed(0)
    m = build_model(len(tr.classes), mode).to(dev)
    opt = build_optimizer(m, mode)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    lossf = nn.CrossEntropyLoss()
    hist, best, t_total, ep90 = [], 0.0, 0.0, None
    os.makedirs(f"runs/{mode}", exist_ok=True)

    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        set_train_mode(m, mode)
        tl_sum = n = ok = 0
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            out = m(x)
            loss = lossf(out, y)
            loss.backward()
            opt.step()
            tl_sum += loss.item() * len(y); n += len(y)
            ok += (out.argmax(1) == y).sum().item()
        sch.step()

        m.eval(); vn = vok = 0
        with torch.no_grad():
            for x, y in vl:
                x, y = x.to(dev), y.to(dev)
                vok += (m(x).argmax(1) == y).sum().item(); vn += len(y)
        vacc = vok / vn
        dt = time.time() - t0; t_total += dt
        if vacc >= 0.9 and ep90 is None:
            ep90 = ep
        if vacc > best:
            best = vacc
            torch.save(m.state_dict(), f"runs/{mode}/best.pt")
        hist.append(dict(epoch=ep, train_loss=tl_sum / n, train_acc=ok / n, val_acc=vacc))
        print(f"[{mode}] ep{ep:02d} loss={tl_sum/n:.3f} train={ok/n:.3f} val={vacc:.3f} ({dt:.0f}s)")

    pd.DataFrame(hist).to_csv(f"runs/{mode}/history.csv", index=False)
    return hist, dict(mode=mode, best_val_acc=round(best, 4),
                      train_time_s=round(t_total, 1), epoch_acc_90=ep90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all", choices=["all", "feature", "partial", "scratch"])
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--workers", type=int, default=0,
                    help="0 = paling aman di Windows (dataset kecil, tidak perlu worker)")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr = datasets.ImageFolder(f"{a.data}/train", train_tf)
    va = datasets.ImageFolder(f"{a.data}/val", eval_tf)
    tl = DataLoader(tr, a.bs, shuffle=True, num_workers=a.workers)
    vl = DataLoader(va, a.bs, num_workers=a.workers)
    os.makedirs("runs", exist_ok=True)
    json.dump(tr.classes, open("runs/classes.json", "w"))
    print("Kelas:", tr.classes, "| device:", dev)

    modes = ["feature", "partial", "scratch"] if a.mode == "all" else [a.mode]
    summ, hists = [], {}
    for md in modes:
        hists[md], s = run(md, a, dev, tr, tl, vl)
        summ.append(s)

    df = pd.DataFrame(summ)
    df.to_csv("runs/summary.csv", index=False)
    print("\n", df.to_string(index=False))

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    for md, h in hists.items():
        plt.plot([r["epoch"] for r in h], [r["val_acc"] for r in h], marker="o", label=md)
    plt.xlabel("epoch"); plt.ylabel("akurasi validasi"); plt.legend(); plt.grid(alpha=.3)
    plt.title("MobileNetV3-Large: feature vs partial vs scratch")
    plt.tight_layout(); plt.savefig("runs/akurasi.png", dpi=150)
    print("Grafik tersimpan: runs/akurasi.png")


if __name__ == "__main__":      # wajib di Windows agar tidak memicu proses ganda
    main()