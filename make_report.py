"""Buat TABEL HASIL 3 mode (markdown) + GRAFIK akurasi per epoch dari hasil train.py.

    python make_report.py
Keluaran: runs/tabel_hasil.md  (tempel ke README)
          runs/akurasi_detail.png  (grafik train vs val per epoch, 3 mode)
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

modes = [m for m in ("feature", "partial", "scratch") if os.path.exists(f"runs/{m}/history.csv")]
if not modes:
    raise SystemExit("runs/<mode>/history.csv tidak ditemukan. Jalankan 'python train.py' dulu.")

summ = pd.read_csv("runs/summary.csv").set_index("mode")
hist = {m: pd.read_csv(f"runs/{m}/history.csv") for m in modes}
nama = {"feature": "Feature extraction (backbone beku)",
        "partial": "Fine-tuning parsial (blok akhir)",
        "scratch": "Scratch (tanpa pretrained)"}

rows = ["| Mode | Pendekatan | Akurasi val terbaik | Akurasi train akhir | Waktu latih (dtk) | Epoch pertama val >= 90% |",
        "|---|---|---|---|---|---|"]
for m in modes:
    s, h = summ.loc[m], hist[m]
    ep90 = "tidak tercapai" if pd.isna(s["epoch_acc_90"]) else str(int(s["epoch_acc_90"]))
    rows.append(f"| {m} | {nama[m]} | {s['best_val_acc']*100:.2f}% | "
                f"{h['train_acc'].iloc[-1]*100:.2f}% | {s['train_time_s']:.0f} | {ep90} |")
tabel = "\n".join(rows)
open("runs/tabel_hasil.md", "w", encoding="utf-8").write(tabel + "\n")
print(tabel)

fig, ax = plt.subplots(1, len(modes), figsize=(5 * len(modes), 3.8), sharey=True, squeeze=False)
for a, m in zip(ax[0], modes):
    h = hist[m]
    a.plot(h["epoch"], h["train_acc"], marker="o", label="train")
    a.plot(h["epoch"], h["val_acc"], marker="s", label="val")
    a.set_title(m); a.set_xlabel("epoch"); a.set_ylim(0, 1.05); a.grid(alpha=.3)
ax[0][0].set_ylabel("akurasi"); ax[0][0].legend(loc="lower right")
fig.suptitle("Akurasi per epoch: MobileNetV3-Large")
fig.tight_layout(); fig.savefig("runs/akurasi_detail.png", dpi=150)
print("\nGrafik: runs/akurasi_detail.png  (dan runs/akurasi.png dari train.py)")