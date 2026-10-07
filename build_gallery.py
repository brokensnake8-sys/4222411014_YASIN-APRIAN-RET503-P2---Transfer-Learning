"""Bangun 'galeri' untuk menolak benda asing (open-set): hasilnya label BUKAN.

Idea: tiap kelas komponen punya 'pusat' (prototype) di ruang fitur model.
Gambar baru yang jauh dari pusat kelas tebakannya -> BUKAN, walau belum pernah
diajarkan seperti apa benda itu (laptop, tangan, dll).

    python build_gallery.py --weights runs/feature/best.pt
Jalankan SETELAH train.py. Ulangi tiap kali model/dataset berubah.
"""
import argparse, json, os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from common import build_model, eval_tf, embed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/feature/best.pt")
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--classes", default="runs/classes.json")
    ap.add_argument("--pct", type=float, default=5.0,
                    help="persentil kemiripan terendah yang masih dianggap anggota kelas")
    ap.add_argument("--slack", type=float, default=0.03,
                    help="kelonggaran ambang (besar = lebih mudah menerima)")
    a = ap.parse_args()

    classes = json.load(open(a.classes))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = build_model(len(classes), "scratch")
    m.load_state_dict(torch.load(a.weights, map_location=dev))
    m.to(dev).eval()

    embs, labels = [], []
    for split in ("train", "val"):
        ds = datasets.ImageFolder(f"{a.data}/{split}", eval_tf)
        if ds.classes != classes:
            raise SystemExit(f"Kelas di {split} {ds.classes} != {classes}. Jalankan split.py & train.py ulang.")
        for x, y in DataLoader(ds, 32, num_workers=0):
            with torch.no_grad():
                e = F.normalize(embed(m, x.to(dev)), dim=1)
            embs.append(e.cpu().numpy()); labels.append(y.numpy())
    E, L = np.concatenate(embs), np.concatenate(labels)

    protos, thr = [], []
    print(f"{'kelas':20s} {'n':>5s} {'sim rata2':>10s} {'ambang':>8s}")
    for c, name in enumerate(classes):
        Ec = E[L == c]
        p = Ec.mean(0); p /= np.linalg.norm(p)
        s = Ec @ p
        t = float(np.percentile(s, a.pct) - a.slack)
        protos.append(p); thr.append(t)
        print(f"{name:20s} {len(Ec):5d} {s.mean():10.3f} {t:8.3f}")

    out = os.path.join(os.path.dirname(a.weights), "gallery.npz")
    np.savez(out, protos=np.stack(protos).astype(np.float32),
             thr=np.array(thr, dtype=np.float32), classes=np.array(classes))
    print("Tersimpan:", out)


if __name__ == "__main__":
    main()
