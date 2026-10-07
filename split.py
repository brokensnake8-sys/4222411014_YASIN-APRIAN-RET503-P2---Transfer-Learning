"""Bagi dataset_raw -> dataset/train & dataset/val BERDASARKAN SESI
(tanggal + kondisi_cahaya) supaya tidak ada data leakage (slide 22).

    python split.py                # val ~20%
"""
import argparse, os, random, shutil
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--raw", default="dataset_raw")
ap.add_argument("--out", default="dataset")
ap.add_argument("--val", type=float, default=0.2)
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()
random.seed(a.seed)

df = pd.read_csv(f"{a.raw}/metadata.csv").drop_duplicates("nama_file")
ada = df.apply(lambda r: os.path.exists(f"{a.raw}/{r['kelas']}/{r['nama_file']}"), axis=1)
if (~ada).any():
    print(f"PERINGATAN: {int((~ada).sum())} baris metadata.csv menunjuk ke file yang sudah "
          "tidak ada (terhapus/dipindah). Baris itu diabaikan dan metadata dibersihkan.")
    df = df[ada]
    df.to_csv(f"{a.raw}/metadata.csv", index=False)
df["sesi"] = df["tanggal"].astype(str) + "_" + df["kondisi_cahaya"].astype(str)

if os.path.exists(a.out):
    shutil.rmtree(a.out)

for kelas, g in df.groupby("kelas"):
    sesi = sorted(g["sesi"].unique())
    random.shuffle(sesi)
    if len(sesi) >= 2:
        n_val = max(1, round(len(sesi) * a.val))
        val_sesi = set(sesi[:n_val])
        is_val = g["sesi"].isin(val_sesi)
        print(f"{kelas}: {len(sesi)} sesi -> val: {sorted(val_sesi)}")
    else:
        # cuma 1 sesi: potong berurutan (bukan acak) agar frame tetangga
        # tidak tersebar di train & val. Tetap lebih baik tambah sesi/kondisi!
        g = g.sort_values("nama_file")
        cut = int(len(g) * (1 - a.val))
        is_val = pd.Series([False] * cut + [True] * (len(g) - cut), index=g.index)
        print(f"PERINGATAN {kelas}: hanya 1 sesi, split berurutan. "
              f"Ambil data di kondisi cahaya/hari lain.")
    for split, sub in (("train", g[~is_val]), ("val", g[is_val])):
        os.makedirs(f"{a.out}/{split}/{kelas}", exist_ok=True)
        for fn in sub["nama_file"]:
            shutil.copy(f"{a.raw}/{kelas}/{fn}", f"{a.out}/{split}/{kelas}/{fn}")
    print(f"   train={int((~is_val).sum())}  val={int(is_val.sum())}")