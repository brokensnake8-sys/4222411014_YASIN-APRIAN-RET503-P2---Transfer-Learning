"""Cek kelayakan dataset_raw + metadata.csv sebelum dikumpulkan.

    python cek_dataset.py
Yang dicek: >= 50 citra/kelas, kondisi cahaya per kelas, file vs metadata cocok,
format nama file, dan ukuran citra seragam.
"""
import os, re, sys
from collections import Counter
import cv2
import pandas as pd

RAW, MIN_PER_KELAS = "dataset_raw", 50
meta_path = f"{RAW}/metadata.csv"
if not os.path.exists(meta_path):
    sys.exit(f"{meta_path} tidak ditemukan.")

df = pd.read_csv(meta_path)
kolom = ["nama_file", "kelas", "tanggal", "kondisi_cahaya"]
if list(df.columns) != kolom:
    print("PERINGATAN: kolom metadata =", list(df.columns), "| seharusnya", kolom)

kelas_folder = sorted(d for d in os.listdir(RAW) if os.path.isdir(f"{RAW}/{d}"))
masalah, sizes = [], Counter()
print(f"{'kelas':14s} {'file':>5s} {'metadata':>9s}  kondisi cahaya                status")
print("-" * 78)
for k in kelas_folder:
    files = sorted(f for f in os.listdir(f"{RAW}/{k}") if f.lower().endswith((".png", ".jpg", ".jpeg")))
    rows = df[df["kelas"] == k]
    kond = rows["kondisi_cahaya"].value_counts().to_dict()

    hilang = [f for f in rows["nama_file"] if f not in files]
    yatim = [f for f in files if f not in set(rows["nama_file"])]
    if hilang: masalah.append(f"{k}: {len(hilang)} baris metadata tanpa file (mis. {hilang[0]})")
    if yatim:  masalah.append(f"{k}: {len(yatim)} file tidak ada di metadata (mis. {yatim[0]}), tidak akan dipakai split.py")

    pola = re.compile(r"^(\d{8})_([a-z0-9]+)_([a-z0-9]+)_(\d{3})\.(png|jpg|jpeg)$")
    for _, r in rows.iterrows():
        nm = str(r["nama_file"])
        if not nm.startswith(k + "_"):
            masalah.append(f"{k}: nama tidak diawali nama kelas: {nm}"); break
        m = pola.match(nm[len(k) + 1:])
        if not m:
            masalah.append(f"{k}: format nama tidak sesuai: {nm}"); break
        if m.group(3) != str(r["kondisi_cahaya"]) or m.group(1) != str(r["tanggal"]):
            masalah.append(f"{k}: nama file tidak cocok dengan metadata: {nm}"); break

    for f in files:
        im = cv2.imread(f"{RAW}/{k}/{f}")
        if im is None:
            masalah.append(f"{k}: file rusak/tidak terbaca: {f}")
        else:
            sizes[(im.shape[1], im.shape[0])] += 1

    n = len(files)
    status = "OK" if n >= MIN_PER_KELAS and len(kond) >= 2 else (
        f"KURANG {MIN_PER_KELAS - n} foto" if n < MIN_PER_KELAS else "hanya 1 kondisi cahaya")
    print(f"{k:14s} {n:5d} {len(rows):9d}  {str(kond):28s}  {status}")

extra = set(df["kelas"]) - set(kelas_folder)
if extra:
    masalah.append(f"kelas di metadata tanpa folder: {sorted(extra)}")
print("\nUkuran citra (lebar x tinggi):", dict(sizes))
if len(sizes) > 1:
    masalah.append("ukuran citra tidak seragam (kamera/resolusi berbeda?)")
print("\nMASALAH:" if masalah else "\nTidak ada masalah ditemukan.")
for m in masalah:
    print(" -", m)
